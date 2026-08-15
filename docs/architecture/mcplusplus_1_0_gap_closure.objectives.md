# MCP++ 1.0 Gap-Closure Objective Heap

This is the durable goal/subgoal hierarchy for the `MCPP-` program. The
normative design is `MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md`; executable work
is projected into `mcplusplus_1_0_gap_closure.todo.md`.

Program invariants:

- Mcp-Plus-Plus owns normative specs, schemas, vectors, validators, and matrices;
- runtimes own adapters and must not fork protocol bytes;
- schema acceptance is never implementation;
- fail closed for authorization, policy, signature, proof, lease, fencing, payment, and trust boundaries;
- historical artifact CIDs remain readable through versioned adapters;
- official MCP 2026-07-28 and official A2A are verified from primary sources before binding changes;
- existing user overlays are never discarded;
- completion requires current-tree commands and evidence artifacts.

## Goal tree

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

## MCPP-G000 Deliver an independently verifiable MCP++ 1.0 release-candidate architecture

- Status: blocked
- Review only: true
- Parent: 
- Depends on: 
- Fib priority: 1
- Track: release-candidate
- Priority: P0
- Bundle: mcplusplus/1.0/control
- Parallel lane: control
- Resource class: cpu-large
- Goal: Produce a release-candidate MCP++ architecture in which agents can discover capabilities, exchange portable envelopes, delegate attenuated authority, evaluate policy, share explicitly consistent state, recover without duplicate effects, fence exclusive work, speak HTTP and optional P2P, emit signed receipts, reconstruct a bounded Event DAG, interoperate with current MCP and A2A, and demonstrate these properties in a repeatable three-peer environment.
- Subgoals: MCPP-G010, MCPP-G020, MCPP-G030, MCPP-G040, MCPP-G050, MCPP-G060, MCPP-G070, MCPP-G080, MCPP-G090, MCPP-G100, MCPP-G110, MCPP-G120, MCPP-G130, MCPP-G140, MCPP-G150, MCPP-G160, MCPP-G170
- Evidence: MCPP-G010, MCPP-G020, MCPP-G030, MCPP-G040, MCPP-G050, MCPP-G060, MCPP-G070, MCPP-G080, MCPP-G090, MCPP-G100, MCPP-G110, MCPP-G120, MCPP-G130, MCPP-G140, MCPP-G150, MCPP-G160, MCPP-G170
- Evidence criteria: Every child goal has a current-tree evidence bundle and the terminal joined release receipt binds their exact revisions, schema versions, and validation CIDs. No gate in the sealed plan is marked passed without a command and artifact.
- Evidence source policy: Reviewed contracts, schemas, and signed receipts define claims. Tests, traces, and documentation cannot promote support or authorize mutation alone. Schema acceptance is never implementation.
- Outputs: docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md, docs/architecture/mcplusplus_1_0_gap_closure.objectives.md, docs/architecture/mcplusplus_1_0_gap_closure.todo.md, config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json
- Predicted files: docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md, docs/architecture/mcplusplus_1_0_gap_closure.objectives.md, docs/architecture/mcplusplus_1_0_gap_closure.todo.md, config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json
- Interfaces: MCPPlusPlusReleaseReceipt@1, ConformanceLevel@1
- Validation: python scripts/validate_mcplusplus_1_0_gap_closure_board.py --check-all
- Acceptance: All 84 tasks and 17 child goals are terminal; every required acceptance gate has a command and evidence artifact; no user overlay was discarded; draft PRs are unmerged; the report recommendation is NO-GO, RC, or GO with evidence.
- Gap task: Aggregate independently validated child evidence and decide release; do not implement subsystem behavior at the root.
- Refinement: Prefer an explicit experimental, unsupported, blocked, or approval-required disposition over an unearned production claim.
- Embedding query: mcp plus plus release candidate envelope state delegation receipt event dag a2a p2p
- AST query: ExecutionEnvelope StateRef DelegationProof DurableExecutor AgentAdvertisement
- Conflict policy: Root is review and evidence aggregation only; child goals own implementation and the terminal task owns the immutable joined receipt.

## MCPP-G010 Establish a current-tree baseline, inventory, and traceability matrix

- Status: active
- Parent: MCPP-G000
- Depends on: 
- Fib priority: 1
- Track: baseline
- Priority: P0
- Bundle: mcplusplus/1.0/baseline
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-medium
- Goal: Preserve existing user work, create program branches without reset, run every existing Python, TypeScript, Go, and Rust suite plus runtime MCP++ tests, verify official MCP 2026-07-28 and A2A sources, inventory Profiles A-H, and create the requirement-to-evidence matrix.
- Evidence: MCPP-001, MCPP-002, MCPP-003, MCPP-004, MCPP-005, MCPP-006, MCPP-007, MCPP-008, MCPP-009, MCPP-010, MCPP-011, MCPP-012
- Evidence criteria: Exact commands, environment, SHAs, pass/fail, counts, coverage, skips, and external dependencies are recorded. Requirements are classified implemented, partial, structural-only, missing, or blocked.
- Evidence source policy: Current process output is inventory authority. Static coverage documents are observations, not evidence.
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline, ipfs_accelerate_py/mcplusplus/docs/roadmap/mcplusplus-1.0-gap-closure.md
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline, ipfs_accelerate_py/mcplusplus/docs/roadmap/mcplusplus-1.0-gap-closure.md
- Interfaces: BaselineReceipt@1, RequirementTraceRow@1, OfficialSpecNote@1
- Validation: python -m pytest -q test/api/test_mcplusplus_1_0_gap_closure_board.py
- Acceptance: Dirty overlays are inventoried and untouched; branches exist; every in-scope suite has a dated receipt; official MCP 2026-07-28 and A2A notes cite primary URLs; the matrix refuses to mark schema-only rows implemented.
- Gap task: Measure what exists before changing protocol bytes.
- Refinement: A green historical coverage markdown file is not a current-tree baseline.
- Embedding query: baseline inventory traceability mcp 2026-07-28 a2a profile a-h test matrix
- AST query: InitializeResult UCANDelegationValidator ProfileG ProfileH
- Conflict policy: Own only new evidence and roadmap files; do not refactor validators or runtimes.

## MCPP-G020 Record binding architecture decisions and proceed without waiting for product choice

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G010
- Fib priority: 1
- Track: adr
- Priority: P0
- Bundle: mcplusplus/1.0/adr
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-small
- Goal: Write ADRs for spec-versus-runtime ownership, the mandatory crypto suite and canonicalization version, conformance levels, state modes and CRDT library, durable-executor choice, and the A2A extension identifier plus dual MCP bindings.
- Evidence: MCPP-013, MCPP-014, MCPP-015, MCPP-016, MCPP-017, MCPP-018
- Evidence criteria: Each ADR is Accepted, cites current-tree evidence, names rejected alternatives, and is specific enough to implement without further product questions.
- Evidence source policy: The sealed plan Key Decisions are the defaults. An ADR may refine them with evidence but may not reopen a choice already decided without a new risk.
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions, docs/architecture/decisions
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions
- Interfaces: ArchitectureDecisionRecord@1
- Validation: test -d ipfs_accelerate_py/mcplusplus/docs/architecture/decisions
- Acceptance: Six ADRs exist covering KD-1 through KD-13; Restate/Dapr evaluation is recorded; A2A URI default is confirmed or replaced with the verified official convention.
- Gap task: Freeze defaults so parallel lanes do not invent incompatible suites.
- Refinement: Do not ask the operator to choose among reasonable engineering defaults.
- Embedding query: architecture decision record ed25519 jcs automerge restate a2a extension
- AST query: ConformanceLevel CanonicalizationId DurableExecutor
- Conflict policy: ADRs are new files only; they do not rewrite sealed plan identity.

## MCPP-G030 Separate abstract MCP++ semantics from protocol-version bindings

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G020
- Fib priority: 2
- Track: mcp-bindings
- Priority: P0
- Bundle: mcplusplus/1.0/bindings
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-medium
- Goal: Refactor Profiles A-H so they are transport- and MCP-version-independent, keep the 2024-11-05 initialize binding as an explicitly named legacy version, add a current 2026-07-28 binding that does not use the removed initialize exchange, and prove dual-binding peers reject version downgrade and forgery.
- Evidence: MCPP-019, MCPP-020, MCPP-021, MCPP-022, MCPP-023
- Evidence criteria: Legacy clients still work; current clients do not need initialize; a peer can advertise both; forged or downgraded protocolVersion is rejected deterministically.
- Evidence source policy: Official MCP 2026-07-28 text is binding for the current module. Historical initialize behavior is allowed only under the legacy binding name.
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/bindings, ipfs_accelerate_py/mcp_server/mcplusplus/bindings, ipfs_datasets_py/mcp_server/mcplusplus
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/bindings, ipfs_accelerate_py/mcp_server/mcplusplus/bindings
- Interfaces: McpBindingLegacy20241105@1, McpBinding20260728@1, ProfileCapabilityAdvertisement@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k binding
- Acceptance: Profile specs no longer require initialize as current behavior; both bindings have positive and negative tests; accelerate and datasets adapters speak the declared binding without ambiguity.
- Gap task: Stop encoding one historical MCP lifecycle into abstract profile semantics.
- Refinement: libp2p negotiation is a transport handshake, not an MCP application initialize.
- Embedding query: mcp binding 2026-07-28 legacy 2024-11-05 initialize tasks discovery
- AST query: InitializeParams protocolVersion Tasks extension
- Conflict policy: Binding modules are new versioned files; existing 2024-11-05 vectors remain readable.

## MCPP-G040 Normalize canonicalization and prevent cross-language schema drift

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G020
- Fib priority: 1
- Track: canonicalization
- Priority: P0
- Bundle: mcplusplus/1.0/canonical
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Goal: Specify mcpp-jcs-v1 as RFC 8785 JCS, keep historical artifact CIDs readable, publish golden vectors with canonical bytes, SHA-256, CID, and signature input, implement them identically in Python, TypeScript, Go, and Rust, and generate schemas from one canonical source.
- Evidence: MCPP-024, MCPP-025, MCPP-026, MCPP-027, MCPP-028, MCPP-029
- Evidence criteria: The same object yields identical canonical bytes, digest, and CID in all four languages. Existing Profile B and G vectors still verify under their recorded algorithm.
- Evidence source policy: Canonical bytes are computed, not described. Language-specific pretty printers are not authority.
- Outputs: ipfs_accelerate_py/mcplusplus/schemas, ipfs_accelerate_py/mcplusplus/conformance/vectors, ipfs_accelerate_py/mcplusplus/tests-py, ipfs_accelerate_py/mcplusplus/tests-ts, ipfs_accelerate_py/mcplusplus/tests-go, ipfs_accelerate_py/mcplusplus/tests-rs
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas, ipfs_accelerate_py/mcplusplus/conformance/vectors
- Interfaces: McppJcsV1@1, GoldenVector@1, SchemaId@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_conformance_vectors.py
- Acceptance: mcpp-jcs-v1 is named; migration path exists; four-language identity tests pass; schema generation has one source; numeric, Unicode, null, and duplicate-key negatives exist.
- Gap task: Four hand-written models currently drift and existing CID bytes must not silently change.
- Refinement: If JCS cannot represent a needed value, document the exclusion rather than invent a silent fork.
- Embedding query: rfc 8785 jcs canonical json cid sha256 cross language golden vector
- AST query: canonical_dumps cidv1 sha2_256 mcpp-jcs-v1
- Conflict policy: New algorithm id and vectors; do not rewrite historical vector CIDs.

## MCPP-G050 Create one canonical portable execution envelope

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G040
- Fib priority: 2
- Track: envelope
- Priority: P0
- Bundle: mcplusplus/1.0/envelope
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Goal: Define ExecutionEnvelope@1, ExecutionResult@1, ExecutionReceipt@1, and PortableError@1 covering identity, authority, intent, inputs/outputs, constraints, state, policy, and provenance, with adapters from existing Profile B and G objects.
- Evidence: MCPP-030, MCPP-031, MCPP-032, MCPP-033, MCPP-034
- Evidence criteria: Existing valid B and G artifacts remain readable. New envelopes validate structurally and canonically. Runtime accelerate adapter emits the new carrier.
- Evidence source policy: The envelope is the top-level carrier; Profiles B, C, D, F, and G remain modular contents or references.
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/execution, ipfs_accelerate_py/mcplusplus/docs/spec/execution-envelope.md, ipfs_accelerate_py/mcp_server/mcplusplus
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/execution, ipfs_accelerate_py/mcplusplus/docs/spec/execution-envelope.md
- Interfaces: ExecutionEnvelope@1, ExecutionResult@1, ExecutionReceipt@1, PortableError@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k envelope
- Acceptance: All four schemas exist and are versioned; adapters do not change historical CIDs; four-language vectors pass; accelerate can create and verify an envelope.
- Gap task: Profiles B, C, D, F, and G currently overlap execution semantics without one carrier.
- Refinement: Wall-clock timestamps are informational; causal order comes from parents and logical clocks.
- Embedding query: execution envelope receipt portable error interface_cid intent_cid policy_cid
- AST query: ExecutionEnvelope ExecutionReceipt PortableError
- Conflict policy: New schema family plus adapters; do not mutate Profile B historical fixtures.

## MCPP-G060 Define interoperable shared-state semantics and working providers

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G040
- Fib priority: 2
- Track: state
- Priority: P0
- Bundle: mcplusplus/1.0/state
- Parallel lane: mcpp-lane-state
- Resource class: io-large
- Goal: Specify StateRef@1 with explicit consistency modes and implement immutable CID, durable single-authority SQLite, Automerge CRDT, and an honestly labeled consensus plugin, plus tests that Event DAG branches do not silently merge mutable values.
- Evidence: MCPP-035, MCPP-036, MCPP-037, MCPP-038, MCPP-039, MCPP-040
- Evidence criteria: SQLite restart and CAS tests pass. Automerge concurrent offline, reorder, duplicate, partition-heal, and convergence tests pass. Consensus docs distinguish coordination, majority, crash, and BFT.
- Evidence source policy: Mode is mandatory. Missing mode is invalid. Neighborhood majority is not BFT.
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/state, ipfs_accelerate_py/mcplusplus/docs/spec/state-ref.md, ipfs_accelerate_py/mcp_server/mcplusplus/state
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/state, ipfs_accelerate_py/mcp_server/mcplusplus/state
- Interfaces: StateRef@1, StateProvider@1, ImmutableCidState@1, SqliteAuthorityState@1, AutomergeCrdtState@1, ConsensusPlugin@1
- Validation: cd ipfs_accelerate_py && python -m pytest -q test/api/test_mcplusplus_state_providers.py
- Acceptance: Every StateRef declares one mode; four providers exist; restart, CRDT convergence, and non-merge tests pass; no BFT claim is made without a BFT engine.
- Gap task: Shared state is currently implied by Event DAG parents and Profile G artifacts.
- Refinement: Two concurrent DAG branches are observations, not a merge.
- Embedding query: stateref consistency automerge sqlite cas lease fencing consensus
- AST query: StateRef StateProvider compare_and_swap Automerge
- Conflict policy: New provider package; Profile G artifacts adapt through StateRef rather than growing informal fields.

## MCPP-G070 Complete cryptographic delegation and receipt conformance

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G040
- Fib priority: 1
- Track: crypto
- Priority: P0
- Bundle: mcplusplus/1.0/crypto
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-security
- Goal: Replace structural UCAN field checks with real Ed25519 verification of attenuation, audience, time, depth, replay, revocation, and executor binding, and require signed receipts for cross-trust-domain execution.
- Evidence: MCPP-041, MCPP-042, MCPP-043, MCPP-044, MCPP-045
- Evidence criteria: Negative vectors for forged signature, altered bytes, wrong audience, expanded capabilities or resources, expired, future nbf, revoked, missing proof, replay, wrong executor, wrong policy CID, and valid PeerID with invalid UCAN all fail closed.
- Evidence source policy: Test doubles are forbidden for signature and attenuation checks. Transport identity is never execution authority.
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/validators, ipfs_accelerate_py/mcplusplus/schemas/delegation, ipfs_kit_py/ipfs_kit_py/mcp_server
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/ucan_delegation.py, ipfs_accelerate_py/mcplusplus/schemas/delegation
- Interfaces: DelegationProof@1, RevocationRecord@1, ReceiptVerifier@1, ConformanceLevel@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_ucan_delegation.py
- Acceptance: Cryptographic level is actually implemented; revocation records are signed and discoverable; cross-domain receipts verify in a separate process; SwissKnife and kit call the same vectors.
- Gap task: Current Python UCAN validator only checks iss/aud/att/exp presence.
- Refinement: A schema that accepts a signature field is structural, not cryptographic.
- Embedding query: ed25519 ucan attenuation revocation replay receipt audience executor
- AST query: UCANDelegationValidator verify_signature RevocationRecord
- Conflict policy: Replace structural checks in place with versioned cryptographic APIs; keep fixtures that are now expected to fail at cryptographic level.

## MCPP-G080 Harden temporal deontic policy execution and obligation lifecycle

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G070
- Fib priority: 2
- Track: policy
- Priority: P0
- Bundle: mcplusplus/1.0/policy
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Goal: Implement a deterministic policy evaluator that returns allow, deny, or allow_with_obligations plus decision_cid, and emit obligation lifecycle events with deadline, revocation-before-execution, and compensation tests.
- Evidence: MCPP-046, MCPP-047, MCPP-048, MCPP-049
- Evidence criteria: Authority, policy compliance, and execution receipt remain distinct. Statement commitments are not treated as verified policy proofs.
- Evidence source policy: Missing context, stale root, or version mismatch is deny. Human-approval requirements are obligations, not implicit allow.
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/temporal-deontic-policy.md, ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/policy_evaluation.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus
- Interfaces: PolicyEvaluator@1, PolicyDecision@1, ObligationEvent@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_policy_evaluation.py
- Acceptance: Evaluator interface is implemented; six obligation events exist; datasets Profile D wiring fails closed; conflicting-policy resolution is deterministic and tested.
- Gap task: Policy validators currently check document shape, not obligation lifecycle.
- Refinement: An allow decision that creates an obligation is incomplete until the obligation is satisfied or compensated.
- Embedding query: temporal deontic obligation deadline compensation policy decision_cid
- AST query: PolicyEvaluator obligation_created obligation_violated decision_cid
- Conflict policy: Spec and datasets own evaluator semantics; accelerate may consume decisions but not re-implement policy.

## MCPP-G090 Add a durable execution adapter with crash-recovery evidence

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G050, MCPP-G060
- Fib priority: 3
- Track: durable
- Priority: P0
- Bundle: mcplusplus/1.0/durable
- Parallel lane: mcpp-lane-state
- Resource class: io-large
- Goal: Define DurableExecutor and implement at least one production-capable local adapter that journals externally visible steps, uses idempotency keys, preserves cancellation and obligations, rejects stale fencing tokens, and resumes after process kill without repeating committed side effects.
- Evidence: MCPP-050, MCPP-051, MCPP-052, MCPP-053
- Evidence criteria: Crash-recovery test starts a multi-step task, commits one side effect, kills the process, restarts, resumes, does not repeat the effect, and emits one receipt.
- Evidence source policy: Restate/Dapr may be a second adapter only with a repeatable local environment. The mandatory adapter is the SQLite journaled executor.
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/durable-execution.md, ipfs_accelerate_py/mcp_server/mcplusplus/durable
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/durable, ipfs_accelerate_py/mcplusplus/docs/architecture/durable-execution.md
- Interfaces: DurableExecutor@1, DurableJournalRecord@1, CrashRecoveryReceipt@1
- Validation: cd ipfs_accelerate_py && python -m pytest -q test/api/test_mcplusplus_durable_crash_recovery.py
- Acceptance: Interface covers start/resume/signal/cancel/checkpoint/retry/timer/compensation/inspect/recover/finalize; one adapter passes crash recovery; accelerate consumes it.
- Gap task: MCP++ must not reimplement a full workflow engine, but it must have a stable durable contract.
- Refinement: A successful in-memory retry is not crash recovery.
- Embedding query: durable executor journal idempotency crash recovery fencing lease
- AST query: DurableExecutor checkpoint recover fencing_token
- Conflict policy: New durable package; do not fold journaling into Event DAG validators.

## MCPP-G100 Define a versioned A2A interoperability extension

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G020, MCPP-G050
- Fib priority: 2
- Track: a2a
- Priority: P0
- Bundle: mcplusplus/1.0/a2a
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Goal: Specify and implement an A2A extension that maps Agent Card, Task, Message, Part, Artifact, status, cancel, streaming, and auth onto MCP-IDL and MCP++ envelope/state/receipt objects without creating a competing public task lifecycle.
- Evidence: MCPP-054, MCPP-055, MCPP-056, MCPP-057
- Evidence criteria: Two independently instantiated agents complete a handoff. Cancellation, retry, streaming, malformed-extension, and unsupported-profile tests exist.
- Evidence source policy: Official A2A extension identifiers are URIs advertised on the Agent Card. The working name io.mcplusplus.execution@1 is an alias until MCPP-010 records the verified URI.
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/a2a-extension.md, ipfs_accelerate_py/mcplusplus/schemas/a2a
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/a2a-extension.md, ipfs_accelerate_py/mcplusplus/schemas/a2a
- Interfaces: A2AExecutionExtension@1, AgentCardMapping@1, A2ATaskAdapter@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k a2a
- Acceptance: Extension spec, schemas, vectors, reference adapter, and SwissKnife adapter exist; e2e handoff passes; unsupported profile fails closed.
- Gap task: A2A already provides the public agent-task lifecycle; MCP++ must extend it, not replace it.
- Refinement: Final A2A results must carry receipt_cid, event_cid, and proof references.
- Embedding query: a2a agent card task artifact extension io.mcplusplus.execution
- AST query: AgentCard Task Artifact A2A-Extensions
- Conflict policy: New extension package; do not fork A2A Task status names.

## MCPP-G110 Implement discovery, advertisement, and routing interfaces

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G100
- Fib priority: 3
- Track: discovery
- Priority: P0
- Bundle: mcplusplus/1.0/discovery
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Goal: Extend MCP-IDL with an advertisement schema mappable to A2A Agent Cards, implement a pluggable registry with publish/refresh/withdraw/lookup and deterministic selection, and provide static, libp2p, and isolated AGNTCY adapters.
- Evidence: MCPP-058, MCPP-059, MCPP-060, MCPP-061
- Evidence criteria: Tests cover registration, expiration, withdrawal, duplicates, compromised records, and deterministic provider selection. A registry record is never execution authority.
- Evidence source policy: Stale and unsigned advertisements are rejected. Health and load are selection inputs, not trust.
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/discovery, ipfs_accelerate_py/mcp_server/mcplusplus/registry
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/discovery, ipfs_accelerate_py/mcp_server/mcplusplus/registry
- Interfaces: AgentAdvertisement@1, Registry@1, StaticRegistry@1, Libp2pDiscovery@1
- Validation: cd ipfs_accelerate_py && python -m pytest -q test/api/test_mcplusplus_registry.py
- Acceptance: Advertisement describes identity, endpoints, transports, MCP/A2A/MCP++ versions, interface CIDs, policy/proof/runtime/locality/price/health/TTL/trust/residency; three adapters exist or AGNTCY is isolated with a documented blocker.
- Gap task: Discovery currently lives in initialize experimental flags and ad-hoc P2P bootstrap lists.
- Refinement: Finding an agent is not permission to execute.
- Embedding query: advertisement registry libp2p agntcy interface_cid health ttl
- AST query: AgentAdvertisement Registry lookup_by_interface_cid
- Conflict policy: New registry package; do not make bootstrap peer lists authoritative.

## MCPP-G120 Harden the P2P transport binding

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G030
- Fib priority: 2
- Track: p2p
- Priority: P0
- Bundle: mcplusplus/1.0/p2p
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-security
- Goal: Update Profile E so network negotiation, MCP application semantics, and MCP++ execution authority are separate, and implement versioned libp2p protocol IDs, deterministic length-prefixed framing, quotas, replay detection, and abuse tests.
- Evidence: MCPP-062, MCPP-063, MCPP-064, MCPP-065
- Evidence criteria: Adversarial tests cover oversized/truncated/invalid frames, request-before-negotiation, forged version, unknown method, empty success on transport failure, replay, flood, stream exhaustion, valid PeerID invalid UCAN, stale fence, duplicate response, and wrong correlation id.
- Evidence source policy: Transport success is not application success. Current MCP is sessionless; legacy sessions exist only under the legacy binding.
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/transport-mcp-p2p.md, ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/p2p_libp2p_transport.py, ipfs_kit_py/ipfs_kit_py/mcp_server
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/transport.py, ipfs_accelerate_py/mcplusplus/docs/spec/transport-mcp-p2p.md
- Interfaces: McpP2pBinding@1, LengthPrefixedFrame@1, TransportQuota@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_transport.py
- Acceptance: Spec split is explicit; framing and quotas are implemented; abuse tests pass; datasets and kit adapters use the versioned protocol id.
- Gap task: Profile E currently mixes transport handshake language with MCP initialize.
- Refinement: A valid PeerID with an invalid UCAN is a deny, not a degraded allow.
- Embedding query: libp2p framing backpressure replay quota protocol id peerid ucan
- AST query: LengthPrefixedFrame protocol_id max_frame_size
- Conflict policy: Transport files only; do not change envelope or UCAN bytes here.

## MCPP-G130 Consolidate Profile G into a normative coordination specification

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G050, MCPP-G060
- Fib priority: 2
- Track: profile-g
- Priority: P0
- Bundle: mcplusplus/1.0/profile-g
- Parallel lane: mcpp-lane-state
- Resource class: cpu-large
- Goal: Publish one normative Profile G for TaskSpec, claims, leases, epochs, fencing, receipts, and neighborhood records, and expand the three-peer harness to prove partition healing, stale-fence rejection, and exactly one authoritative completion.
- Evidence: MCPP-066, MCPP-067, MCPP-068, MCPP-069
- Evidence criteria: Simultaneous claims, partitions, duplicates, reordering, restart, expiry, takeover, late/alternate completion, malicious stale publisher, frontier reconciliation, and no policy bypass are tested.
- Evidence source policy: Local coordination is not global consensus. Majority/neighborhood requirements are labeled as such.
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/risk-scheduling.md, ipfs_accelerate_py/mcplusplus/tests-py/harness/profile_g_three_peer.py
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/risk-scheduling.md, ipfs_accelerate_py/mcplusplus/tests-py/harness
- Interfaces: TaskSpec@1, TaskClaim@1, Lease@1, LogicalEpoch@1, FencingToken@1, TaskReceipt@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_g_three_peer.py
- Acceptance: One normative spec; stale fenced completion is rejected; three peers converge after partition heal; exclusive tasks have one authoritative completion.
- Gap task: Profile registry, risk-scheduling.md, validators, harnesses, and runtime copies currently disagree.
- Refinement: A published completion without a current fencing token is not authoritative.
- Embedding query: profile g claim lease epoch fencing three peer partition takeover
- AST query: TaskClaim Lease FencingToken ClaimResolution
- Conflict policy: Normative spec in Mcp-Plus-Plus; kit and accelerate adapt, they do not fork fields.

## MCPP-G140 Preserve and harden Profile H so payment is never authorization

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G070
- Fib priority: 3
- Track: profile-h
- Priority: P0
- Bundle: mcplusplus/1.0/profile-h
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-security
- Goal: Harden quotes, payment authorization, settlement, entitlements, refunds, and signed access receipts so a successful payment never grants execution authority and unpaid authorized work is rejected.
- Evidence: MCPP-070, MCPP-071, MCPP-072
- Evidence criteria: Negative tests cover paid-but-unauthorized, authorized-but-unpaid, replay, price mismatch, wrong recipient, duplicate settlement, expired quote, refund-after-consumed, and forged settlement.
- Evidence source policy: Upstream x402 HTTP conformance and libp2p carriage of equivalent objects remain distinct. Payment objects are entitlements, not UCANs.
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/x402-payments.md, ipfs_accelerate_py/mcplusplus/schemas/profile-h
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/x402-payments.md, ipfs_accelerate_py/mcplusplus/tests-py/validators/profile_h.py
- Interfaces: PaymentQuote@1, SettlementReceipt@1, Entitlement@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_codec.py
- Acceptance: Payment success without delegation is deny; unpaid authorized invocation is deny; SwissKnife and accelerate adapters keep the split; x402 and libp2p mappings are tested separately.
- Gap task: Profile H schemas exist; the authorization boundary is not adversarially tested.
- Refinement: A settled quote is not a capability token.
- Embedding query: x402 payment entitlement settlement refund authorization boundary
- AST query: SettlementReceipt Entitlement UCANDelegationValidator
- Conflict policy: Harden Profile H modules only; do not grant capabilities from payment code.

## MCPP-G150 Make confidential CID-native artifacts safe

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G050
- Fib priority: 3
- Track: confidential
- Priority: P0
- Bundle: mcplusplus/1.0/confidential
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-security
- Goal: Define an encrypted artifact reference and prove that ciphertext is verifiable, plaintext does not leak through logs, Event DAG metadata, or local caches, and key access is capability-controlled.
- Evidence: MCPP-073, MCPP-074
- Evidence criteria: Tests cover unauthorized reads, altered ciphertext, wrong recipient, revoked key access, and accidental plaintext persistence.
- Evidence source policy: Content addressing is not publication. Receipts may attest use of an encrypted artifact without disclosing it.
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/confidential, ipfs_accelerate_py/mcplusplus/docs/spec/cid-native-artifacts.md, ipfs_kit_py/ipfs_kit_py
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/confidential, ipfs_kit_py/ipfs_kit_py
- Interfaces: EncryptedArtifactRef@1, KeyEnvelope@1, DisclosurePolicy@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k confidential
- Acceptance: Encrypted reference schema exists; kit persistence paths do not write plaintext in the tested fallback/cache routes; revocation behavior is documented honestly.
- Gap task: CID-native artifacts currently assume content is publishable.
- Refinement: A protected plaintext digest is optional and must not appear beside raw plaintext.
- Embedding query: encrypted artifact ciphertext cid key envelope disclosure retention
- AST query: EncryptedArtifactRef ciphertext_cid key_envelope
- Conflict policy: New confidential schema and kit persistence hooks; do not change historical public artifact CIDs.

## MCPP-G160 Create a real installation and three-peer demonstration path

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G090, MCPP-G100, MCPP-G130
- Fib priority: 3
- Track: demo
- Priority: P0
- Bundle: mcplusplus/1.0/demo
- Parallel lane: mcpp-lane-runtime
- Resource class: io-large
- Goal: Ship an installable mcpp CLI and a one-command three-peer Compose environment that exercises discovery, A2A handoff, delegation, policy, envelope, state, durable crash, fenced takeover, signed receipt, Event DAG reconstruction, and independent verification.
- Evidence: MCPP-075, MCPP-076, MCPP-077
- Evidence criteria: The demo writes a machine-readable evidence bundle with SHAs, schema versions, CIDs, signatures, and test results. A separate verifier process validates the bundle.
- Evidence source policy: Demo success is not production admission. Missing optional P2P relays are documented blockers, not skipped assertions.
- Outputs: ipfs_accelerate_py/mcplusplus/cli, ipfs_accelerate_py/mcplusplus/demo, docs/reports/mcplusplus-1.0-gap-closure/demo
- Predicted files: ipfs_accelerate_py/mcplusplus/cli, ipfs_accelerate_py/mcplusplus/demo
- Interfaces: McppCli@1, DemoEvidenceBundle@1, IndependentVerifier@1
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m mcpp doctor && python -m mcpp demo --peers 3 --verify
- Acceptance: CLI commands inspect/validate/conformance/envelope/receipt/artifact/peer/demo/doctor exist; compose brings up three peers; verifier exits 0 on the produced bundle and nonzero on tampered copies.
- Gap task: There is no installable CLI or one-command three-peer demonstration today.
- Refinement: A scripted happy-path printout is not an evidence bundle.
- Embedding query: mcpp cli docker compose three peer demo evidence bundle verifier
- AST query: mcpp demo doctor receipt verify
- Conflict policy: CLI and demo are new packages; they consume, not reimplement, profile modules.

## MCPP-G170 Publish honest documentation, CI, and a release recommendation

- Status: active
- Parent: MCPP-G000
- Depends on: MCPP-G160
- Fib priority: 3
- Track: release
- Priority: P0
- Bundle: mcplusplus/1.0/release
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-large
- Goal: Rewrite stale coverage and production claims, add architecture and threat-model docs marked by authority class, run multi-language and security CI, write the implementation report, and open unmerged draft PRs when credentials exist.
- Evidence: MCPP-078, MCPP-079, MCPP-080, MCPP-081, MCPP-082, MCPP-083
- Evidence criteria: Static claims match generated CI artifacts. Required workflows are green or the report is NO-GO with the exact failing command. Draft PRs are not merged.
- Evidence source policy: Forbidden phrases: production-ready, zero vulnerabilities, fully conformant, zero knowledge — unless an automated artifact proves that exact claim.
- Outputs: ipfs_accelerate_py/mcplusplus/README.md, ipfs_accelerate_py/mcplusplus/docs, ipfs_accelerate_py/mcplusplus/docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md, .github/workflows
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md, docs/reports/mcplusplus-1.0-gap-closure
- Interfaces: ImplementationReport@1, CompatibilityMatrix@1, ReleaseRecommendation@1
- Validation: python scripts/validate_mcplusplus_1_0_gap_closure_board.py --check-all
- Acceptance: Docs mark normative/non-normative/experimental/reference/production-admitted; CI covers the required suites; report contains every required section; recommendation is evidenced; user overlays remain intact.
- Gap task: Documentation currently over-claims coverage and still describes initialize as current MCP.
- Refinement: A draft PR URL is optional; a missing permission is a documented blocker, not a silent skip of the report.
- Embedding query: implementation report ci sbom threat model compatibility matrix release candidate
- AST query: MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT ReleaseRecommendation
- Conflict policy: Documentation and CI only after implementation evidence exists; do not rewrite sealed control artifacts.
