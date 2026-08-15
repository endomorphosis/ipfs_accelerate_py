# MCP++ Profiles A–H Inventory (MCPP-011)

**Schema:** `ProfileInventory@1`  
**Task:** MCPP-011  
**Generated:** 2026-08-15  
**Authority basis:** Mcp-Plus-Plus nested checkout bound by MCPP-001 (`repository-forest.json`)  
**Forest reference:** `docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json`  
**Program branch:** `codex/mcplusplus-1.0-gap-closure`

## 0. How to read this inventory

### 0.1 Classification vocabulary

Status values are **exactly** one of:

| Status | Meaning |
| --- | --- |
| `implemented` | Behavior exists and is enforced at the named conformance level with positive and negative evidence. Schema field presence alone never qualifies. |
| `partial` | Meaningful implementation exists, but gaps remain relative to the draft/normative claims (missing negative vectors, optional crypto, incomplete cross-runtime parity, etc.). |
| `structural-only` | Validators/schemas accept shapes (fields, types, regex CID form) without enforcing the security/policy/crypto semantics the profile claims. |
| `missing` | No readable implementation, schema, or runtime for the claim was found in the forest. |
| `blocked` | Implementation exists only behind unavailable external deps, flags, or unresolvable authority (recorded as such). |

Conformance levels (plan KD-6), for cross-reference only:

`structural` → `canonical` → `cryptographic` → `policy-enforced` → `receipt-signed` → `proof-verified`.

### 0.2 Evidence roots (bound SHAs from forest)

| Checkout | Path (operator) | HEAD (MCPP-001 forest) |
| --- | --- | --- |
| Mcp-Plus-Plus (spec authority) | `/home/barberb/lift_coding/Mcp-Plus-Plus` | `6965f89f066769f3b3ac7b5f753b1a0044562570` |
| Nested `ipfs_accelerate_py/mcplusplus` | worktree gitlink | same as Mcp-Plus-Plus `6965f89f…` |
| accelerate | `/home/barberb/lift_coding/external/ipfs_accelerate` | `ea11293bb996f052d620eae989f5377a956764b1` |
| datasets | `/home/barberb/lift_coding/external/ipfs_datasets` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` |
| kit | `/home/barberb/lift_coding/external/ipfs_kit` | `6196017ca3df016c7159dce43af60f2a0d96a9ae` |
| SwissKnife | `/home/barberb/lift_coding/swissknife` | `afdbf885175fde34505ef05a2ea6aac5535ad03e` |

Workspace paths used for this inventory are the gap-closure program worktree copies of the nested trees (`ipfs_accelerate_py/mcplusplus`, `ipfs_datasets_py`, `ipfs_kit_py`) plus the operator SwissKnife path above.

### 0.3 Global findings (pre-summary)

1. **Registry is draft.** Top-level registry `ipfs_accelerate_py/mcplusplus/docs/spec/mcp++-profiles-draft.md` is **Draft (Non-Normative / Discussion)**. Chapters A–F/H are **Draft**; Profile G chapter (`risk-scheduling.md`) is **Draft (Mostly Non-Normative)**.
2. **Versioned JSON Schema exists only for Profile H** under `schemas/profile-h/1.0/`. Other profiles use hand-written per-language models/codecs.
3. **Four-language validators** (Python / TypeScript / Go / Rust under `tests-{py,ts,go,rs}`) are the Mcp-Plus-Plus conformance surface. For Profiles C and D they are **structural field-presence checks**, not cryptographic or policy-enforced conformance.
4. **Runtime code is real and multi-repo.** accelerate, datasets, kit, and SwissKnife each carry profile-shaped modules that must be adapted, not rewritten.
5. **Capability negotiation is still pinned to legacy MCP `initialize` / `2024-11-05`** (`conformance/vectors/initialize_result.json`). Official MCP **2026-07-28** is a separate binding problem (MCPP-010), not a profile implementation claim.
6. **“100% coverage / validation complete / production-ready” documents are contradictory evidence and are listed in §10.** Line coverage of structural validators is not cryptographic or production conformance.
7. **Profile C cryptographic enforcement is not structural-only overall:** real verifiers were found (kit `UCANVerifier`, SwissKnife `@ucans/ucans`, accelerate optional Ed25519). The *Mcp-Plus-Plus four-language validators* remain structural-only for C.

---

## 1. Executive status matrix

| Profile | Capability key (draft) | Spec status | Overall status | Dominant conformance level today | Crypto / policy claim |
| --- | --- | --- | --- | --- | --- |
| **A** MCP-IDL | `mcp++/idl` (via experimental keys; methods `interfaces/*`) | Draft | `partial` | structural + partial canonical | No crypto required; CID helper is approximate in validators |
| **B** CID-native artifacts | (envelope fields; no single key in registry body) | Draft | `partial` | structural (validators); partial canonical in runtimes | Receipt `signature` presence only in validators |
| **C** UCAN delegation | (UCAN chain; no single registry key) | Draft | `partial` | **validators structural-only; runtimes cryptographic (kit/SwissKnife)** | **Real verifiers found** → not structural-only for crypto enforcement |
| **D** Temporal deontic policy | (policy_cid / decision_cid) | Draft | `partial` | structural (validators); partial policy-enforced (accelerate/datasets/SwissKnife) | Policy engines exist; validator is structural-only |
| **E** `mcp+p2p` transport | `mcp++/p2p-transport` | Draft | `partial` | structural + partial runtime framing | Carriage-only; mixes MCP initialize language |
| **F** Event DAG + compaction/ZK | `mcp++/event-dag` | Draft (+ Groth16 chapter) | `partial` | structural (validators); ZK often simulated | Real ZK path opt-in/flagged; simulated proofs are not proof-verified |
| **G** Risk / neighborhood / scheduling | (not fully consolidated) | Draft mostly non-normative; codecs are stricter | `partial` | structural + canonical codec (G/H style) | Signature fields accepted; no crypto verify in codec |
| **H** x402 payments | `mcp++/x402-payments` | Draft interoperability candidate 1.0 | `partial` | structural + canonical codec + versioned schema | Codec explicitly **not** crypto; payment ≠ authorization is specified |

---

## 2. Profile A — MCP-IDL (CID-addressed interface contracts)

### 2.1 Normative (draft chapter claims)

| Item | Path / location |
| --- | --- |
| Registry section | `docs/spec/mcp++-profiles-draft.md` §4 |
| Chapter | `docs/spec/mcp-idl.md` (**Status: Draft**) |
| Required descriptor fields | `name`, `namespace`, `version`, `methods[]`, `errors[]`, `compatibility`, `requires[]` |
| Repository APIs | `interfaces/list`, `interfaces/get(interface_cid)`, `interfaces/compat(interface_cid)` |
| Optional | `interfaces/select(task_hint_cid, budget)`, streaming/event semantics |

### 2.2 Guidance / non-normative

- Historical CORBA / AOP analogy in `mcp-idl.md` §1.1.
- Toolset slicing under context budgets.

### 2.3 Implemented (evidence)

| Surface | Evidence | Status |
| --- | --- | --- |
| Python validator | `tests-py/validators/mcp_idl.py` — required fields, method shape, rough `compute_interface_cid` | `partial` (canonical CID is not Kubo-byte-identical; comment admits simplification) |
| TS / Go / Rust validators | `tests-ts/src/validators/mcpIDL.ts`, `tests-go/validators/mcp_idl.go`, `tests-rs/src/validators/` | `partial` structural parity |
| accelerate runtime | `mcp_server/mcplusplus/idl_registry.py`, `mcplusplus_module/interface_descriptor.py` | `partial` |
| SwissKnife | `src/services/mcp-idl.ts`, tests under `test/mcp-plus-plus/mcp-idl.test.ts`; CONFORMANCE_MATRIX claims PASS | `partial` (self-asserted PASS; not gap-closure admitted) |

### 2.4 Structural-only

- Four-language validators treat presence/type of descriptor fields as success.
- `compute_interface_cid` in Python validator uses sorted-JSON SHA-256 then a non-standard `bafy…` string assembly — **not** proof of Kubo CIDv1 dag-json/dag-pb identity.

### 2.5 Cryptographic

- Not required for Profile A contracts. **N/A** (content addressing is integrity, not authority crypto).

### 2.6 Runtime-specific

| Runtime | Notes |
| --- | --- |
| accelerate | IDL registry + interface descriptor modules |
| SwissKnife | Full InterfaceRepository + CLI (`idl list/get/compat`) |
| datasets / kit | No dedicated Profile A IDL registry module found at inventory time |

### 2.7 Missing

- Versioned JSON Schema family for Interface Descriptor (only Profile H has versioned schemas).
- Cross-language golden vectors proving identical `interface_cid` for the same canonical bytes across Py/TS/Go/Rust and Kubo.
- Official MCP 2026-07-28 binding for IDL discovery without legacy `initialize` (binding track, not IDL core).

### 2.8 Contradictory docs (A-related)

- Coverage docs claim 100% of `mcp_idl.py` lines; that is **line coverage of a structural validator**, not interface-repository production conformance. See §10.

**Profile A overall:** `partial`.

---

## 3. Profile B — CID-native execution artifacts

### 3.1 Normative (draft)

| Item | Path |
| --- | --- |
| Registry | `docs/spec/mcp++-profiles-draft.md` §5 |
| Chapter | `docs/spec/cid-native-artifacts.md` (**Draft**) |
| Envelope fields | `interface_cid`, `input_cid`, optional `intent_cid` / `policy_cid` / `proof_cid`, `parents[]` |
| Outputs | `output_cid`, `receipt_cid`; receipts MAY be signed |
| Canonicalization | MUST be deterministic; CIDs MUST match Kubo for same canonical bytes (chapter claim) |

### 3.2 Guidance

- Provenance / audit / credit-assignment narrative in chapter §1–2.

### 3.3 Implemented

| Surface | Evidence | Status |
| --- | --- | --- |
| Python validator | `tests-py/validators/cid_artifacts.py` — field presence + CID regex | `structural-only` for security; `partial` for format |
| Conformance vectors | `conformance/vectors/execution_receipt.json` | `partial` |
| accelerate | `mcp_server/mcplusplus/artifacts.py`, `mcplusplus_module/cid_ucan.py` (Intent/Decision/Receipt/Envelope), `kubo_cid.py` | `partial` |
| kit | `mcp_server/mcplusplus/artifacts.py` | `partial` |
| SwissKnife | `src/services/mcp-envelope.ts` (+ tests) | `partial` |

### 3.4 Structural-only

- Validator marks `signed=True` if a `signature` **field exists**; it does **not** verify signatures.
- CID check is a base32/Qm regex, not multihash/codec validation of the linked content.

### 3.5 Cryptographic

- Receipt signatures: **structural-only** in Mcp-Plus-Plus validators.
- Runtime signing paths exist (SwissKnife optional signature flow; accelerate artifact modules) but are not elevated to forest-wide `receipt-signed` conformance without shared vectors.

### 3.6 Runtime-specific

| Runtime | Notes |
| --- | --- |
| accelerate | CID helpers + artifact store modules |
| kit | Artifact helpers used with UCAN admission |
| SwissKnife | Envelope/receipt builders with CID determinism tests |
| datasets | Uses envelopes indirectly via workflow/P2P; no dedicated B codec package found |

### 3.7 Missing

- Shared `ExecutionEnvelope@1` / `ExecutionResult@1` / `ExecutionReceipt@1` family (planned MCPP envelope work).
- Versioned JSON Schema for B (none under `schemas/` except H).
- Four-language adapter proving historical B CIDs unchanged (MCPP-031).

### 3.8 Contradictory docs

- 100% coverage of `cid_artifacts.py` does not imply CID-native production readiness. See §10.

**Profile B overall:** `partial`.

---

## 4. Profile C — Capability delegation (UCAN)

### 4.1 Normative (draft)

| Item | Path |
| --- | --- |
| Registry | `docs/spec/mcp++-profiles-draft.md` §6 |
| Chapter | `docs/spec/ucan-delegation.md` (**Draft**) |
| Claims | Delegation chains; **execution-time validation REQUIRED**; invocations reference valid proof; receipts bind outcomes |

### 4.2 Guidance

- Multi-hop User → Planner → Worker → Tool narrative in chapter.

### 4.3 Implemented (cryptographic runtimes — real verifiers found)

Acceptance rule for this task: *Profile C cryptographic enforcement is structural-only **unless a real verifier is found**.*

**Real verifiers found:**

| Verifier | Path | Behavior | Status |
| --- | --- | --- | --- |
| **kit `UCANVerifier`** | `ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/ucan.py` | Compact EdDSA envelopes, capability attenuation, audience/time windows, durable `RevocationLedger`, fail-closed without ledger | `implemented` (kit runtime cryptographic) |
| kit tests | `ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_ucan_verifier.py` | Tampered / unsigned / alg-none / wrong key / audience / expiry negatives | `implemented` evidence |
| **SwissKnife `@ucans/ucans`** | `swissknife/src/services/mcp/mcp-plus-plus-profile-c.ts`, `src/auth/ucan-auth.ts` | `ucans.verify`, did:key Ed25519, revocation registry, DelegationManager | `partial`–`implemented` (runtime; not four-language conformance) |
| accelerate optional Ed25519 | `ipfs_accelerate_py/mcp_server/mcplusplus/delegation.py` (`verify_delegation_signature_ed25519`, `require_signatures=`) | Real Ed25519 verify when `require_signatures=True` and cryptography available; default path allows non-crypto HMAC-style tokens | `partial` |
| accelerate module | `mcplusplus_module/cid_ucan.py` | Optional `nacl` verify; HMAC fallback if keys/nacl missing | `partial` |

### 4.4 Structural-only (Mcp-Plus-Plus conformance validators)

| Language | Path | What it checks | What it does **not** check |
| --- | --- | --- | --- |
| Python | `tests-py/validators/ucan_delegation.py` | `iss`, `aud`, `att`, `exp` present; `att` is list; `proof_cid` present on invocation | signatures, attenuation subset, audience continuity, expiry vs clock, revocation |
| TypeScript | `tests-ts/src/validators/ucanDelegation.ts` | Zod schema parse of token/chain | crypto verify |
| Go | `tests-go/validators/ucan_delegation.go` | struct tags + non-empty capabilities | crypto verify |
| Rust | `tests-rs/src/validators/ucan_delegation.rs` | field structure (fixtures use did:key strings) | crypto verify |

**Therefore:** Mcp-Plus-Plus **conformance validators** for Profile C cryptographic enforcement are **`structural-only`**. Forest-wide **cryptographic enforcement** is **`partial`** (real kit/SwissKnife verifiers; accelerate opt-in; no single shared four-language crypto API).

### 4.5 Cryptographic (summary classification)

| Layer | Classification |
| --- | --- |
| Four-language Mcp-Plus-Plus validators | **`structural-only`** |
| kit admission path | **`implemented`** (cryptographic + revocation) |
| SwissKnife auth path | **`partial` / strong** (library-backed verify + tests) |
| accelerate unified dispatch | **`partial`** (crypto optional) |
| **Profile C cryptographic enforcement (overall)** | **`partial` — not structural-only (real verifiers found)** |

### 4.6 Runtime-specific

| Runtime | Module(s) |
| --- | --- |
| kit | `ucan.py`, `revocation.py`, `delegation.py`, readiness tests |
| accelerate | `mcp_server/mcplusplus/delegation.py`, `mcplusplus_module/cid_ucan.py`, MCP tests `test_mcp_server_mcplusplus_ucan.py` |
| SwissKnife | `src/auth/ucan-auth.ts`, `src/auth/delegation-manager.ts`, profile-c tests |
| datasets | No dedicated UCAN verifier module inventoried; may consume accelerate/kit patterns |

### 4.7 Missing

- Single normative crypto API + shared adversarial vectors in all four Mcp-Plus-Plus languages (MCPP crypto track).
- Default-on signature requirement in accelerate validators/dispatch.
- Versioned JSON Schema for UCAN tokens in Mcp-Plus-Plus `schemas/`.
- Replacement of structural validator “success” for invalid signatures (planned: keep fixtures, change expected results).

### 4.8 Contradictory docs

- Python `ucan_delegation.py` at “100% line coverage” with only ~30 lines of field checks — coverage docs treat this as complete Profile C validation. **False as cryptographic conformance.** See §10.
- SwissKnife `docs/mcp-plus-plus/CONFORMANCE_MATRIX.md` marks Profile C **PASS** for the SwissKnife tree only; that is not a four-language or Mcp-Plus-Plus production admission.

**Profile C overall:** `partial` (validators structural-only; crypto enforcement partial with real verifiers in kit/SwissKnife).

---

## 5. Profile D — Temporal deontic policy evaluation

### 5.1 Normative (draft)

| Item | Path |
| --- | --- |
| Registry | `docs/spec/mcp++-profiles-draft.md` §7 |
| Chapter | `docs/spec/temporal-deontic-policy.md` (**Draft**) |
| Representation | content-addressed `policy_cid`; permissions / prohibitions / obligations / temporal constraints |
| Runtime | validate delegation proofs; evaluate policy; emit `decision_cid`; obligations MAY have deadlines |

### 5.2 Guidance

- Prompt→Delegation and Intent→Decision→Receipt placement narrative.

### 5.3 Implemented

| Surface | Evidence | Status |
| --- | --- | --- |
| Python validator | `tests-py/validators/policy_evaluation.py` | **`structural-only`** (type enum + field presence; temporal ISO checks are `pass`) |
| accelerate engine | `mcp_server/mcplusplus/policy_engine.py` — `evaluate_policy` / obligations | `partial` policy-enforced |
| datasets | `ipfs_datasets_py/logic/profile_d_policy.py` — package-export evaluator; ZKP certificate is **statement request**, not a proof | `partial` |
| SwissKnife | `src/services/mcp-policy.ts`, deontic broker, remote TDFOL hooks | `partial` |
| Vector | `conformance/vectors/policy_decision.json` | structural sample |

### 5.4 Structural-only

- Four-language policy validators: field shape only.
- Temporal constraint parsing intentionally stubbed in Python validator.

### 5.5 Cryptographic

- Profile D does not require UCAN crypto by itself, but **draft** says execution-time must validate delegation proofs first → depends on Profile C.
- datasets `zkp_certificate` for D is explicitly **not** a verified ZK proof (`profile_d_policy.py` header).

### 5.6 Runtime-specific

| Runtime | Notes |
| --- | --- |
| accelerate | Local clause engine + optional datasets bridge (`evaluate_with_ipfs_datasets_policy`) |
| datasets | Canonical `profile_d_policy` export for HTTP/libp2p gates |
| SwissKnife | Richer deontic UI / ORB / remote engine surface |
| kit | Authorization dispatch gates exist (`test_authorization_dispatch_gate.py`); not a full deontic language |

### 5.7 Missing

- Shared six-obligation event suite and fail-closed datasets wiring (MCPP-049+).
- Adversarial negatives for stale / revoked / conflicting policies in four-language validators.
- Versioned policy JSON Schema.

### 5.8 Contradictory docs

- 100% coverage of `policy_evaluation.py` (26 lines) presented as complete policy validation. See §10.

**Profile D overall:** `partial` (runtime engines exist; Mcp-Plus-Plus validators structural-only).

---

## 6. Profile E — `mcp+p2p` transport binding

### 6.1 Normative (draft)

| Item | Path |
| --- | --- |
| Registry | `docs/spec/mcp++-profiles-draft.md` §8 |
| Chapter | `docs/spec/transport-mcp-p2p.md` (**Draft**) |
| Scope | Carriage of MCP JSON-RPC over libp2p; does not redefine MCP methods |
| Registry also | `mcp++/p2p-transport` experimental key (appendix) |

### 6.2 Guidance

- NAT traversal / relays as non-mandatory deployment notes.
- Non-goals: global consensus; replacing client↔server transports.

### 6.3 Implemented

| Surface | Evidence | Status |
| --- | --- | --- |
| Python validator | `tests-py/validators/transport.py` — protocol ids, frame length field, session phases, JSON-RPC field preservation | `structural-only` / `partial` |
| accelerate | `mcp_server/mcplusplus/p2p_framing.py`, peer_* modules, `mcplusplus_module/p2p*` | `partial` |
| datasets | `mcp_server/mcp_p2p_transport.py`, `p2p_libp2p_transport.py`, workflow P2P tools | `partial` |
| SwissKnife | `src/services/mcp-p2p-session.ts`, pubsub bus, rate limits | `partial` |
| Vector | `conformance/vectors/p2p_message.json`, `bus_message.json` | samples |

### 6.4 Structural-only

- Validator checks dictionary keys for frames/sessions; not wire adversarial limits (oversize, replay window, flood) as a complete suite in the four-language validators alone.
- Draft still couples transport discussion to MCP `initialize` / `2024-11-05` (registry §3), which conflicts with official MCP 2026-07-28 lifecycle (binding gap).

### 6.5 Cryptographic

- Transport security is peer identity / libp2p layer; not Profile C. Peer-identity UCAN checks appear in SwissKnife Profile C module (cross-profile).

### 6.6 Runtime-specific

| Runtime | Notes |
| --- | --- |
| datasets | Heaviest P2P runtime surface |
| accelerate | Framing + peer registry/bootstrap |
| SwissKnife | Session state machine + PubSubBus |
| kit | Transport security parity tests under runtime_readiness |

### 6.7 Missing

- Split normative text: transport negotiation vs MCP application semantics vs execution authority (MCPP-062).
- Shared abuse vectors (oversize, replay, flood, peerId/UCAN) bound into datasets/kit (MCPP-063–065).
- Versioned framing schema.

### 6.8 Contradictory docs

- Coverage “complete” docs for `transport.py` ≠ hardened P2P production admission. See §10.
- SwissKnife CONFORMANCE_MATRIX Profile E **PASS** is SwissKnife-local.

**Profile E overall:** `partial`.

---

## 7. Profile F — Event DAG provenance, archival, and compaction

### 7.1 Normative (draft)

| Item | Path |
| --- | --- |
| Registry | `docs/spec/mcp++-profiles-draft.md` §9 (detailed F text including ZK extension) |
| Chapter | `docs/spec/event-dag-ordering.md` (**Draft**) |
| Capability key | `mcp++/event-dag` |
| Event commits | intent, interface, proofs, decision, outputs, parents |
| Compaction certificate | `certificate_cid`, `archive_cid`, `merkle_root`, …; `zero_knowledge` true **only** with real verifiable ZK |
| ZK ops | `mcp++/dag/zk/{status,prove,verify}`; fail closed if keys unavailable |
| Groth16 ceremony chapter | `docs/spec/groth16-mpc-ceremony.md` (**Draft**) |

### 7.2 Guidance

- Hot / archive / compact tiers; audit traversal story.

### 7.3 Implemented

| Surface | Evidence | Status |
| --- | --- | --- |
| Python validator | `tests-py/validators/event_dag.py` — event_cid/timestamp/parents; DAG list checks | `structural-only` |
| accelerate | `mcp_server/mcplusplus/event_dag.py`, `mcplusplus_module/dag_compaction.py` | `partial` |
| kit | `mcp_server/mcplusplus/event_dag.py` | `partial` |
| Fixture | `tests-py/fixtures/valid/profile_f_groth16_mpc_ceremony.json` | structural ceremony shape |
| Vectors | `dag_event_epoch.json`, `dag_event_iso.json`, `zkp_proof_artifact.json`, `wasm_proof_result.json` | mixed structural / proof-shaped |

### 7.4 Structural-only

- Event validator does not verify Merkle inclusion or ZK proofs.
- Presence of Groth16-shaped JSON is not `proof-verified`.

### 7.5 Cryptographic / proof-verified

| Claim | Finding | Status |
| --- | --- | --- |
| Simulated Groth16 | `dag_compaction.py` defaults to `proof_type: simulated_groth16` | **not** proof-verified; must set `zero_knowledge: false` per draft |
| Real Groth16 | opt-in via `IPFS_DATASETS_ENABLE_GROTH16=1` path | `partial` / often `blocked` without backend |
| Registry rule | hash/Merkle/signature/simulated digest MUST NOT claim `zero_knowledge: true` | normative text present; enforcement uneven |

### 7.6 Runtime-specific

| Runtime | Notes |
| --- | --- |
| accelerate | Compaction + event DAG modules |
| kit | Event DAG helpers |
| SwissKnife | Circuit id `event_dag_compaction_v1` referenced in draft |
| datasets | Groth16 enable flag coupling |

### 7.7 Missing

- Always-on archive verifier that recomputes commitments + verifies Groth16 + Merkle before accepting certificates.
- Four-language proof-verify conformance suite.
- Production admission of any ZK path (plan: F material is not Verified Execution until real verifier path is proven).

### 7.8 Contradictory docs

- Any “validation complete” claim that equates event field tests with ZK-verified compaction. See §10.

**Profile F overall:** `partial` (structure + some runtime; ZK mostly simulated / opt-in).

---

## 8. Profile G — Risk scoring, neighborhood coordination, scheduling

### 8.1 Normative vs guidance

| Item | Path | Notes |
| --- | --- | --- |
| Registry links | `mcp++-profiles-draft.md` §10–11 | Scheduling / risk largely **non-normative** in registry |
| Chapter | `docs/spec/risk-scheduling.md` | **Draft (Mostly Non-Normative)** |
| Codecs (stricter than chapter status) | `tests-py/validators/profile_g.py` (+ TS/Go/Rust codec tests) | Canonical DAG-JSON fields, CID rules, size limits |
| Vectors | `profile_g_{artifacts,protocol}_{valid,invalid}.json`, `profile_g_three_peer.json` | Stronger than chapter prose |
| Harness docs | `docs/testing/profile-g-three-peer-conformance.md`, release/performance runbooks | operational |

**Tension:** chapter is mostly non-normative, but codecs and three-peer harness behave as if a wire profile exists. Consolidation into one normative Profile G is planned (MCPP-G130 / MCPP-066+).

### 8.2 Implemented

| Surface | Evidence | Status |
| --- | --- | --- |
| Canonical codec | `profile_g.py` field sets for Goal…TaskReceipt; CIDv1 sha2-256 checks; size limits | `partial` (canonical + structural) |
| accelerate | `risk_scheduler.py`, `profile_g_transport.py`, workflow/task queue | `partial` |
| kit | `profile_g_transport.py`, `coordination_storage.py` | `partial` |
| SwissKnife | `test/mcp-plus-plus/profile-g-connector.test.ts` | `partial` |
| Integration | `tests-py/integration/test_profile_g_three_peer.py`, codec tests multi-language | `partial` |

### 8.3 Structural-only

- Codec requires `signature` / `signature_alg` **strings** on RiskEvidence / NeighborhoodRecord / NeighborhoodAttestation — **does not verify** signatures.
- Neighborhood agreement is **coordination / majority approval**, **not BFT** (plan KD-11). No code path should label G results as BFT.

### 8.4 Cryptographic

- Signature fields: **structural-only** in codecs.
- No four-language cryptographic verifier for neighborhood attestations inventoried.

### 8.5 Runtime-specific

| Runtime | Notes |
| --- | --- |
| accelerate | Risk scheduler + G transport + three-peer harness (Python) |
| kit | Coordination storage + G transport |
| SwissKnife | Connector tests |
| datasets | Scheduler/queue overlap via workflow engines; not a pure G codec owner |

### 8.6 Missing

- One normative Profile G specification reconciling registry, `risk-scheduling.md`, codecs, and harnesses.
- Stale fenced completion rejection as forest-wide guarantee (MCPP-069).
- Versioned JSON Schema for G artifacts.
- Crypto verification for signed neighborhood records.

### 8.7 Contradictory docs

- Marketing language that “neighborhood consensus” implies BFT or global consensus (chapter itself says coordination optimization, not consensus requirement — preserve that).
- Coverage-complete docs that predate Profile G codecs still claim “all validators” complete without G/H.

**Profile G overall:** `partial`.

---

## 9. Profile H — x402 payments and paid capability access

### 9.1 Normative

| Item | Path |
| --- | --- |
| Registry | `mcp++-profiles-draft.md` (Profile H link + appendix) |
| Chapter | `docs/spec/x402-payments.md` (**Draft interoperability candidate 1.0**) |
| Profile key | `mcp++/x402-payments` |
| Versioned schemas | `schemas/profile-h/1.0/{artifacts,common,x402-v2}.schema.json` |
| Generated docs | `docs/generated/profile-h-schemas-1.0.md` |
| Codec | `tests-py/validators/profile_h.py` (+ TS `profileH.ts`) |
| Vectors | `profile_h_artifacts_valid.json`, `profile_h_invalid.json`, `profile_h_transport_valid.json` |

**Critical normative boundary:** payment authorization is **payment authority only**; it **MUST NOT** grant execution identity/capability. Protected work requires independent Profile C/D (and related) checks.

### 9.2 Guidance

- Upstream x402 v2 HTTP header mapping; commercial catalog UX notes.

### 9.3 Implemented

| Surface | Evidence | Status |
| --- | --- | --- |
| Versioned JSON Schema | `schemas/profile-h/1.0/*` | `partial`–strongest schema story of any profile |
| Canonical codec | `profile_h.py` — DAG-JSON, CID, amount canonicity; **explicitly no wallet/crypto verify** | `partial` (canonical + structural) |
| accelerate | `mcp_server/mcplusplus/profile_h.py` — seller dispatch separates payment from effect | `partial` |
| datasets | `mcp_server/mcplusplus/profile_h.py` | `partial` |
| kit | `profile_h.py`, `profile_h_http.py`, `tests/mcplusplus_profile_h/` | `partial` |
| SwissKnife | `test/mcp-plus-plus/profile-h-adapter.test.ts` | `partial` |

### 9.4 Structural-only

- Codec enforces wire bounds and field sets; seller-side signature verification is **out of codec scope** (module docstring).
- Schema acceptance of payment fields is not settlement or authorization success.

### 9.5 Cryptographic

- **Structural-only / deferred to seller runtime** for payment signatures.
- Must not be confused with Profile C execution crypto.

### 9.6 Runtime-specific

| Runtime | Notes |
| --- | --- |
| accelerate | Seller runtime + payment ledger hooks; payment fence before effect |
| kit | HTTP + paid kit tests |
| datasets | profile_h module present |
| SwissKnife | Adapter tests |

### 9.7 Missing

- Adversarial suite proving payment success never authorizes execution when C/D deny (MCPP-070–072).
- Full upstream x402 HTTP v2 interop certification (separate from MCP++ H claim).
- Four-language complete H codec parity (Python/TS strong; Go/Rust less complete than G/H Python).

### 9.8 Contradictory docs

- None claiming H is 100% cryptographically complete were treated as authoritative; H codec honestly disclaims crypto. Still, broader “validation complete” docs omit H or predate it — listed in §10.

**Profile H overall:** `partial` (best schema/codec maturity; payment≠authorization needs adversarial hardening).

---

## 10. Contradictory “100 percent / validation complete” documents

These documents claim complete, perfect, or production-ready validator coverage. They are **not** accepted as cryptographic, policy-enforced, or production conformance evidence for Profiles A–H. They contradict the structural nature of C/D validators and each other on percentages and dates.

### 10.1 Mcp-Plus-Plus testing docs (primary list)

| Path | Claim pattern | Why contradictory |
| --- | --- | --- |
| `ipfs_accelerate_py/mcplusplus/docs/testing/FINAL_100_PERCENT_COVERAGE_SUMMARY.md` | Python/TS/Rust **100%**, Go **97.6%**, “Mission Complete”, production-ready | Line coverage ≠ crypto/policy; C/D validators are field checks |
| `ipfs_accelerate_py/mcplusplus/docs/testing/FINAL_COVERAGE_ACHIEVEMENT.md` | “100% Mission Complete”, PRODUCTION-READY | Same |
| `ipfs_accelerate_py/mcplusplus/docs/testing/VERIFICATION_COMPLETE.md` | Coverage 100%, pass rate 100% | Same |
| `ipfs_accelerate_py/mcplusplus/docs/testing/MULTI_LANGUAGE_VALIDATION_COMPLETE.md` | “COMPLETE ✅”, Python 100% reference | Lists only pre-G/H core validators; omits profile_g/h maturity |
| `ipfs_accelerate_py/mcplusplus/docs/testing/VALIDATION_TESTING_COMPLETE.md` | “All validator tests… production-ready”; also cites Python **90%** in same family of docs | **Internal contradiction** vs 100% siblings |
| `ipfs_accelerate_py/mcplusplus/docs/testing/VALIDATION_TESTING_SUMMARY.md` | Validation complete narrative | Coverage ≠ conformance levels |
| `ipfs_accelerate_py/mcplusplus/docs/testing/VALIDATION_STATUS_SUMMARY.md` | Status complete framing | Same |
| `ipfs_accelerate_py/mcplusplus/docs/testing/VALIDATOR_TESTING_FINAL_STATUS.md` | Final status complete | Same |
| `ipfs_accelerate_py/mcplusplus/docs/testing/CURRENT_COVERAGE_STATUS.md` | Snapshot that drifts from “final 100%” peers | **Cross-doc percentage contradiction** |
| `ipfs_accelerate_py/mcplusplus/docs/testing/COVERAGE_ROADMAP_TO_100_PERCENT.md` | Roadmap to 100% while finals claim already done | **Process vs outcome contradiction** |
| `ipfs_accelerate_py/mcplusplus/docs/testing/TESTING_SUMMARY.md` | Summary complete tone | Same family |
| `ipfs_accelerate_py/mcplusplus/docs/testing/README.md` | Indexes coverage achievement as complete | Points readers at contradictory finals |
| `ipfs_accelerate_py/mcplusplus/docs/testing/FINAL_VERIFICATION.txt` | Verification complete text | Same |

### 10.2 Per-language coverage trophies

| Path | Claim pattern |
| --- | --- |
| `ipfs_accelerate_py/mcplusplus/tests-py/COVERAGE_100_PERCENT.md` | 720/720 lines, MISSION ACCOMPLISHED, PRODUCTION-READY |
| `ipfs_accelerate_py/mcplusplus/tests-py/VERIFICATION.md` | Verification of 100% narrative |
| `ipfs_accelerate_py/mcplusplus/tests-py/README.md` | 100% framing |
| `ipfs_accelerate_py/mcplusplus/tests-ts/README.md` | “100% Test Coverage” |
| `ipfs_accelerate_py/mcplusplus/tests-rs/COVERAGE_100_PERCENT_ACHIEVED.md` | 100% achieved |
| `ipfs_accelerate_py/mcplusplus/tests-rs/COVERAGE_VERIFICATION.txt` | Verification of 100% |
| `ipfs_accelerate_py/mcplusplus/tests-rs/coverage_100_percent_final.txt` | Final 100% dump |
| `ipfs_accelerate_py/mcplusplus/tests-go/validators/COVERAGE_89_6_PERCENT_FINAL.md` | “Functional **100%**” at 89.6% statements |
| `ipfs_accelerate_py/mcplusplus/tests-go/validators/GO_VALIDATORS_COVERAGE_REPORT.md` | Functional 100% / complete coverage language |
| `ipfs_accelerate_py/mcplusplus/tests-go/validators/FINAL_COVERAGE_PUSH_SUMMARY.md` | 89.6% = functional 100% |

### 10.3 Cleanup / meta docs that still preserve the contradiction

| Path | Notes |
| --- | --- |
| `ipfs_accelerate_py/mcplusplus/DOCUMENTATION_CLEANUP_SUMMARY.md` | Discusses supersession of intermediate coverage reports while keeping 100%/89.6% “authoritative” language |
| `ipfs_accelerate_py/mcplusplus/REORGANIZATION_SUMMARY.md` | Organizes coverage reports as if complete |

### 10.4 Cross-repo self-PASS matrices (not Mcp-Plus-Plus authority)

| Path | Claim | Treatment |
| --- | --- | --- |
| `/home/barberb/lift_coding/swissknife/docs/mcp-plus-plus/CONFORMANCE_MATRIX.md` | Profiles A–E **PASS** for SwissKnife | Runtime-local claim; **not** gap-closure production admission; does not prove four-language crypto |

### 10.5 Inventory rule for later tasks

- **Do not** promote any requirement to `implemented` solely because a document in §10 says 100% or COMPLETE.
- Recompute tests; map to conformance levels in the traceability matrix (MCPP-012).
- Structural validators with full line coverage remain `structural-only` at higher levels.

---

## 11. Cross-cutting gaps (all profiles)

| Gap | Impacted profiles | Notes |
| --- | --- | --- |
| Draft / non-normative registry | A–H | No profile is sealed normative for 1.0 admission |
| Only H has versioned JSON Schema | A–G | Hand-written multi-language drift risk |
| Legacy MCP `initialize` / `2024-11-05` | negotiation for all | Conflicts with official MCP 2026-07-28 |
| Structural C/D validators | C, D | Field presence ≠ security |
| Signature fields without verify | B, C (validators), G, H (codec) | Crypto level incomplete except kit/SwissKnife C |
| Simulated ZK | F | Not proof-verified |
| Profile G chapter vs codec strength | G | Needs normative consolidation |
| Payment vs authorization | H | Spec clear; adversarial hardening remaining |
| Contradictory coverage docs | all | §10 |

---

## 12. Per-profile classification checklist (required splits)

For each profile, the split demanded by MCPP-011 effects:

| Profile | Normative sources | Guidance | Implemented (selected) | Structural-only | Cryptographic | Runtime-specific | Missing | Contradictory docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | profiles-draft §4, mcp-idl.md | CORBA notes | idl validators, accelerate registry, SwissKnife IDL | field validators; pseudo-CID | N/A integrity-only | accelerate, SwissKnife | versioned schema; Kubo-identical CIDs | §10 coverage |
| B | profiles-draft §5, cid-native-artifacts.md | provenance narrative | artifacts modules, vectors | cid_artifacts signature flag | receipt verify incomplete | accelerate, kit, SwissKnife | Envelope@1 family | §10 |
| C | profiles-draft §6, ucan-delegation.md | multi-hop story | **kit UCANVerifier**, SwissKnife ucans, accelerate opt-in | **4-lang validators** | **partial overall (real verifiers found)** | kit, SwissKnife, accelerate | shared 4-lang crypto API | §10 + SwissKnife PASS matrix |
| D | profiles-draft §7, temporal-deontic-policy.md | dual placement story | accelerate/datasets/SwissKnife engines | policy_evaluation validators | ZKP statement ≠ proof | accelerate, datasets, SwissKnife | adversarial suite; schema | §10 |
| E | profiles-draft §8, transport-mcp-p2p.md | NAT/relay notes | framing modules multi-repo | transport.py shape checks | peer-id via C elsewhere | datasets, accelerate, SwissKnife, kit | abuse suite; MCP bind split | §10 + SwissKnife E PASS |
| F | profiles-draft §9, event-dag-ordering.md, groth16 chapter | tier narrative | event_dag modules | event_dag validator | simulated vs real ZK | accelerate, kit, SwissKnife circuits | always-on verify path | §10 |
| G | risk-scheduling.md (weak), codecs (strong) | LSH/risk heuristics | profile_g codec + runtimes | signature strings | no attest verify | accelerate, kit, SwissKnife | normative consolidation | consensus wording + §10 |
| H | x402-payments.md, schemas/profile-h/1.0 | commercial UX | schema+codec+runtimes | pre-crypto codec | seller-side deferred | accelerate, kit, datasets, SwissKnife | payment≠auth adversarial | pre-H “all validators complete” docs |

---

## 13. Acceptance mapping (MCPP-011)

| Acceptance criterion | Result |
| --- | --- |
| Profile C cryptographic enforcement classified **structural-only unless a real verifier is found** | **Real verifiers found** (`ipfs_kit_py/.../ucan.py` `UCANVerifier` + tests; SwissKnife `@ucans/ucans`; accelerate optional Ed25519). Overall crypto enforcement = **`partial`**. Four-language Mcp-Plus-Plus validators remain **`structural-only`**. |
| Contradictory 100-percent coverage docs listed | **§10** enumerates Mcp-Plus-Plus testing trophies, per-language 100%/“functional 100%” reports, cleanup meta-docs, and SwissKnife PASS matrix. |
| Each profile split into normative / guidance / implemented / structural-only / cryptographic / runtime-specific / missing / contradictory | **§2–§9** plus summary **§12**. |
| Schema field presence not treated as `implemented` | Status vocabulary enforced; C/D validators marked structural-only at validator layer. |

---

## 14. Evidence index (quick paths)

```
ipfs_accelerate_py/mcplusplus/docs/spec/mcp++-profiles-draft.md
ipfs_accelerate_py/mcplusplus/docs/spec/{mcp-idl,cid-native-artifacts,ucan-delegation,
  temporal-deontic-policy,transport-mcp-p2p,event-dag-ordering,risk-scheduling,
  x402-payments,groth16-mpc-ceremony}.md
ipfs_accelerate_py/mcplusplus/schemas/profile-h/1.0/
ipfs_accelerate_py/mcplusplus/tests-py/validators/{mcp_idl,cid_artifacts,ucan_delegation,
  policy_evaluation,transport,event_dag,profile_g,profile_h}.py
ipfs_accelerate_py/mcp_server/mcplusplus/{idl_registry,artifacts,delegation,policy_engine,
  p2p_framing,event_dag,risk_scheduler,profile_g_transport,profile_h}.py
ipfs_accelerate_py/mcplusplus_module/{interface_descriptor,cid_ucan,p2p_transport,dag_compaction,temporal_policy}.py
ipfs_datasets_py/ipfs_datasets_py/logic/profile_d_policy.py
ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/
ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/{ucan,revocation,delegation,event_dag,profile_h}.py
ipfs_kit_py/tests/runtime_readiness/mcplusplus/test_ucan_verifier.py
/home/barberb/lift_coding/swissknife/docs/mcp-plus-plus/CONFORMANCE_MATRIX.md
/home/barberb/lift_coding/swissknife/src/services/mcp/mcp-plus-plus-profile-c.ts
/home/barberb/lift_coding/swissknife/src/auth/ucan-auth.ts
docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json
```

---

## 15. Downstream consumers

- **MCPP-012** traceability matrix must import these statuses; no row may upgrade to `implemented` from §10 coverage claims alone.
- Crypto lane should treat kit `UCANVerifier` as the strongest existing Profile C reference implementation while replacing structural four-language validators.
- Profile G/H lanes should preserve codec vectors and payment≠authorization boundary respectively.

**End of ProfileInventory@1 for MCPP-011.**
