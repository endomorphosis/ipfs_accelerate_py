# SCG-004 MCP++ interoperability and proof-sealer inventory

**Evidence:** `scg/mcplusplus-boundary@1`  
**Machine-readable twin:** `docs/architecture/semantic_compression_governor_inventory/interoperability.json`  
**Task:** SCG-004 — Inventory MCP++ shared schemas/vectors and proof-sealer availability  
**Inspected tree:** `8ba3ada72212dcf516adc1ae81d8bc2e563926b9`  
**Planning authority pin:** `dfd92b554e662d4312411f2e8e63a52368806f2a` (ancestor of inspected tree)  
**IVP public-API freeze:** `8c7800cedc5e1b848367db9952f912428466f8cc`  
**Incremental proof-sealer program pin (external):** `7dc8f1422cb7e80757077948dc0785c1aaa4fd25`

This is a read-only inventory of MCP++ shared wire schemas, conformance vectors,
Profile A/B/F harness wiring, existing Profile G scheduling/artifact codecs, and
release-time full-checkpoint / delta (incremental) proof-sealer availability.
No production code is changed. No new MCP++ profile and no local generic
envelope are introduced.

Nested gitlinks observed at inventory time:

| Nested repository | Commit |
| --- | --- |
| `ipfs_datasets_py` | `1330038f626ef92993f03d46f21e1a57719e9c25` |
| `ipfs_kit_py` | `df2f9cc092456329de9724c45a50c54b410875d1` |
| `ipfs_accelerate_py/mcplusplus` | `dc3164653a48d059ae9812078359daeafb451c07` |

---

## 1. Claim boundary

| Surface | Status on inspected tree | Notes |
| --- | --- | --- |
| Profile A (MCP-IDL) | **available** | Closed semantic-state descriptor + runtime IDL registry |
| Profile B (CID-native artifacts) | **available** | `artifacts` / `kubo_cid` + wire envelopes/receipts |
| Profile F (event-DAG ordering) | **available** | `EventDAGStore` + wire DAG events |
| Profile G (risk scheduling) | **available** | Existing authority (datasets codecs, accelerate/kit transport, hermetic vectors); **not** a new SCG profile |
| `VerificationCommitment` (IVP Merkle) | **available**, **non-ZK** | Structural commitment only; **cannot** substitute for either sealer |
| `FullCheckpointSeal` / `create_full_checkpoint` / `publish_full_checkpoint` | **typed_unavailable** | No public Python symbols on this tree |
| `DeltaSeal` / `build_delta_seal` / `publish_delta_seal` / `IncrementalProofSealer` | **typed_unavailable** | Independent probe; also absent on this tree |

Conflict policy (scheduler capability policy + plan §2/§14):

- `new_mcplusplus_profile_allowed = false`
- missing full or incremental sealer disposition = `typed_unavailable`
- IVP Merkle commitment **must not** be treated as a ZK or execution seal

---

## 2. MCP++ authorities

| Authority | Path / package | Commit or role |
| --- | --- | --- |
| Shared wire/conformance | `ipfs_accelerate_py/mcplusplus` (gitlink → `endomorphosis/Mcp-Plus-Plus`) | `dc3164653a48d059ae9812078359daeafb451c07` |
| Canonical runtime primitives | `ipfs_accelerate_py.mcp_server.mcplusplus` | Feature-authoritative runtime used by unified MCP server and semantic-state wire |
| Alternate Trio surface | `ipfs_accelerate_py.mcplusplus_module` | Existing alternate host surface; not SCG ownership for new profiles |
| Root docs checkout | `mcpplusplus/` | History/checklist docs; planning authority remains the nested gitlink |

---

## 3. Profile A — interface description (MCP-IDL)

**Capability:** `mcp++/profile-a/interface-description`  
**Spec:** `ipfs_accelerate_py/mcplusplus/docs/spec/mcp-idl.md`  
@ `3022e05d1ba90a0ba47dd7a0b1534655642a1f80` (mcplusplus gitlink)

### Semantic-state closed descriptor

- Module: `ipfs_accelerate_py.agent_supervisor.semantic_state.wire`
- Source: `wire.py` @ `bd5fad7ac854e3127e8aef4a5af58ee7f0748e29`
- Schema file: `semantic_state/schemas/semantic-state-harness.interface.json`
  @ `39bf9f9776289f18785f7cfc2a579410404a39ba`
- Wire boundary constant: `mcp-plus-plus-profiles-a-b-f`
- Interface: `semantic-state-harness` / namespace
  `ipfs-accelerate.agent-supervisor` / version `1.0.0`
- Requires: Profile A + B + F capability strings
- Primary test: `test/api/semantic_state/test_wire.py`
  @ `bd5fad7ac854e3127e8aef4a5af58ee7f0748e29`

### Runtime IDL registry

- Package: `ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry`
- Source: `idl_registry.py` @ `5e677d7ef7972d012cb228eae0e47c0f9073babf`
- Symbols: `InterfaceDescriptorRegistry`, `build_descriptor`,
  `canonicalize_descriptor`, `compute_interface_cid`,
  `identify_interface_descriptor`
- Tests: `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py`

---

## 4. Profile B — CID-native artifacts

**Capability:** `mcp++/profile-b/cid-native-artifacts`  
**Spec:** `ipfs_accelerate_py/mcplusplus/docs/spec/cid-native-artifacts.md`  
@ `f81ec39789da10444f77f8d475dd2dda7ca04f2e`

### Runtime builders

Source `mcp_server/mcplusplus/artifacts.py` @
`dcd0617fe1f9a83172e3df18204096dd157bd85b`:

- `canonicalize_artifact`
- `compute_artifact_cid`
- `build_intent` / `build_decision` / `build_receipt` / `build_event`
- `envelope_from_payloads`
- `ArtifactStore`

CID helper: `kubo_cid.cid_for_bytes` @ same revision family
`dcd0617fe1f9a83172e3df18204096dd157bd85b`.

### Wire envelopes and receipts

`SemanticStateWireCodec` encodes/decodes execution envelopes and receipts with
identity = `canonicalize_artifact` + `cid_for_bytes`. The module does **not**
reimplement datasets identity and does **not** invent a local generic envelope.

### Conformance vectors (mcplusplus gitlink)

| Vector | Model |
| --- | --- |
| `conformance/vectors/execution_receipt.json` | `ExecutionReceipt` |
| `conformance/vectors/initialize_result.json` | `InitializeResult` |

Vector-touching revision observed in the submodule:
`2a2c765425fc2f6f593cf2b346fd029ce7f9fd10` (nested under gitlink
`dc3164653a48d059ae9812078359daeafb451c07`).

---

## 5. Profile F — event-DAG ordering

**Capability:** `mcp++/profile-f/event-dag-ordering`  
**Spec:** `ipfs_accelerate_py/mcplusplus/docs/spec/event-dag-ordering.md`  
@ `adbfff9818e3d884b9e6e0fc3b47429bd4ec63cb`

### Runtime store

- `EventDAGStore` / `EventNode` in `mcp_server/mcplusplus/event_dag.py`
  @ `dcd0617fe1f9a83172e3df18204096dd157bd85b`
- Test: `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py`

### Wire DAG events

`SemanticStateWireCodec.encode_dag_event` / `decode_dag_event` bind
`event_cid` to parents + payload CID + timestamp.

### Conformance vectors

| Vector | Model |
| --- | --- |
| `conformance/vectors/dag_event_epoch.json` | `DAGEvent` |
| `conformance/vectors/dag_event_iso.json` | `DAGEvent` |

---

## 6. Profile G — existing risk-scheduling authority

**Capability:** `mcp++/risk-scheduling` / `mcp++/risk-scheduling@1.0`  
**Spec:** `ipfs_accelerate_py/mcplusplus/docs/spec/risk-scheduling.md`  
@ `3ab193340980c1d9df001783f1dc1431cd884218`  
**Conformance suite schema:** `mcp++/profile-g/conformance-suite@1`

Profile G is **existing** authority consumed by kit and accelerate. SCG must not
create a new Profile G variant or treat G as an SCG execution profile.

### Artifact kinds (datasets-owned codecs)

Owner: `ipfs_datasets_py.logic.profile_g`  
Source: `ipfs_datasets_py/ipfs_datasets_py/logic/profile_g.py`  
@ `ebe73d4c8179ce7cf0d9850cfe540c4bf3f887b1`  
Nested gitlink: `1330038f626ef92993f03d46f21e1a57719e9c25`

Kinds: `Goal`, `Subgoal`, `PlanBranch`, `PlanSelection`, `TaskSpec`,
`RiskModel`, `RiskEvidence`, `RiskAssessment`, `NeighborhoodRecord`,
`NeighborhoodAttestation`, `ScheduleProposal`, `TaskClaim`,
`ClaimResolution`, `TaskReceipt`.

Public helpers include `canonical_profile_g_bytes`, `profile_g_cid`,
`validate_profile_g_artifact`, `GoalPlanValidator`, `RiskEvidenceStore`,
`NeighborhoodAttestationEngine`.

### Transport dispatchers

| Owner | Path | Revision |
| --- | --- | --- |
| accelerate | `mcp_server/mcplusplus/profile_g_transport.py` | `fd866abbbe114a9bef2f83d28465ff1053abf728` |
| kit | `ipfs_kit_py/.../mcplusplus/profile_g_transport.py` | `df2f9cc092456329de9724c45a50c54b410875d1` |

Accelerate facade: `PROFILE_G_PROFILE = "mcp++/risk-scheduling"`, 24
`PROFILE_G_METHODS`, transports `jsonrpc-http` and `mcp+p2p`,
`ProfileGDispatcher`.

Daemon-lane adapters that consume Profile G CIDs also live in
`agent_supervisor/merge/lease_coordination.py` @
`8f613252c2ff1460e6f2b551a2a8600a2d3ee519` (not a profile definition).

### Conformance / hermetic vectors

| Path | Role |
| --- | --- |
| `mcplusplus/conformance/vectors/profile_g_artifacts_valid.json` | Canonical valid codec suite (14 cases) |
| `mcplusplus/conformance/vectors/profile_g_artifacts_invalid.json` | Invalid codec cases |
| `mcplusplus/conformance/vectors/profile_g_protocol_valid.json` | Protocol suite (10 cases) |
| `mcplusplus/conformance/vectors/profile_g_protocol_invalid.json` | Invalid protocol cases |
| `mcplusplus/conformance/vectors/profile_g_three_peer.json` | Three-peer scenario |
| `ipfs_kit_py/tests/fixtures/mcp_plus_plus/profile_g_artifacts_valid.json` | Hermetic kit-vendored copy (pin `df2f9cc0...`) |

---

## 7. Shared conformance vector catalog

Authority directory:
`ipfs_accelerate_py/mcplusplus/conformance/vectors` at gitlink
`dc3164653a48d059ae9812078359daeafb451c07`.

| File | Model / affinity |
| --- | --- |
| `audit_entry.json` | `AuditEntry` / shared |
| `bus_message.json` | `BusMessage` / shared |
| `dag_event_epoch.json` | `DAGEvent` / Profile F |
| `dag_event_iso.json` | `DAGEvent` / Profile F |
| `delegation.json` | `Delegation` / UCAN (C) |
| `execution_receipt.json` | `ExecutionReceipt` / Profile B |
| `initialize_result.json` | `InitializeResult` / shared |
| `p2p_message.json` | `P2PMessage` / transport |
| `policy_decision.json` | `PolicyDecision` / policy (D) |
| `profile_g_*.json` (5 files) | Profile G suite |
| `profile_h_*.json` (3 files) | Profile H / x402 — **existing upstream, not SCG scope** |
| `session_error.json` | `SessionError` / shared |
| `wasm_proof_result.json` | Wire fixture only — **not** a released sealer |
| `zkp_proof_artifact.json` | Wire fixture only — **not** a released sealer |

JSON Schema files under `mcplusplus/schemas/` currently cover Profile H only;
Profiles A/B/F/G are enforced by runtime codecs, sealed harness descriptors,
and conformance vectors rather than a parallel JSON-Schema tree on this pin.

Planning baseline (plan §2): **34 passed** MCP++ CID, conformance-vector, and
event-DAG tests on the pinned clean controller.

---

## 8. Semantic-state wire codec (A/B/F consumer)

| Field | Value |
| --- | --- |
| Interface | `SemanticStateWireCodec@1` |
| Source | `semantic_state/wire.py` @ `bd5fad7...` |
| Boundary | `mcp-plus-plus-profiles-a-b-f` |
| Identity | `canonicalize_artifact` + `cid_for_bytes` only |
| Tests | `test/api/semantic_state/test_wire.py` |

Methods: interface descriptor; encode/decode execution envelope and receipt;
encode/decode DAG event; encode/decode root manifest.

Cross-check: `config/semantic_state_dependencies.seal.json` records
`profile_a` / `profile_b` / `profile_f` and `profile_g_cid` selectors as
harness dependency expectations (dependency seal inventory, **not** a
cryptographic proof seal).

---

## 9. IVP Merkle commitment (explicitly non-ZK)

| Field | Value |
| --- | --- |
| Type | `VerificationCommitment` |
| Interface | `VerificationCommitment@1` |
| Schema | `ipfs_accelerate_py/agent-supervisor/verification-commitment@1` |
| `IS_ZERO_KNOWLEDGE_PROOF` | **`False`** |
| Hash | `sha2-256` |
| Domains | `IVP-LEAF@1` / `IVP-NODE@1` / `IVP-EMPTY@1` |
| Leaf codec | `canonical-dag-json@1` |
| Contracts source | `verification/contracts.py` @ `0cdf81bdd283dad6c27c9d23bbb6637d7dd54cff` |
| Builder | `build_verification_commitment` in `verification/bundle.py` @ `52e8fe17fdc7ac63f905872b194d930ae36ab1db` |
| Public API freeze | `8c7800cedc5e1b848367db9952f912428466f8cc` |

The commitment is a **structural** Merkle root over admitted verification
receipt leaves in a `VerificationBundle`. It is:

- **not** a zero-knowledge proof
- **not** cryptographic proof of underlying execution by itself
- **not** a full-checkpoint seal
- **not** a delta / incremental proof seal
- **cannot substitute** for `FullCheckpointSeal` or `DeltaSeal` /
  `IncrementalProofSealer`

Tests: `test/api/test_agent_supervisor_verification_bundle.py`,
`test/api/test_agent_supervisor_verification_contracts.py`.

---

## 10. Proof-sealer capability probes

### Upstream program state

The IncrementalProofSealer (`IPS-*`) is a **separate live program**. Planning
observed pin `7dc8f1422cb7e80757077948dc0785c1aaa4fd25` with early contracts
only. That commit is **not** an ancestor of this SCG controller tree
(`8ba3ada7...`). A development branch `agent/incremental-proof-sealer-v1`
exists with plan/board/inventory artifacts; those APIs are **not released**
onto the inspected SCG HEAD.

Policy: missing sealer → `typed_unavailable`. SCG must not import unfinished
private sealer code. Artifacts may still be content-addressed without a seal.

### Full-checkpoint seal (independent probe)

Probed exact names: `FullCheckpointSeal`, `create_full_checkpoint`,
`publish_full_checkpoint`.

| Result | Value |
| --- | --- |
| Status | **`typed_unavailable`** |
| Source path | `null` |
| Source revision | `null` |
| Import path | `null` |
| Probe | Repository-wide exact symbol search over `*.py` on the inspected tree → **zero hits** |

Cannot be inferred from IVP commitment presence, MCP++ ZKP wire vectors,
`agent_supervisor.proof`, or documents on another branch.

### Delta / incremental seal (independent probe)

Probed exact names: `DeltaSeal`, `DeltaSeal@1`, `build_delta_seal`,
`publish_delta_seal`, `IncrementalProofSealer`.

| Result | Value |
| --- | --- |
| Status | **`typed_unavailable`** |
| Source path | `null` |
| Source revision | `null` |
| Import path | `null` |
| Probe | Repository-wide exact symbol search over `*.py` on the inspected tree → **zero hits** |

Independently typed unavailable from the full-checkpoint probe. Absence of one
does not imply presence of the other; presence of IVP Merkle commitment does
not release either.

### Related non-sealer surfaces (do not overclaim)

| Surface | Why it is not the IPS sealer |
| --- | --- |
| `ipfs_accelerate_py.agent_supervisor.proof` | Code-contract / formal / doctor helpers |
| `control/manual_completion_seal.py` | Supervisor manual completion helper |
| `config/semantic_state_dependencies.seal.json` | Dependency inventory seal |
| MCP++ `zkp_proof_artifact` / `wasm_proof_result` vectors | Wire fixtures only |

---

## 11. Proof vs non-proof distinctions (summary)

1. **Wire / vectors (A/B/F/G)** — interoperability and scheduling authority;
   not cryptographic seals of SCG evaluation.
2. **IVP `VerificationCommitment`** — structural non-ZK Merkle commitment over
   admitted receipts; useful integrity evidence inside IVP; **not** a sealer.
3. **Full checkpoint seal** — planned IPS public surface; **typed unavailable**
   here.
4. **Delta / incremental seal** — planned IPS public surface; **typed
   unavailable** here, independently of (3).
5. Until a released sealer is commit-bound and qualified, SCG seal status for
   benchmark/ContextPack/bundle/differential/calibration/promotion artifacts
   remains `unavailable` even when those objects are content-addressed.

---

## 12. Focused interop tests

| Path | Covers |
| --- | --- |
| `test/api/semantic_state/test_wire.py` | A/B/F wire codec |
| `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py` | Profile A registry |
| `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py` | Profile B |
| `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py` | Profile F |
| `ipfs_accelerate_py/mcp/tests/test_profile_g_transport.py` | Profile G transport |
| `test/integration/test_mcp_mcplusplus_interop_smoke.py` | Runtime interop smoke |
| `test/api/test_agent_supervisor_verification_bundle.py` | Commitment builder |
| `test/api/test_agent_supervisor_verification_contracts.py` | Commitment contract / non-ZK |

---

## 13. Acceptance (SCG-004)

| Criterion | Result |
| --- | --- |
| Full seal interface located and commit-bound **or** independently typed unavailable | **typed_unavailable** (independent probe) |
| Incremental/delta seal interface located and commit-bound **or** independently typed unavailable | **typed_unavailable** (independent probe) |
| IVP Merkle commitment explicitly non-ZK | **`IS_ZERO_KNOWLEDGE_PROOF = False`** |
| IVP commitment cannot substitute for either sealer | **recorded; substitution forbidden** |
| Profile A/B/F + existing Profile G schemas/vectors commit-bound | **yes** |
| No new MCP++ profile | **yes** |
