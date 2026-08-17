# AAE-004 — MCP++ conformance boundary and proof-sealer release surfaces

**Evidence id:** `aae/mcplusplus-boundary@1`  
**Interfaces:** `MCPPlusPlusBoundary@1`, `IncrementalProofSealerCapabilityProbe@1`  
**Task:** AAE-004  
**Schema:** `aae/mcplusplus-boundary@1`  
**Machine-readable companion:** [`interoperability.json`](interoperability.json)

## Authority

| Field | Value |
| --- | --- |
| Controller repository | `endomorphosis/ipfs_accelerate_py` |
| Inspected tree | `dcfc11e83e904197bf0d93a502466e441aa6a3d2` |
| Tree OID | `a065e39b372125cfc98910e89fdb52e433e57da7` |
| Planning authority pin | `7c9f3fa3d2ac14c7b5bfa5036e2fe6fb59f0afda` (ancestor of inspected tree) |
| Inspected | 2026-08-13 |
| Status | Existing MCP++ Profiles A/B/F/G wire surfaces and conformance vectors are present. Full-checkpoint and delta/incremental sealer public APIs are independently `typed_unavailable`. No new MCP++ profile is proposed. |

Nested gitlinks observed at inventory time:

| Nested repository | Commit |
| --- | --- |
| `ipfs_datasets_py` | `fbd1ba9f70803de157622bb20e22595ef09d606f` |
| `ipfs_kit_py` | `c7e5feeb24582ab68c1f5ca626366b665a82ad61` |
| `ipfs_accelerate_py/mcplusplus` | `dc3164653a48d059ae9812078359daeafb451c07` |

Planning-time external IPS observations (not AAE nested pins):

| Observation | Commit |
| --- | --- |
| Live IncrementalProofSealer branch at planning | `65ef6ed4cb9fa51d7c2c6c1a4f9036678f8cb893` |
| IPS branch tip at inspection | `c983d4db33e83aba0cfb92a74fedb9ff26e4bf73` |
| Proof-sealer datasets pin (partial evidence only) | `321fe10c191b9dab84206b5f9bd598aa2e46bcc8` |
| Proof-sealer kit pin (partial store only) | `866ae2cd0a1a94a794ff9316a9a4f67a10245957` |

This inventory records **existing** shared conformance and **probes** checkpoint/delta sealer APIs separately. It does **not** invent missing sealer APIs, second CID profiles, or a new MCP++ profile.

## Ownership and conflict policy

MCP++ shared wire ownership:

- shared schemas/vectors/harnesses under gitlink `ipfs_accelerate_py/mcplusplus` → `endomorphosis/Mcp-Plus-Plus`
- feature-authoritative runtime under `ipfs_accelerate_py.mcp_server.mcplusplus`
- closed semantic-state wire boundary under `ipfs_accelerate_py.agent_supervisor.semantic_state.wire`
- Profile G canonical codecs under `ipfs_datasets_py.logic.profile_g`
- Profile G transport under accelerate and kit `mcp_server.mcplusplus.profile_g_transport`

AAE / accelerate do **not** own a new MCP++ profile. Application payloads remain inside existing profiles. Shared schemas or flat vectors may be added later only after a genuine cross-language requirement is demonstrated; optional paths are a permission envelope, not a requirement to invent interoperability work.

**Conflict policy** (scheduler `capability_policy` + AAE plan §3/§4):

| Rule | Value |
| --- | --- |
| `new_mcplusplus_profile_allowed` | `false` |
| `local_generic_envelope_allowed` | `false` |
| `new_proof_or_zk_system_allowed` | `false` |
| missing full-checkpoint sealer | `typed_unavailable` |
| missing delta / IncrementalProofSealer | `typed_unavailable` |
| IVP Merkle commitment may substitute for sealer | `false` |
| AAE-local proof system masquerading as seal | forbidden |

---

## 1. Claim boundary

| Surface | Status on inspected tree | Notes |
| --- | --- | --- |
| Profile A (MCP-IDL) | **available** | Closed semantic-state descriptor + runtime IDL registry |
| Profile B (CID-native artifacts) | **available** | `artifacts` / `kubo_cid` + wire envelopes/receipts |
| Profile F (event-DAG ordering) | **available** | `EventDAGStore` + wire DAG events |
| Profile G (risk scheduling) | **available** | Existing authority (datasets codecs, accelerate/kit transport, hermetic vectors); **not** a new AAE profile |
| Profiles C/D/E/H | existing upstream, not AAE authority | Vectors/specs present; do not create variants |
| `VerificationCommitment` (IVP Merkle) | **available**, **non-ZK** | Structural commitment only; **cannot** substitute for either sealer |
| `FullCheckpointSeal` / `create_full_checkpoint` / `publish_full_checkpoint` | **typed_unavailable** | Independent probe; no public Python symbols on this tree |
| `DeltaSeal` / `build_delta_seal` / `publish_delta_seal` / `IncrementalProofSealer` | **typed_unavailable** | Independent probe; also absent on this tree |

---

## 2. MCP++ authorities

| Authority | Path / package | Commit or role |
| --- | --- | --- |
| Shared wire/conformance | `ipfs_accelerate_py/mcplusplus` (gitlink → `endomorphosis/Mcp-Plus-Plus`) | `dc3164653a48d059ae9812078359daeafb451c07` |
| Canonical runtime primitives | `ipfs_accelerate_py.mcp_server.mcplusplus` | Feature-authoritative runtime used by unified MCP server and semantic-state wire |
| Alternate Trio surface | `ipfs_accelerate_py.mcplusplus_module` | Existing alternate host surface; not AAE ownership for new profiles |
| Root docs checkout | `mcpplusplus/` | History/checklist docs; planning authority remains the nested gitlink |

Cross-language harness roots (same vectors):

| Language | Root |
| --- | --- |
| Python | `ipfs_accelerate_py/mcplusplus/tests-py` |
| Go | `ipfs_accelerate_py/mcplusplus/tests-go` |
| Rust | `ipfs_accelerate_py/mcplusplus/tests-rs` |
| TypeScript | `ipfs_accelerate_py/mcplusplus/tests-ts` |

Planning baseline: **58 passed** MCP++ CID envelopes, event DAG, conformance vectors, and Profile G codec tests in 0.15 seconds on the clean controller.

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

## 6. Profile G — risk scheduling (existing authority)

**Capability:** `mcp++/risk-scheduling` @ `1.0`  
**Spec:** `ipfs_accelerate_py/mcplusplus/docs/spec/risk-scheduling.md`  
@ `3ab193340980c1d9df001783f1dc1431cd884218`

Profile G is **not** a new AAE profile. It is existing scheduling/artifact
authority consumed across datasets, accelerate, and kit.

### Artifact kinds (14)

`Goal`, `Subgoal`, `PlanBranch`, `PlanSelection`, `TaskSpec`, `RiskModel`,
`RiskEvidence`, `RiskAssessment`, `NeighborhoodRecord`,
`NeighborhoodAttestation`, `ScheduleProposal`, `TaskClaim`, `ClaimResolution`,
`TaskReceipt`

### Datasets codecs

- Import: `ipfs_datasets_py.logic.profile_g`
- Source: `profile_g.py` @ `ebe73d4c8179ce7cf0d9850cfe540c4bf3f887b1`
  (nested pin `fbd1ba9f...`)
- Constants: `PROFILE_G_VERSION = "1.0"`,
  `PROFILE_G_CAPABILITY = "mcp++/risk-scheduling"`
- Symbols include `canonical_profile_g_bytes`, `profile_g_cid`,
  `validate_profile_g_artifact`, `GoalPlanValidator`, `RiskEvidenceStore`,
  `NeighborhoodAttestationEngine`

### Accelerate transport

- Source: `mcp_server/mcplusplus/profile_g_transport.py`
  @ `fd866abbbe114a9bef2f83d28465ff1053abf728`
- `PROFILE_G_PROFILE = "mcp++/risk-scheduling"`
- **23** methods (goals/tasks/risk/neighborhood/schedule)
- Transports: `jsonrpc-http`, `mcp+p2p`

### Kit transport + hermetic fixture

- Transport: `ipfs_kit_py/.../profile_g_transport.py`
  @ `a344994f448e3106e31dcd14b39e3f0721d09cfc`
  (nested pin `c7e5feeb...`)
- Hermetic fixture:
  `ipfs_kit_py/tests/fixtures/mcp_plus_plus/profile_g_artifacts_valid.json`
  @ `df2f9cc092456329de9724c45a50c54b410875d1`

### Conformance suites

| Vector | Schema | Cases |
| --- | --- | --- |
| `profile_g_artifacts_valid.json` | `mcp++/profile-g/conformance-suite@1` | 14 |
| `profile_g_artifacts_invalid.json` | same | 10 |
| `profile_g_protocol_valid.json` | same | 10 |
| `profile_g_protocol_invalid.json` | same | 19 |
| `profile_g_three_peer.json` | `mcp++/profile-g/three-peer-fixture@1` | fixture |

---

## 7. Shared conformance vectors (full list)

Authority directory:
`ipfs_accelerate_py/mcplusplus/conformance/vectors/` @ `dc316465...`

Default shape: `{"model": "<CanonicalModelName>", "payload": { ... }}`.

| Vector | Model / affinity |
| --- | --- |
| `audit_entry.json` | `AuditEntry` / shared |
| `bus_message.json` | `BusMessage` / shared |
| `dag_event_epoch.json` | `DAGEvent` / Profile F |
| `dag_event_iso.json` | `DAGEvent` / Profile F |
| `delegation.json` | `Delegation` / Profile C (not AAE authority) |
| `execution_receipt.json` | `ExecutionReceipt` / Profile B |
| `initialize_result.json` | `InitializeResult` / shared |
| `p2p_message.json` | `P2PMessage` / transport |
| `policy_decision.json` | `PolicyDecision` / Profile D (not AAE authority) |
| `profile_g_*` | Profile G suites (above) |
| `profile_h_*` | Profile H payments (not AAE authority) |
| `session_error.json` | `SessionError` / shared |
| `wasm_proof_result.json` | `WasmProofResult` / wire fixture only — **not** a sealer |
| `zkp_proof_artifact.json` | `ZKProofArtifact` / wire fixture only — **not** a sealer |

---

## 8. Semantic-state wire codec

**Interface:** `SemanticStateWireCodec@1`  
**Status:** available  
**Source:** `ipfs_accelerate_py/agent_supervisor/semantic_state/wire.py`  
@ `bd5fad7ac854e3127e8aef4a5af58ee7f0748e29`

```text
WIRE_BOUNDARY = "mcp-plus-plus-profiles-a-b-f"
INTERFACE_NAME = "semantic-state-harness"
INTERFACE_NAMESPACE = "ipfs-accelerate.agent-supervisor"
INTERFACE_VERSION = "1.0.0"
```

Public operations include `interface_descriptor`,
`encode`/`decode_execution_envelope`, `encode`/`decode_execution_receipt`,
`encode`/`decode_dag_event`, `encode`/`decode_root_manifest`, and
`cid_for_payload`.

Rules:

1. Identity is exclusively `canonicalize_artifact` + `cid_for_bytes`.
2. No local generic envelope and no second CID implementation.
3. No new MCP++ profile is permitted.

---

## 9. IVP Merkle commitment (non-ZK, non-sealer)

**Name:** `VerificationCommitment`  
**Interface:** `VerificationCommitment@1`  
**Status:** available  
**Source:** `verification/contracts.py` @ `0cdf81bdd283dad6c27c9d23bbb6637d7dd54cff`  
**Builder:** `verification/bundle.py` @ `52e8fe17fdc7ac63f905872b194d930ae36ab1db`

| Property | Value |
| --- | --- |
| `IS_ZERO_KNOWLEDGE_PROOF` | `False` |
| Hash | `sha2-256` |
| Leaf codec | `canonical-dag-json@1` |
| Domains | `IVP-LEAF@1` / `IVP-NODE@1` / `IVP-EMPTY@1` |
| Is proof sealer | **no** |
| May substitute for `FullCheckpointSeal` | **no** |
| May substitute for `DeltaSeal` | **no** |
| May substitute for `IncrementalProofSealer` | **no** |

Structural Merkle commitment over admitted verification receipts only. Structural
validation is not cryptographic validation of underlying execution.

---

## 10. Proof-sealer capability probe (`IncrementalProofSealerCapabilityProbe@1`)

Checkpoint and delta sealer APIs are probed **independently**. Absence of one
does not imply presence of the other. Missing surfaces are `typed_unavailable`.

### Program status

| Field | Value |
| --- | --- |
| Program | IncrementalProofSealer (prefix `IPS-`) |
| Planning pin | `65ef6ed4...` (external; not ancestor of AAE HEAD) |
| Branch tip at inspection | `c983d4db...` on `agent/incremental-proof-sealer-v1` |
| Released on inspected AAE tree | **false** |
| Policy disposition when missing | `typed_unavailable` |

The AAE plan records that the live proof-sealer branch currently exposes partial
datasets evidence/`ProofUnit` and kit proof-seal store surfaces on separate pins
(`321fe10c...` / `866ae2cd...`) but does **not** expose the required public
Python sealer APIs. Those partial pins are not the AAE nested gitlinks and are
not present as objects in this worktree.

Early IPS-branch modules
`agent_supervisor/proof/incremental_sealing/{admission,backends}.py` exist on
the external branch as cache-admission / backend binding work and are
**not** importable on this AAE tree. They are **not**
`FullCheckpointSeal` / `DeltaSeal` release surfaces.

### 10.1 Full-checkpoint sealer (independent probe)

| Probed name | Status |
| --- | --- |
| `FullCheckpointSeal` | `typed_unavailable` |
| `create_full_checkpoint` | `typed_unavailable` |
| `publish_full_checkpoint` | `typed_unavailable` |

**Probe method:** repository-wide exact symbol search over `*.py` on
`dcfc11e8...` plus importlib probes. Zero production hits; names appear only in
protected plan/todo/validator text and external IPS docs.

**Cannot be inferred from:** `VerificationCommitment`, MCP++ zkp/wasm vectors,
`agent_supervisor.proof`, `MergeCheckpoint` (git submodule merge helper), IPS
planning documents, or IPS-branch admission modules.

### 10.2 Delta / incremental sealer (independent probe)

| Probed name | Status |
| --- | --- |
| `DeltaSeal` / `DeltaSeal@1` | `typed_unavailable` |
| `build_delta_seal` | `typed_unavailable` |
| `publish_delta_seal` | `typed_unavailable` |
| `IncrementalProofSealer` | `typed_unavailable` |

**Probe method:** same tree-wide search; importlib of
`ipfs_accelerate_py.agent_supervisor.incremental_proof_sealer` and
`...proof.incremental_sealing` → `ModuleNotFoundError` on this tree.

**Cannot be inferred from:** `VerificationCommitment`, full-checkpoint presence
or absence, MCP++ zkp vectors, `manual_completion_seal`, `MergeCheckpoint`, or
IPS plan text that documents *planned* APIs.

### 10.3 Related surfaces that are not sealers

| Surface | Role | Status |
| --- | --- | --- |
| `agent_supervisor.proof` | Code-contract / doctor / FV helpers | available, not sealer |
| `control/manual_completion_seal.py` | Supervisor manual completion helper | available, not sealer |
| `merge/merge_checkpoint.py` (`MergeCheckpoint`) | Git submodule merge resume | available, not sealer |
| MCP++ `wasm_proof_result` / `zkp_proof_artifact` | Wire fixtures only | fixture only |
| `VerificationCommitment` | Structural non-ZK IVP commitment | available, not sealer |

---

## 11. Missing APIs (typed_unavailable — do not invent)

| Name | Status |
| --- | --- |
| `FullCheckpointSeal` | typed_unavailable |
| `create_full_checkpoint` | typed_unavailable |
| `publish_full_checkpoint` | typed_unavailable |
| `DeltaSeal` / `DeltaSeal@1` | typed_unavailable |
| `build_delta_seal` | typed_unavailable |
| `publish_delta_seal` | typed_unavailable |
| `IncrementalProofSealer` public package | typed_unavailable |
| New AAE or MCP++ profile | **must_not_invent** |

---

## 12. Focused tests and validation

Recommended evidence paths:

- `test/api/semantic_state/test_wire.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_{artifacts,event_dag,idl}.py`
- `ipfs_accelerate_py/mcp/tests/test_profile_g_transport.py`
- `test/integration/test_mcp_mcplusplus_interop_smoke.py`
- `test/api/test_agent_supervisor_verification_{bundle,contracts}.py`
- `ipfs_accelerate_py/mcplusplus/tests-py/integration/test_conformance_vectors.py`
- `ipfs_accelerate_py/mcplusplus/tests-go/conformance_vectors_test.go`

Task validation command:

```bash
python3 -m json.tool docs/architecture/adversarial_assurance_inventory/interoperability.json >/dev/null
```

---

## 13. Handoff summary for Adversarial Assurance Engine

```text
MCP++ shared gitlink (dc316465)
  Profiles A/B/F specs + vectors + four-language harnesses
  Profile G suites (existing risk-scheduling authority)
        │
        ▼
accelerate mcp_server.mcplusplus runtime
  artifacts / event_dag / idl_registry / profile_g_transport
        │
        ├─ SemanticStateWireCodec  (mcp-plus-plus-profiles-a-b-f)
        ├─ datasets profile_g codecs (fbd1ba9f)
        ├─ kit profile_g transport + hermetic vectors (c7e5feeb)
        └─ VerificationCommitment (non-ZK structural only)
                 │
                 └─ FullCheckpointSeal / DeltaSeal / IncrementalProofSealer
                    ── typed_unavailable on this AAE tree ──
                    (external IPS program incomplete; do not invent)
```

AAE **consumes** existing MCP++ wire and Profile G authority. It must not
propose a new profile, treat IVP commitments or MCP++ proof vectors as seals, or
import unfinished private sealer code as if released.

## Acceptance (AAE-004)

| Criterion | Status |
| --- | --- |
| Existing profiles/vectors recorded | Yes — A/B/F/G commit-bound; shared vector inventory; four-language harnesses |
| Checkpoint sealer APIs probed separately | Yes — `FullCheckpointSeal` / `create_full_checkpoint` / `publish_full_checkpoint` |
| Delta sealer APIs probed separately | Yes — `DeltaSeal` / `build_delta_seal` / `publish_delta_seal` / `IncrementalProofSealer` |
| Missing surfaces typed unavailable | Yes — both probes independent; status vocabulary `typed_unavailable` |
| No profile proposed | Yes — `new_mcplusplus_profile_allowed=false`; no AAE profile declared |
| IVP commitment not treated as seal | Yes — `IS_ZERO_KNOWLEDGE_PROOF=False`; cannot substitute for either sealer |
