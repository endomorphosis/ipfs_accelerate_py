# MCPP-081 — Runtime adapter matrix

| Field | Value |
| --- | --- |
| Task | `MCPP-081` |
| Title | Close remaining runtime adapter and documentation gaps |
| Track | `runtime-closeout` |
| Goal | `MCPP-G170` |
| Interface | `RuntimeAdapterMatrix@1` |
| Depends on | `MCPP-023`, `MCPP-034`, `MCPP-049`, `MCPP-053`, `MCPP-057`, `MCPP-065`, `MCPP-069`, `MCPP-072` |
| Status | **complete** (matrix documentation) |
| Recorded at (UTC) | `2026-08-16T07:29:01Z` |
| Forest reference | `docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json` |
| Inventory reference | `docs/reports/mcplusplus-1.0-gap-closure/baseline/profiles-a-h-inventory.md` |

## 1. How to read this matrix

### 1.1 Disposition vocabulary

| Disposition | Meaning |
| --- | --- |
| `implemented` | Runtime owns an adapter surface for the profile/capability **and** a concrete verification command is recorded in this matrix. Schema presence alone never qualifies. |
| `partial` | Meaningful adapter code and (usually) a command exist, but known gaps remain relative to draft/normative claims or forest-wide admission. |
| `blocked` | No dedicated adapter surface for this runtime×profile, or the path is blocked by a recorded dependency/availability constraint. |

**Hard rule:** no cell is marked `implemented` without a **Command** column entry that can re-run the evidence.

SwissKnife runtime-local `CONFORMANCE_MATRIX.md` PASS claims are **not** forest-wide or four-language admission. They may support a `partial` (or, when a gap-closure receipt landed, `implemented`) disposition only when the corresponding command is listed here.

### 1.2 Runtimes and profiles covered

| Runtimes (columns) | Profiles / capabilities (rows) |
| --- | --- |
| accelerate (`ipfs_accelerate_py`) | **A** MCP-IDL |
| datasets (`ipfs_datasets_py`) | **B** CID-native artifacts / envelopes |
| kit (`ipfs_kit_py`) | **C** UCAN delegation |
| SwissKnife (bound operator checkout) | **D** Temporal deontic policy |
| | **E** `mcp+p2p` transport |
| | **F** Event DAG / compaction / ZK |
| | **G** Risk / neighborhood / fencing |
| | **H** x402 payments |
| | **Bindings** dual MCP legacy/current (cross-cutting) |
| | **Durable** DurableExecutor binding (cross-cutting) |
| | **A2A** execution extension (cross-cutting) |

---

## 2. Bound checkouts (fail-closed discovery)

Paths and remotes are taken from `repository-forest.json` (`operator_checkouts`) and re-verified for SwissKnife via `git remote -v` (discovered, **not invented**).

| Runtime | Worktree / program path | Operator checkout (forest) | Nested / bound HEAD | Origin remote |
| --- | --- | --- | --- | --- |
| accelerate | `ipfs_accelerate_py/` (program worktree) | `/home/barberb/lift_coding/external/ipfs_accelerate` | `ea11293bb996f052d620eae989f5377a956764b1` | `https://github.com/endomorphosis/ipfs_accelerate_py` |
| datasets | `ipfs_datasets_py/` (program worktree) | `/home/barberb/lift_coding/external/ipfs_datasets` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` | `https://github.com/endomorphosis/ipfs_datasets_py` |
| kit | `ipfs_kit_py/` (program worktree) | `/home/barberb/lift_coding/external/ipfs_kit` | `6196017ca3df016c7159dce43af60f2a0d96a9ae` | `https://github.com/endomorphosis/ipfs_kit_py` |
| SwissKnife | `/home/barberb/lift_coding/swissknife` | same | `afdbf885175fde34505ef05a2ea6aac5535ad03e` | **`https://github.com/endomorphosis/swissknife`** (discovered) |

### 2.1 SwissKnife remote discovery (authoritative)

Discovery command (cwd = SwissKnife checkout):

```bash
git -C /home/barberb/lift_coding/swissknife remote -v
```

Observed remotes (matches MCPP-001 forest + MCPP-009 baseline):

```text
origin	https://github.com/endomorphosis/swissknife (fetch)
origin	https://github.com/endomorphosis/swissknife (push)
upstream	https://github.com/dnakov/anon-kode.git (fetch)
upstream	https://github.com/dnakov/anon-kode.git (push)
```

| Field | Value |
| --- | --- |
| Canonical origin (discovered) | `https://github.com/endomorphosis/swissknife` |
| Invented URL | **no** |
| Matches forest `operator_checkouts.swissknife` | **yes** |
| Live HEAD | `afdbf885175fde34505ef05a2ea6aac5535ad03e` |
| Pre-existing dirty overlay (preserve) | `test-results/virtual-desktop-ipfs-mcp-orb/svd-132.json` |

No task in this program may replace the SwissKnife origin with a guessed URL.

---

## 3. Executive matrix — Profiles A–H

Disposition only. Commands, modules, and gap notes are in §5–§8.

| Profile | accelerate | datasets | kit | SwissKnife |
| --- | --- | --- | --- | --- |
| **A** MCP-IDL | `partial` | `blocked` | `blocked` | `partial` |
| **B** CID-native artifacts | `implemented` | `partial` | `partial` | `partial` |
| **C** UCAN delegation | `partial` | `blocked` | `implemented` | `partial` |
| **D** Temporal deontic policy | `partial` | `implemented` | `partial` | `partial` |
| **E** P2P transport | `partial` | `implemented` | `partial` | `partial` |
| **F** Event DAG / ZK | `partial` | `partial` | `partial` | `partial` |
| **G** Risk / fencing | `implemented` | `partial` | `partial` | `partial` |
| **H** x402 payments | `implemented` | `partial` | `partial` | `implemented` |

### 3.1 Cross-cutting capability matrix

| Capability | accelerate | datasets | kit | SwissKnife |
| --- | --- | --- | --- | --- |
| Dual MCP bindings (legacy + current) | `implemented` | `implemented` | `blocked` | `blocked` |
| DurableExecutor runtime bind | `implemented` | `blocked` | `blocked` | `blocked` |
| A2A execution extension | `implemented` | `blocked` | `blocked` | `implemented` |

---

## 4. Command index (every `implemented` cell)

If a cell in §3 or §3.1 is `implemented`, its command **must** appear below.

| Runtime | Profile / capability | Disposition | Command |
| --- | --- | --- | --- |
| accelerate | **B** Envelope / CID artifacts | `implemented` | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_envelope.py` |
| accelerate | **G** Stale fence rejection | `implemented` | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_profile_g_fence.py` |
| accelerate | **H** Payment≠authorization + seller | `implemented` | `cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_negatives.py` |
| accelerate | Bindings | `implemented` | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_bindings.py` |
| accelerate | DurableExecutor | `implemented` | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_durable_runtime.py` |
| accelerate | A2A reference adapter | `implemented` | `python -m pytest -q test/api/test_mcplusplus_a2a_handoff.py` |
| datasets | **D** Policy evaluator fail-closed | `implemented` | `cd ipfs_datasets_py && python -m pytest -q tests/unit/mcp_server/test_mcplusplus_policy_evaluator.py` |
| datasets | **E** P2P framing / abuse reuse | `implemented` | `cd ipfs_datasets_py && python -m pytest -q tests/unit/mcp_server/test_mcplusplus_p2p_framing.py` |
| datasets | Bindings | `implemented` | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_bindings.py` |
| kit | **C** UCANVerifier + negatives | `implemented` | `cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/mcplusplus/test_ucan_verifier.py` |
| SwissKnife | **H** Profile H adapter | `implemented` | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/profile-h-adapter.test.ts` |
| SwissKnife | A2A extension | `implemented` | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/a2a-adapter.test.ts` |

Supplementary (support `partial` cells; **not** sufficient alone for forest-wide production admission):

| Runtime | Profile | Command |
| --- | --- | --- |
| accelerate | A | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py` |
| accelerate | C | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py` |
| accelerate | D | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py` |
| accelerate | E | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_peer_primitives.py` |
| accelerate | F | `python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py` |
| accelerate | H seller (supplementary) | `python -m pytest -q tests/mcplusplus_profile_h/test_paid_accelerator.py` |
| datasets | F ceremony/P2P | `cd ipfs_datasets_py && python -m pytest -q tests/mcp/integration/test_profile_f_ceremony_p2p.py` |
| datasets | D + P2P gate | `cd ipfs_datasets_py && python -m pytest -q tests/mcp/integration/test_profile_d_policy_p2p.py` |
| datasets | G | `cd ipfs_datasets_py && python -m pytest -q tests/unit_tests/logic/test_profile_g.py` |
| datasets | H suite dir | `cd ipfs_datasets_py && python -m pytest -q tests/mcplusplus_profile_h` |
| kit | G transport | `cd ipfs_kit_py && python -m pytest -q tests/test_profile_g_transport.py` |
| kit | D policy surface | `cd ipfs_kit_py && python -m pytest -q tests/test_profile_d_policy.py` |
| kit | E transport security | `cd ipfs_kit_py && python -m pytest -q tests/runtime_readiness/mcplusplus/test_transport_security_parity.py` |
| kit | H paid kit | `cd ipfs_kit_py && python -m pytest -q tests/mcplusplus_profile_h/test_paid_kit.py` |
| SwissKnife | A | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/mcp-idl.test.ts` |
| SwissKnife | B | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/mcp-envelope.test.ts` |
| SwissKnife | C | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/profile-c-ucan.test.ts` |
| SwissKnife | D | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/profile-d-policy.test.ts` |
| SwissKnife | E | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/mcp-p2p-session.test.ts` |
| SwissKnife | F | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/event-dag.test.ts` |
| SwissKnife | G | `cd /home/barberb/lift_coding/swissknife && npm run test:run -- test/mcp-plus-plus/profile-g-connector.test.ts` |

SwissKnife vitest equivalent (any of the above):

```bash
npx vitest run --config build-tools/configs/vitest.config.ts <test-path>
```

---

## 5. accelerate — adapter surfaces

Program-tree modules under `ipfs_accelerate_py/mcp_server/mcplusplus/` unless noted.

| Profile | Disposition | Primary adapter surface | Evidence / gap |
| --- | --- | --- | --- |
| A | `partial` | `idl_registry.py`; `mcplusplus_module/interface_descriptor.py` | Registry + tests present; interface CID not Kubo-byte-identical (inventory §2). Command in §4 supplementary. |
| B | `implemented` | `envelope.py`, `artifacts.py`, `kubo_cid.py` | MCPP-034 create/persist/verify via mcpp-jcs-v1. Command: envelope test. |
| C | `partial` | `delegation.py` | Real Ed25519 when `require_signatures=True`; default path still allows non-crypto tokens. |
| D | `partial` | `policy_engine.py` | Local clause engine; may bridge datasets. Not full adversarial six-obligation forest admission. |
| E | `partial` | `p2p_framing.py`, `peer_*.py` | Framing + peer registry/bootstrap; not the heaviest P2P owner (datasets). |
| F | `partial` | `event_dag.py`; `mcplusplus_module/dag_compaction.py` | Structure real; ZK often `simulated_groth16` unless opt-in backend available. |
| G | `implemented` | `profile_g.py`, `profile_g_transport.py`, `risk_scheduler.py` | MCPP-069 stale fence denial. Command: fence test. Three-peer harness is separate codec/harness evidence. |
| H | `implemented` | `profile_h.py` (`PaidAcceleratorService`) | MCPP-072 seller fence + negatives (paid≠auth, replay, transport split). Command: negatives suite. |
| Bindings | `implemented` | `bindings.py` | MCPP-023 dual/legacy/current/fail-closed. |
| Durable | `implemented` | `durable/runtime.py` | MCPP-053 start/resume/cancel via DurableExecutor; no second journal. |
| A2A | `implemented` | `a2a_adapter.py` (`A2ATaskAdapter@1`) | Reference handoff; MCPP-056 gate. |

Prior runtime receipts: `runtime/profile-h-adapters.md` (accelerate §4).

---

## 6. datasets — adapter surfaces

| Profile | Disposition | Primary adapter surface | Evidence / gap |
| --- | --- | --- | --- |
| A | `blocked` | — | No dedicated IDL InterfaceRepository in datasets; IDL owned by accelerate/SwissKnife. |
| B | `partial` | workflow / executor envelopes (indirect) | Consumes CID envelopes via workflow/P2P paths; no dedicated Profile B codec package. |
| C | `blocked` | — | No dedicated UCAN verifier module; depends on kit/accelerate crypto paths. |
| D | `implemented` | `mcp_server/mcplusplus/policy.py`; `logic/profile_d_policy.py` | MCPP-049: dispatch consults evaluator; missing evaluator is deny. Command: policy evaluator test. |
| E | `implemented` | `p2p_libp2p_transport.py`, `p2p_framing.py` | MCPP-065 versioned protocol IDs + shared framing; abuse vectors reused. Command: p2p framing test. |
| F | `partial` | ceremony/P2P integration | Groth16 enable flag coupling; not always-on proof-verified. Command: ceremony P2P test (supplementary). |
| G | `partial` | `logic/profile_g.py`; workflow queues | Coordination/scheduler overlap; not primary fence owner (accelerate/kit). |
| H | `partial` | `mcp_server/mcplusplus/profile_h.py` | Module present; not the MCPP-072 seller/negative owner (accelerate/SwissKnife). |
| Bindings | `implemented` | `mcp_server/mcplusplus/bindings.py` | Covered by MCPP-023 joint bindings suite. |
| Durable | `blocked` | — | No DurableExecutor runtime bind task for datasets. |
| A2A | `blocked` | — | No datasets A2A adapter receipt. |

---

## 7. kit — adapter surfaces

Modules under `ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/` unless noted.

| Profile | Disposition | Primary adapter surface | Evidence / gap |
| --- | --- | --- | --- |
| A | `blocked` | — | No IDL registry module. |
| B | `partial` | `artifacts.py` | Artifact helpers used with UCAN admission; no MCPP-034-class envelope gate. |
| C | `implemented` | `ucan.py` (`UCANVerifier`), `revocation.py`, `delegation.py` | Cryptographic verifier + tamper/unsigned/alg-none/wrong-key/audience/expiry negatives (inventory §4; MCPP baseline kit). Command: UCAN verifier test. |
| D | `partial` | authorization dispatch / `tests/test_profile_d_policy.py` | Gates exist; not a full temporal deontic language. |
| E | `partial` | `mcp_server/p2p_transport.py`; readiness transport security | MCPP-065 bound kit mcp_server; parity tests present. Not the primary framing owner. |
| F | `partial` | `event_dag.py` | Event helpers; ZK not proof-verified forest-wide. |
| G | `partial` | `profile_g.py`, `profile_g_transport.py`, `coordination_storage.py` | MCPP-069 adapted kit coordination; authoritative fence gate recorded on accelerate. Kit transport command listed under supplementary. |
| H | `partial` | `profile_h.py`, `profile_h_http.py` | Paid kit tests exist; payment≠auth adversarial suite owned by MCPP-072 on accelerate. |
| Bindings | `blocked` | — | Dual MCP binding adapters are accelerate/datasets only. |
| Durable | `blocked` | — | No kit DurableExecutor bind. |
| A2A | `blocked` | — | No kit A2A adapter. |

---

## 8. SwissKnife — adapter surfaces

Paths relative to discovered checkout `/home/barberb/lift_coding/swissknife`. Origin remains `https://github.com/endomorphosis/swissknife`.

| Profile | Disposition | Primary adapter surface | Evidence / gap |
| --- | --- | --- | --- |
| A | `partial` | `src/services/mcp/mcp-idl.ts` | Full InterfaceRepository + CLI; local PASS matrix is SwissKnife-local, not four-language admission. Command supplementary. |
| B | `partial` | `src/services/mcp/mcp-envelope.ts` | Envelope/receipt CID builders + tests; receipt-signed forest level incomplete without shared vectors. |
| C | `partial` | `src/services/mcp/mcp-plus-plus-profile-c.ts`, `src/auth/ucan-auth.ts`, `src/auth/delegation-manager.ts` | Library-backed `@ucans/ucans` verify + revocation; strong runtime, not four-language crypto API. |
| D | `partial` | `src/services/logic/deontic/mcp-policy.ts`, deontic broker / remote engine | Rich policy/ORB surface; remote TDFOL depends on datasets availability. |
| E | `partial` | `src/services/mcp/mcp-p2p-session.ts`, pubsub bus | Session SM + rate limits; local PASS only. |
| F | `partial` | `src/services/mcp/mcp-event-dag.ts` (+ ceremony connectors) | Event DAG + circuit references; simulated vs real ZK still a forest gap. |
| G | `partial` | connector tests / profile-g connector | Connector evidence; fencing authority lives primarily in accelerate/kit Python paths. |
| H | `implemented` | `scripts/mcpplusplus-profile-h-adapter.cjs` | MCPP-072: HTTP **and** libp2p readiness, fail-closed unconfigured seller, header redaction. Command: profile-h-adapter test. Receipt: `runtime/profile-h-adapters.md`. Nested HEAD `afdbf885…`. |
| Bindings | `blocked` | — | Dual MCP binding adapters not landed as SwissKnife gap-closure output. |
| Durable | `blocked` | — | No SwissKnife DurableExecutor bind in gap-closure outputs. |
| A2A | `implemented` | `src/services/mcp/mcp-plus-plus-a2a.ts` (`SwissKnifeA2AAdapter@1`) | MCPP-057 Agent Card + two-agent handoff + fail-closed activation. Command: a2a-adapter test. Receipt: `runtime/swissknife-a2a.md`. |

---

## 9. Remaining gaps (do not silently upgrade to `implemented`)

| Gap | Runtimes | Notes |
| --- | --- | --- |
| No dedicated Profile A IDL on datasets/kit | datasets, kit | `blocked` until registry adapters exist |
| No dedicated Profile C verifier on datasets | datasets | Consume kit/accelerate; do not invent crypto |
| Accelerate Profile C default-on crypto | accelerate | Still optional/`require_signatures` path |
| Profile F always-on ZK verify | all | Simulated proofs must not claim `zero_knowledge: true` |
| Profile G attestation crypto | all | Signature fields structural in codecs |
| Dual bindings / Durable / A2A on kit | kit | Explicitly out of landed adapter set |
| Durable / A2A / bindings on datasets (except bindings) | datasets | Bindings done; durable/A2A blocked |
| SwissKnife self-PASS ≠ forest admission | SwissKnife A–E | Keep `partial` unless a gap-closure receipt elevates a cell |
| Four-language validators vs runtime adapters | Mcp-Plus-Plus | Spec validators remain structural-only for C/D; this matrix is **runtime** disposition only |

This task **does not** re-implement protocol bytes in runtimes (ownership: Mcp-Plus-Plus schemas/vectors/validators; runtimes own adapters only).

---

## 10. Dependency evidence map

| Landed task | What it contributes to this matrix |
| --- | --- |
| MCPP-023 | accelerate + datasets dual bindings → `implemented` |
| MCPP-034 | accelerate envelope emit/verify → Profile B `implemented` |
| MCPP-049 | datasets Profile D fail-closed evaluator → `implemented` |
| MCPP-053 | accelerate DurableExecutor bind → Durable `implemented` |
| MCPP-057 | SwissKnife A2A adapter + command → A2A `implemented` |
| MCPP-065 | datasets (+ kit bind) Profile E framing → datasets E `implemented` |
| MCPP-069 | accelerate (+ kit) Profile G stale fence → accelerate G `implemented` |
| MCPP-072 | accelerate + SwissKnife Profile H adapters + negatives → H `implemented` |

Baseline receipts under `docs/reports/mcplusplus-1.0-gap-closure/baseline/` record inventory SHAs and SwissKnife remote discovery (`swissknife-mcplusplus.json`).

---

## 11. Acceptance checklist (MCPP-081)

| Criterion | Result |
| --- | --- |
| Matrix lists each runtime | **yes** — accelerate, datasets, kit, SwissKnife (§2–§8) |
| Matrix lists each profile | **yes** — A–H plus Bindings / Durable / A2A (§3) |
| No `implemented` without a command | **yes** — §4 enumerates every implemented cell’s command; zero commandless `implemented` cells |
| SwissKnife remote remains discovered origin | **yes** — `https://github.com/endomorphosis/swissknife` via `git remote -v` / forest / MCPP-009; not invented |
| Dispositions are implemented / partial / blocked with evidence | **yes** — §3–§8 |
| Spec implementations not duplicated | **yes** — documentation-only adapter disposition matrix |

---

## 12. Validation (gap-closure gate)

```bash
test -s docs/reports/mcplusplus-1.0-gap-closure/runtime/adapter-matrix.md
```

Expected: exit 0, non-empty matrix file.

---

## 13. Downstream consumers

- **MCPP-082** implementation report / draft PRs should import this matrix for runtime closeout status.
- Do not promote a cell to `implemented` from SwissKnife `CONFORMANCE_MATRIX.md` or line-coverage trophies alone.
- Future adapter work should add a row/cell here with a command before claiming completion.

**End of RuntimeAdapterMatrix@1 for MCPP-081.**
