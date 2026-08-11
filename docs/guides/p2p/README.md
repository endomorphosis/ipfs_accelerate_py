# P2P and Distributed Workflows

**Status:** Current
**Owner:** package maintainers / distributed-runtime maintainers
**Audience:** Operators and integrators enabling optional IPFS, libp2p,
TaskQueue, or workflow scheduling
**Scope:** P2POperatorJourney@1 — prerequisites, install extras, MCP-backed
enablement, code boundaries, degraded operation, health evidence, and routing
to architecture and historical guides
**Non-goals:** Claiming a global peer mesh; treating process liveness as
queue health; treating synthetic `bafy…` cache keys as multiformats CIDs;
rewriting historical infrastructure fix notes
**Last-verified:** 2026-08-03 @ `d5f3aa5c6`; TaskQueue environment gate,
autoscaler flag semantics, P2P modules, and distributed-runtime guide checked
against this tree
**Sources:** `ipfs_accelerate_py/cli.py` (`run_mcp_start`,
`_maybe_start_taskqueue_p2p_from_env`, `--no-p2p`);
`ipfs_accelerate_py/p2p_tasks/` (`task_queue`, `protocol`,
`peer_trust`, `libp2p_runtime`, `mcp_p2p*`), `ipfs_accelerate_py/p2p_workflow_scheduler.py`,
`ipfs_accelerate_py/p2p_workflow_discovery.py`,
`ipfs_accelerate_py/mcp_server/`, `ipfs_accelerate_py/mcp/tools/`,
`pyproject.toml` extras `mcp-p2p` and `libp2p`,
`docs/architecture/DISTRIBUTED_RUNTIME.md`,
`docs/architecture/INTEGRATION_BOUNDARIES.md`
**Freshness triggers:** MCP CLI, TaskQueue enablement/auth, P2P environment,
protocol, workflow-monitoring, or distributed-runtime contract changes

IPFS, libp2p, TaskQueue, and distributed workflow support are **optional**.
They sit **beside** the local inference path. Core CPU/local inference does
**not** require a peer, an IPFS daemon, or shared tokens.

This page is the operator journey. Semantics for backend roles, real CIDv1
versus synthetic cache keys, trust tiers, and fail-closed coordination live in
[Distributed runtime](../../architecture/DISTRIBUTED_RUNTIME.md). Sibling
package ownership lives in
[Integration boundaries](../../architecture/INTEGRATION_BOUNDARIES.md).

---

## 1. P2POperatorJourney@1 (end-to-end)

```text
0. Prove side-effect-minimized CPU/local baseline without P2P
        |
1. Install optional extras (mcp-p2p and/or libp2p)
        |
2. Configure identity, auth token, ports, bootstrap, queue limits
        |
3. Start MCP (or TaskQueue service) with P2P intentionally enabled
        |
4. Probe capabilities + auth; do not stop at "process is up"
        |
5. Submit/claim/complete a controlled task OR run focused tests
        |
6. Record degradation when peers/IPFS are absent; fail closed for p2p-only work
```

| Step | Success evidence | Failure / non-evidence |
| --- | --- | --- |
| 0 Baseline | Heavy-core-skipped import + direct hardware report, with TaskQueue gate off | Skipping baseline and debugging only the mesh |
| 1 Install | Extras resolve; modules import | Import alone ≠ mesh ready |
| 2 Configure | Documented env (token, ports, bootstrap) present | Empty defaults in production |
| 3 Start | Configured listener; MCP status when using MCP | PID or Docker import healthcheck alone |
| 4 Probe | Capability/MCP manifest; auth_ok path | Peer list without auth or queue state |
| 5 Work | Claim/complete or verified test receipt | Synthetic cache key as “CID proof” |
| 6 Degrade | Explicit skip/fail for missing P2P | Silent “success” for `p2p-only` work |

---

## 2. Install optional capabilities

```bash
# MCP + P2P TaskQueue dependency set
python -m pip install "ipfs-accelerate-py[mcp-p2p]"

# Lower-level libp2p dependency set when MCP is not required
python -m pip install "ipfs-accelerate-py[libp2p]"
```

Extras install **Python dependencies**, not:

- a running libp2p swarm
- bootstrap peers
- shared authentication material
- an IPFS daemon or verified CID backend

Configure network policy, identity, and queue parameters separately. System
build dependencies (for example headers needed by native wheels) may be
required on some hosts; treat those as environment-specific.

---

## 3. MCP-backed P2P (preferred product path)

The canonical MCP runtime owns the current server and P2P integration boundary:

```bash
ipfs-accelerate mcp --help

# TaskQueue P2P is opt-in. Use a real secret source instead of this placeholder.
export IPFS_ACCELERATE_PY_MCP_P2P_SERVICE=1
export IPFS_ACCELERATE_PY_TASK_P2P_TOKEN="<shared-token>"
export IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT=9100
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Without `IPFS_ACCELERATE_PY_MCP_P2P_SERVICE=1`, the product MCP start path does
not start the TaskQueue P2P service. Configure queue path, bootstrap/discovery,
listen address, and trust policy for the deployment as needed; the three
variables above are only the minimum enable/auth/listen shape.

Useful controls:

- Keep development listeners on `127.0.0.1` until authentication and network
  policy exist.
- To keep TaskQueue P2P disabled, leave
  `IPFS_ACCELERATE_PY_MCP_P2P_SERVICE` unset or set it to `0`.
- `mcp start --no-p2p` disables only P2P workflow monitoring in the GitHub
  autoscaler. It does **not** override the TaskQueue environment gate and does
  not stop a service enabled with `IPFS_ACCELERATE_PY_MCP_P2P_SERVICE=1`.
- Inspect direct module help when embedding:

```bash
python -m ipfs_accelerate_py.mcp.cli --help
```

Do not use `get_instance().get_capabilities(detail=True)` as an offline P2P
preflight. Constructing that singleton initializes storage/API adapters and can
touch files, contact configured storage/IPFS endpoints, or attempt optional
daemon initialization. Use MCP status/manifest inspection plus an authenticated
queue claim/complete as the live proof.

There is **no** general-purpose `ipfs-accelerate p2p start` top-level command
in the current product CLI. Older examples using that command are historical.

### Auth and trust (operator minimum)

When TaskQueue trust tiers or shared tokens are configured (see
`docs/architecture/DISTRIBUTED_RUNTIME.md` and env names such as
`IPFS_ACCELERATE_PY_TASK_P2P_TOKEN` / trust-tier flags):

- Shared token or configured auth material is required for RPC that mutates
  queue state.
- Peer connectivity without auth is **not** authorization.
- Trust tiers (TRUSTED / ELEVATED / BASELINE) affect claim priority envelopes;
  they do not certify task result correctness.

---

## 4. Code and test boundaries

Implementation is spread across optional modules rather than one public “P2P
facade”:

| Concern | Primary path |
| --- | --- |
| TaskQueue / libp2p runtime | `ipfs_accelerate_py/p2p_tasks/` |
| Protocols | `p2p_tasks/protocol.py` (MCP++ `/mcp+p2p/1.0.0`; optional legacy NDJSON) |
| Peer trust | `p2p_tasks/peer_trust.py` |
| Workflow discovery | `ipfs_accelerate_py/p2p_workflow_discovery.py` |
| Workflow scheduling | `ipfs_accelerate_py/p2p_workflow_scheduler.py` |
| MCP transport integration | `ipfs_accelerate_py/mcp_server/` |
| Tool adapters | `ipfs_accelerate_py/mcp/tools/` |
| Compatibility / MCP++ module | `ipfs_accelerate_py/mcplusplus_module/p2p/` (not a second product root) |

Use MCP and P2P tests as the conformance surface. Networked tests need optional
packages, ports, peer identity, and an explicit opt-in:

```bash
# Start with import/manifest and offline unit coverage; add live network later
python -m pytest ipfs_accelerate_py/mcp/tests -q
```

Prefer isolated environments for multi-peer experiments. Absence of peers is
“no work claimed,” not “task succeeded.”

---

## 5. Content identity vs P2P execution

Keep these planes separate:

| Plane | Good evidence | Not evidence |
| --- | --- | --- |
| **Immutable content** | Verified put/get with rehash under conformant CIDv1 | Synthetic HF/kit `bafy…` cache keys |
| **Mutable queue** | DuckDB lease/claim/heartbeat/complete | IPFS pin or CAR alone |
| **P2P RPC** | Auth + protocol success + queue state change | Process up / port open only |
| **Workflow tags** | `p2p-only` fail closed without mesh; `p2p-eligible` may degrade | Silent GitHub-only success for `p2p-only` |

Details: [Distributed runtime](../../architecture/DISTRIBUTED_RUNTIME.md).

---

## 6. Operational checklist

- [ ] Prove local inference with
      `IPFS_ACCELERATE_PY_MCP_P2P_SERVICE` unset or `0`.
- [ ] Pin optional dependency versions; record peer/runtime versions.
- [ ] Configure peer identity, bootstrap addresses, queue limits, and timeouts.
- [ ] Keep control-plane and data-plane ports private until authenticated.
- [ ] Bound task payloads, concurrency, retries, and cache retention.
- [ ] Record real content identifiers and receipts for artifacts shared between
      peers; reject synthetic keys for coordination.
- [ ] Test degraded operation with IPFS/P2P disabled.
- [ ] Monitor memory and shutdown; distributed caches amplify artifact size.
- [ ] Do not treat process liveness or import-only Docker healthchecks as
      queue or content proof.

---

## 7. Historical P2P and cache docs (routing only)

Files under `docs/guides/p2p/` with setup runbooks, encryption notes, deadlock
fixes, and GitHub Actions P2P experiments are **workflow history**. Prefer:

1. This README for the operator journey
2. [Distributed runtime](../../architecture/DISTRIBUTED_RUNTIME.md) for
   semantics
3. [MCP setup](../MCP_SETUP_GUIDE.md) for server start
4. Historical pages only when debugging a matching specialized workflow

Examples of historical routing targets (non-normative):
`P2P_SETUP_GUIDE.md`, `P2P_CACHE_QUICK_REF.md`,
`GITHUB_ACTIONS_P2P_SETUP.md`, `GITHUB_P2P_CACHE_TWO_LAPTOP_RUNBOOK.md`,
`LIBP2P_UNIVERSAL_CONNECTIVITY.md`.

---

## Related documentation

- [MCP setup](../MCP_SETUP_GUIDE.md)
- [Deployment](../deployment/README.md)
- [Hardware overview](../hardware/overview.md)
- [Troubleshooting FAQ](../troubleshooting/faq.md)
- [Distributed runtime](../../architecture/DISTRIBUTED_RUNTIME.md)
- [Integration boundaries](../../architecture/INTEGRATION_BOUNDARIES.md)
- [MCP/P2P feature notes](../../features/mcp-integration/p2p-integration.md)
- [Testing](../../development/testing.md)
