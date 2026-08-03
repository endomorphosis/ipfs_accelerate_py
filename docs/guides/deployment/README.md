# Deployment

**Status:** Current
**Audience:** Operators and developers choosing how to run the package on a host
**Scope:** Deployment surfaces, environment matrix (DeploymentCapabilityProfile@1),
local/CPU baseline, optional container and process-manager paths, health evidence
versus liveness, and recovery checks
**Non-goals:** Historical Docker-runner cache runbooks; cloud Terraform/Ansible
deep dives; inventing universal production defaults for every compose file under
`deployments/`; P2P mesh topology (see [P2P guide](../p2p/README.md)); full
hardware tuning (see [hardware overview](../hardware/overview.md))
**Last verified:** `73fd7229111c0553a42d0f11d2370ba1e6e95a45` (2026-08-03);
install entrypoints, compose files, capability probe, and MCP CLI checked against
this tree
**Source anchors:** `pyproject.toml` (`[project.optional-dependencies]`,
`[project.scripts]`), `ipfs_accelerate_py/__init__.py` (`get_instance`),
`ipfs_accelerate_py/ipfs_accelerate.py` (`get_capabilities`),
`ipfs_accelerate_py/cli_entry.py`, `docker-compose.yml`, `Dockerfile`,
`deployments/`, `install/`, `docs/architecture/overview.md`,
`docs/architecture/INFERENCE_RUNTIME.md`

IPFS Accelerate runs first as a **local Python library on CPU**. Optional
extras, model providers, GPU frameworks, browsers, IPFS, P2P, MCP, and the
agent supervisor are **capability-gated**: install and configure them only when
you need them, and verify them with probes and workload smoke tests. Import
success, a live PID, or a Docker healthcheck that only imports the package are
**not** proof that inference, network, or content-identity paths work.

This page is the maintained **DeploymentCapabilityProfile@1** operator landing
page. Workflow-specific files under `docs/guides/deployment/` and
`docs/guides/infrastructure/` remain available as historical or specialized
material; they are not the default path for new deployments.

---

## 1. Choose a deployment surface

| Surface | When to use | Baseline requirement | Optional |
| --- | --- | --- | --- |
| **Library in-process** | App embeds inference or routers | Python 3.8+, base or `minimal` install, CPU | GPU wheels, remote providers, IPFS |
| **Unified CLI** | Smoke tests, models/MCP helpers | Installed console script `ipfs-accelerate` | Feature extras matching the command |
| **MCP server process** | Tool clients and local MCP clients | `mcp` extra; bind to localhost until auth/TLS exist | `mcp-p2p`, dashboard, autoscaler |
| **Compose / Docker images in this checkout** | Isolated processes for a known compose file | Docker Engine; inspect the specific compose services | GPU devices, extra volumes, production reverse proxies |
| **`deployments/` workflow assets** | Team-specific systemd, k8s, SSL, monitoring sketches | Read and adapt; not universal package defaults | Cloud and cluster tooling |
| **Agent supervisor** | Maintainer/operator objective workloads | Separate control plane; not required for inference | Leases, providers, formal assurance |

Related journeys:

- [Installation](../getting-started/installation.md) — extras and version sources
- [Getting started](../getting-started/README.md) — first verified operation
- [Quick start](../QUICKSTART.md) — short local smoke
- [MCP setup](../MCP_SETUP_GUIDE.md) — canonical server entry
- [Hardware overview](../hardware/overview.md) — device discovery
- [P2P guide](../p2p/README.md) — optional distributed plane
- [Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md) — control plane
- [Architecture overview](../../architecture/overview.md) — planes and capability language

---

## 2. DeploymentCapabilityProfile@1 (environment matrix)

Record a profile **before** scaling or enabling optional services. The matrix
below is descriptive: every row except the CPU/local baseline may be `absent`
without breaking ordinary local inference.

| Profile field | Baseline (always plan for) | Optional (capability-gated) | Authoritative signal |
| --- | --- | --- | --- |
| **Python / package** | 3.8+; import + runtime `__version__` | Editable checkout vs published wheel | Import + `pip show` / packaging metadata (may disagree; quote source) |
| **Compute device** | **CPU** | CUDA, ROCm, OpenVINO, MPS, Qualcomm, browser WebNN/WebGPU | `get_capabilities(detail=True)["hardware"]` **and** a model smoke on that device |
| **Model / provider** | Local stub or already-cached model when available | Remote APIs, HF download, llama.cpp, full transformers stack | Successful `run_model` / router call for the intended task |
| **MCP** | Off | `mcp` server on configured host/port | `ipfs-accelerate mcp status` **plus** tool/manifest inspection |
| **IPFS / content identity** | Off | Kit, Kubo, verified IPLD paths | Backend selection receipt + rehash/admission — not synthetic `bafy…` cache keys |
| **P2P / TaskQueue** | Off | `mcp-p2p` / `libp2p`, peer identity, tokens | Queue auth + claim/complete evidence — not peer process liveness |
| **Docker / compose** | Off | Images under root and `deployments/` | Service-specific readiness; import-only healthchecks are weak |
| **Supervisor control plane** | Off | Objective/bundle daemons | Validation receipts and merge evidence — not PID alone |

### Minimum local baseline (CPU)

```bash
python -m pip install "ipfs-accelerate-py[minimal]"
python - <<'PY'
import ipfs_accelerate_py
from ipfs_accelerate_py import get_instance

print("runtime __version__:", ipfs_accelerate_py.__version__)
report = get_instance().get_capabilities(detail=True)
print("task_types:", report.get("task_types"))
print("hardware.available:", (report.get("hardware") or {}).get("available"))
print("hwtest (may be optimistic):", report.get("hwtest"))
PY
```

Notes:

- Prefer the detailed `hardware` block over `hwtest` alone. Internal
  `hwtest` values can be optimistic defaults in some code paths and are **not**
  a substitute for a device or model smoke test.
- Packaging metadata (`pyproject.toml` / `setup.py`) and the runtime
  `__version__` export may disagree; quote the source of any version string.
  See [installation](../getting-started/installation.md#version-sources-code-owned-disagreement).

---

## 3. Local or managed process (MCP)

Install only the extras you need, then bind development servers to localhost:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
ipfs-accelerate mcp --help
```

Use `--no-p2p` when the optional P2P service must stay disabled. P2P extras
install dependencies; they do not create a peer network or prove connectivity.

For a long-running service, put the selected process behind the deployment's
process manager and network boundary. Configure authentication, TLS, logging,
resource limits, readiness checks, and graceful shutdown **in that
environment**. The package CLI is not a substitute for those controls.

---

## 4. Containers in this checkout

The repository ships multiple Docker and compose layouts for different
workflows. Treat each file as a **named sketch**, not a single product default.

```bash
# Root multi-service sketch (inspect services and env before use)
docker compose -f docker-compose.yml config
docker build -f Dockerfile -t ipfs-accelerate-py .

# Optional CI-oriented compose
docker compose -f docker-compose.ci.yml config
```

Additional assets:

| Path | Role |
| --- | --- |
| `docker-compose.yml`, `Dockerfile` | Root application sketches |
| `deployments/docker/`, `deployments/docker-compose.yml` | Deployment-oriented variants |
| `deployments/systemd/`, `deployments/kubernetes.yaml` | Process and cluster sketches |
| `deployments/health_check.py` | Example operator script (environment-specific) |
| `install/requirements_*.txt`, `install/install.sh` | Hardware- and CI-oriented install helpers |

Inspect image targets, ports, volumes, and environment variables before shared
or production use. Image names and published ports are **not** universal
package contracts.

### Compose healthcheck caveat

Some compose files use a healthcheck equivalent to importing the package (or
printing `OK`). That is **process/import liveness**, not:

- model readiness
- GPU availability
- MCP tool registration
- IPFS content identity
- P2P queue correctness

Always pair container restarts with capability reports and workload smokes.

---

## 5. Scaling and parallelism

Scale only after measuring the selected model/provider and confirming the
resource report for that host.

- More processes can duplicate model memory, compete for one accelerator, or
  overload a remote provider.
- GPU, browser, IPFS, P2P, and paid providers remain optional; do not size a
  fleet as if every node has them.
- For the agent supervisor, admission is controlled by leases, CPU/memory,
  provider capacity, dependencies, conflicts, and **validation receipts** —
  not by an arbitrary worker count or process liveness.

---

## 6. Health, recovery, and what counts as evidence

| Signal | Use for | Does **not** prove |
| --- | --- | --- |
| Process PID / `restart: unless-stopped` | Process manager state | Correct inference or network path |
| Docker healthcheck that only imports | Container started and Python import works | Device, model, or CID path |
| `get_capabilities(detail=True)` | Discovery snapshot for this process | Future host state or successful inference |
| `hwtest` map | Coarse internal flags | Real accelerator readiness |
| Successful `run_model` / router smoke | Intended workload on chosen device | All models or remote providers |
| MCP status + tool/manifest query | Server reachable and tools registered | P2P mesh or IPFS durability |
| Backend selection + verified put/get receipts | Content-identity operations | Local cache synthetic keys are multiformats CIDs |
| TaskQueue claim/complete + auth | P2P task execution | Peer is merely online |
| Supervisor validation/merge receipts | Control-plane work product | Inference serving health |

### Recovery checklist

1. Capture Python executable, installed package identity, and capability report.
2. Reproduce with the **smallest** extra set (CPU/local first).
3. Disable optional planes (`--no-p2p`, no remote provider keys, no IPFS
   requirement) and re-test the baseline.
4. Re-enable one optional plane at a time with an explicit probe.
5. Prefer fail-closed errors on coordination CIDs and authenticated queues over
   silent success with synthetic identifiers.

See [Troubleshooting FAQ](../troubleshooting/faq.md) for symptom mapping.

---

## 7. Deployment checklist

- [ ] Install the smallest extras profile for the workload.
- [ ] Verify version sources and `get_capabilities(detail=True)`.
- [ ] Keep **CPU/local** as the proven path before optional accelerators.
- [ ] Pin model/provider versions and cache policy for reproducible deploys.
- [ ] Keep development MCP listeners on `127.0.0.1`.
- [ ] Configure authentication and TLS before any remote exposure.
- [ ] Set explicit CPU, memory, disk, and concurrency limits; add GPU limits
      only when a GPU path is verified.
- [ ] Use capability + smoke evidence for health; do not promote PID or
      import-only checks alone.
- [ ] Keep optional P2P and supervisor services separate from ordinary
      inference until dependencies and trust boundaries are verified.
- [ ] Exercise the same focused tests used by CI in the target environment
      (see [testing guide](../../development/testing.md)).

---

## 8. Historical and specialized deployment docs

These remain in-tree for workflow history; they are **not** the current
operator baseline:

- `docs/guides/deployment/DEPLOYMENT_GUIDE.md` and Docker cache plans
- `docs/guides/infrastructure/*` completion summaries and cache alignment notes
- `docs/guides/docker/*` for container CI/cache detail

Prefer this README, [installation](../getting-started/installation.md),
[Docker guide index](../docker/README.md), and architecture pages for new work.

---

## Related references

- [Architecture overview](../../architecture/overview.md)
- [Inference runtime](../../architecture/INFERENCE_RUNTIME.md)
- [Distributed runtime](../../architecture/DISTRIBUTED_RUNTIME.md)
- [Integration boundaries](../../architecture/INTEGRATION_BOUNDARIES.md)
- [Hardware guide](../hardware/overview.md)
- [P2P guide](../p2p/README.md)
- [Troubleshooting FAQ](../troubleshooting/faq.md)
- [Testing guide](../../development/testing.md)
