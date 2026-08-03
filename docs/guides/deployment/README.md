# Deployment

**Status:** Current
**Owner:** package maintainers
**Audience:** Operators and developers choosing how to run the package on a host
**Scope:** Deployment surfaces, environment matrix (DeploymentCapabilityProfile@1),
local/CPU baseline, optional container and process-manager paths, health evidence
versus liveness, and recovery checks
**Non-goals:** Historical Docker-runner cache runbooks; cloud Terraform/Ansible
deep dives; inventing universal production defaults for every compose file under
`deployments/`; P2P mesh topology (see [P2P guide](../p2p/README.md)); full
hardware tuning (see [hardware overview](../hardware/overview.md))
**Last-verified:** 2026-08-03 @ `d5f3aa5c6`; install entrypoints, compose files,
capability-constructor behavior, MCP/P2P CLI gates, and Python metadata/runtime
disagreement checked against this tree
**Sources:** `requirements.txt`; `pyproject.toml`
(`[project.optional-dependencies]`,
`[project.scripts]`), `ipfs_accelerate_py/__init__.py` (`get_instance`),
`ipfs_accelerate_py/ipfs_accelerate.py` (`get_capabilities`),
`ipfs_accelerate_py/cli_entry.py`, `docker-compose.yml`, `Dockerfile`,
`deployments/`, `install/`, `docs/architecture/overview.md`,
`docs/architecture/INFERENCE_RUNTIME.md`
**Freshness triggers:** packaging/Python metadata, core constructor,
capability-probe, MCP/P2P CLI, compose, deployment, or hardware-detector changes

IPFS Accelerate runs first as a **local Python library on CPU**. Optional model
providers, GPU frameworks, browsers, IPFS, P2P, MCP processes, and the agent
supervisor are **capability-gated**: configure and start them only when needed,
then verify them with probes and workload smoke tests. Current packaging does
not perfectly isolate optional dependencies: notably, base `requirements.txt`
already includes FastMCP and the Flask/PyGithub stack. Import success, a live
PID, or a Docker healthcheck that only imports the package is **not** proof that
inference, network, or content-identity paths work.

This page is the maintained **DeploymentCapabilityProfile@1** operator landing
page. Workflow-specific files under `docs/guides/deployment/` and
`docs/guides/infrastructure/` remain available as historical or specialized
material; they are not the default path for new deployments.

---

## 1. Choose a deployment surface

| Surface | When to use | Baseline requirement | Optional |
| --- | --- | --- | --- |
| **Library in-process** | App embeds inference or routers | Python matching the runtime caveat below, base install, CPU | GPU wheels, remote providers, IPFS |
| **Unified CLI** | Smoke tests, models/MCP helpers | Installed console script `ipfs-accelerate` | Feature extras matching the command |
| **MCP server process** | Tool clients and local MCP clients | Base metadata already includes core MCP deps; `[mcp]` expresses intent; verify FastAPI/Uvicorn; bind locally | `mcp-p2p`, dashboard, autoscaler |
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
| **Python / package** | Metadata declares 3.8+, but core pre-3.10 compatibility is unproven; see below | Editable checkout vs published wheel | Import on the exact Python + `pip show` / metadata (quote both on disagreement) |
| **Compute device** | **CPU** | CUDA, ROCm, OpenVINO, MPS, Qualcomm, browser WebNN/WebGPU | Direct `HardwareDetector` result **and** a model smoke on that device |
| **Model / provider** | Local stub or already-cached model when available | Remote APIs, HF download, llama.cpp, full transformers stack | Successful `run_model` / router call for the intended task |
| **MCP** | Off | `mcp` server on configured host/port | `ipfs-accelerate mcp status` **plus** tool/manifest inspection |
| **IPFS / content identity** | Off | Kit, Kubo, verified IPLD paths | Backend selection receipt + rehash/admission — not synthetic `bafy…` cache keys |
| **P2P / TaskQueue** | Off | `mcp-p2p` / `libp2p`, peer identity, tokens | Queue auth + claim/complete evidence — not peer process liveness |
| **Docker / compose** | Off | Images under root and `deployments/` | Service-specific readiness; import-only healthchecks are weak |
| **Supervisor control plane** | Off | Objective/bundle daemons | Validation receipts and merge evidence — not PID alone |

### Side-effect-minimized local baseline (CPU)

```bash
python -m pip install "ipfs-accelerate-py"
IPFS_ACCEL_SKIP_CORE=1 python - <<'PY'
import ipfs_accelerate_py
from ipfs_accelerate_py.hf_model_server.hardware.detector import HardwareDetector

print("runtime __version__:", ipfs_accelerate_py.__version__)
detector = HardwareDetector()
print("hardware.available:", detector.get_available_hardware())
PY
```

Notes:

- `IPFS_ACCEL_SKIP_CORE=1` avoids the heavy core import and the singleton
  constructor. This command proves package import plus direct hardware
  discovery only; it does **not** prove model inference or full runtime
  capability.
- Extras are additive. `[minimal]` does not subtract the dependencies in
  `requirements.txt`, so it is not a smaller base wheel under current metadata.
- Run a small, already-local model on `device="cpu"` as the separate workload
  proof; no repository-wide model is guaranteed to be downloaded or cached.
- If you later run the full capability report, prefer its detailed `hardware`
  block over `hwtest` alone. Internal `hwtest` values can be optimistic defaults
  in some code paths and are **not** a substitute for a device or model smoke.
- Packaging metadata (`pyproject.toml` / `setup.py`) and the runtime
  `__version__` export may disagree; quote the source of any version string.
  See [installation](../getting-started/installation.md#version-sources-code-owned-disagreement).

#### Full runtime capability snapshot has constructor side effects

`get_instance().get_capabilities(detail=True)` is **not** a side-effect-free
offline probe on this tree. `get_instance()` constructs the core singleton;
its constructor initializes the storage wrapper and API adapters. Depending on
installed sibling packages and configuration, construction can touch cache or
configuration files, contact storage/IPFS endpoints, or attempt optional daemon
initialization before returning the report. Run it only in an environment where
those effects and network policy are acceptable:

```python
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print(report)
```

There is no proven side-effect-free top-level replacement for the complete
report at this revision. That is a code-owned gap; use the direct hardware
probe above for the bounded local baseline.

---

## 3. Local or managed process (MCP)

Current base dependencies already include FastMCP and the Flask/PyGithub stack.
The `mcp` extra repeats that set and adds `async-timeout`; use it as an explicit
deployment profile, not as proof that MCP was absent from the base install.
Then bind development servers to localhost:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
ipfs-accelerate mcp --help
```

The TaskQueue P2P service is off unless
`IPFS_ACCELERATE_PY_MCP_P2P_SERVICE=1` is set. Leave that variable unset (or
set it to `0`) to keep the service disabled. The MCP `--no-p2p` flag controls
only P2P workflow monitoring inside the GitHub autoscaler; it does **not**
suppress an environment-enabled TaskQueue service. P2P extras install
dependencies; they do not create a peer network or prove connectivity.

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
| `get_capabilities(detail=True)` | Full discovery snapshot after constructing the runtime | Side-effect-free/offline inspection, future host state, or successful inference |
| `hwtest` map | Coarse internal flags | Real accelerator readiness |
| Successful `run_model` / router smoke | Intended workload on chosen device | All models or remote providers |
| MCP status + tool/manifest query | Server reachable and tools registered | P2P mesh or IPFS durability |
| Backend selection + verified put/get receipts | Content-identity operations | Local cache synthetic keys are multiformats CIDs |
| TaskQueue claim/complete + auth | P2P task execution | Peer is merely online |
| Supervisor validation/merge receipts | Control-plane work product | Inference serving health |

### Recovery checklist

1. Capture the Python executable, installed package identity, and direct
   hardware report; capture the full capability report only where constructor
   effects are permitted.
2. Reproduce with the **smallest** extra set (CPU/local first).
3. Disable optional planes (leave `IPFS_ACCELERATE_PY_MCP_P2P_SERVICE` unset or
   `0`, remove remote provider keys, and require no IPFS path) and re-test the
   baseline. `--no-p2p` only disables autoscaler workflow monitoring.
4. Re-enable one optional plane at a time with an explicit probe.
5. Prefer fail-closed errors on coordination CIDs and authenticated queues over
   silent success with synthetic identifiers.

See [Troubleshooting FAQ](../troubleshooting/faq.md) for symptom mapping.

---

## 7. Deployment checklist

- [ ] Install the smallest extras profile for the workload.
- [ ] Verify version sources and run the side-effect-minimized hardware probe;
      run `get_capabilities(detail=True)` only where constructor effects are allowed.
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
