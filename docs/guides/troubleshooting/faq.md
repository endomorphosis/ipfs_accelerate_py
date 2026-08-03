# Frequently Asked Questions

**Status:** Current
**Audience:** Operators and developers diagnosing install, inference, hardware,
MCP, IPFS/P2P, and CLI failures
**Scope:** FailureSymptomMap@1 — common symptoms mapped to bounded diagnostics,
capability-first expectations, and links to maintained guides
**Non-goals:** Exhaustive vendor error catalogs; rewriting historical
installation fix blogs; promising GPU, browser, IPFS, P2P, or remote providers
on every host
**Last verified:** `73fd7229111c0553a42d0f11d2370ba1e6e95a45` (2026-08-03);
aligned with installation, deployment, hardware, and P2P landing guides
**Source anchors:** `pyproject.toml`, `ipfs_accelerate_py/__init__.py`,
`ipfs_accelerate_py/ipfs_accelerate.py` (`get_capabilities`, `run_model`),
`ipfs_accelerate_py/cli_entry.py`, `ipfs_accelerate_py/p2p_tasks/`,
`docs/architecture/DISTRIBUTED_RUNTIME.md`,
`docs/architecture/INFERENCE_RUNTIME.md`

This page answers current setup and runtime questions. For a complete
installation sequence, see [Getting started](../getting-started/README.md) and
the [installation guide](../getting-started/installation.md).

**Working rules:**

1. **CPU/local is the baseline.** Optional planes may be absent.
2. **Import ≠ capability ≠ proof.**
3. **Process liveness is not health.** PIDs, restarts, and import-only Docker
   healthchecks do not prove inference, devices, CIDs, or queues.
4. **Synthetic identifiers are not multiformats CIDs.** Local cache keys that
   look like `bafy…` strings are not network content identity.

---

## FailureSymptomMap@1 (quick index)

| Symptom | First checks | Not a fix |
| --- | --- | --- |
| Package will not import | Python version ≥ 3.8; venv; `pip show`; extras only as needed | Installing every optional extra |
| “Works in shell, fails as service” | Same Python, env vars, working directory, secrets | Assuming PID restart healed config |
| Inference slow | Capability report; model already local; batch/precision; first-run download | Enabling P2P or supervisor lanes blindly |
| GPU missing | `nvidia-smi` / vendor tools **and** framework `is_available()`; then package report | Trusting `hwtest` alone |
| MCP will not start | `mcp` extra; port free; `mcp start --help`; bind host | Assuming public bind is required |
| IPFS / CID errors | Backend role + verified admission; offline path without IPFS | Treating synthetic cache keys as CIDs |
| P2P “up” but no work | Auth token; queue state; peer trust; `--no-p2p` baseline | Peer process liveness only |
| Provider / auth failures | Credential scope; explicit provider pin; network | Claiming universal provider availability |
| Offline mode fails | Local models and wheels present; remote extras disabled | Expecting downloads without network |
| Tests flake | Focused offline tests first; skip optional hardware/network | Hiding failures by installing unrelated extras |

---

## Installation

### What are the minimum requirements?

Use **Python 3.8 or newer**. The base package targets **CPU/local** operation.
GPU frameworks, browser stacks, IPFS, MCP, P2P, and analysis features are
optional extras or host environments. Install only what the deployment needs:

```bash
python -m pip install "ipfs-accelerate-py[minimal]"
python -m pip install "ipfs-accelerate-py[mcp]"
```

Extras are defined in `pyproject.toml`. They are **not** a promise that every
optional backend is present. There is no packaging extra named `cuda`,
`openvino`, or `rocm`; see [installation](../getting-started/installation.md).

### Can I install from a checkout?

Yes:

```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
python -m pip install -e .
```

The [installation guide](../getting-started/installation.md) covers source
builds, CUDA wheel selection, and optional extras.

### Why do two version numbers disagree?

Packaging metadata (`pyproject.toml` / `setup.py`) and the runtime
`ipfs_accelerate_py.__version__` export can disagree on a given tree. Quote
the source of any version string; do not invent a single “true” product version
in prose. See
[version sources](../getting-started/installation.md#version-sources-code-owned-disagreement).

### Can I run it offline?

Yes, when models and dependencies are already available locally. Model
downloads, IPFS, remote providers, and P2P services need network access; they
are optional and should be disabled or configured explicitly for offline
deployments.

---

## API and inference

### What is the main Python API?

The package-level API is the stable starting point:

```python
import ipfs_accelerate_py
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
print(accelerator.get_capabilities(detail=True))
```

The runtime also exposes `run_model` and modality routers for common inference
paths. Do not use the retired `IPFSAccelerator` class name from older examples;
inspect [API overview](../../api/overview.md) for current exports. See
[Inference runtime](../../architecture/INFERENCE_RUNTIME.md) for request flow.

### Which models and providers are supported?

Support depends on the installed backend, model format, provider credentials,
and local hardware. There is **no** fixed model-count guarantee. Query
capabilities and run a small model-specific smoke before planning production
capacity. Explicit provider pins must not be assumed to fall back across
boundaries without policy.

### Why is inference slow?

Check in order:

1. Confirm the intended backend with `get_capabilities(detail=True)`.
2. Confirm the model is already cached or locally available.
3. Use batching, an appropriate model size, and a supported precision.
4. Measure first-run download time separately from steady-state inference.
5. Check provider-specific logs before changing scheduler or P2P settings.

The [hardware guide](../hardware/overview.md) documents capability checks and
device smokes.

### Why does a GPU not appear?

The installed framework wheel must match the driver and architecture. Check the
backend directly, then the package capability report:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

python - <<'PY'
from ipfs_accelerate_py import get_instance
r = get_instance().get_capabilities(detail=True)
print("hardware:", r.get("hardware"))
print("hwtest (may be optimistic):", r.get("hwtest"))
PY
```

If the framework check is positive but the accelerator report is not, compare
the environment used by the shell, service unit, and worker. Optional backends
should **fail with a capability report**, not be assumed available.
**Do not** treat internal `hwtest` flags alone as proof.

---

## IPFS, P2P, and MCP

### Do I need a local IPFS daemon?

**No.** Core local inference does not require IPFS. IPFS, libp2p, and P2P task
queues are optional and have their own binaries, credentials, ports, and
lifecycle requirements. See [P2P guide](../p2p/README.md) and
[Distributed runtime](../../architecture/DISTRIBUTED_RUNTIME.md).

### A tool returned a `bafy…` string — is that a CID?

**Not always.** Some cache and compatibility paths emit synthetic identifiers
that look like CIDs but are not multiformats CIDv1 under the frozen profile.
Coordination and verified put/get paths must rehash and admit real CIDs. Do not
treat synthetic keys as network durability or proof. See
[Distributed runtime § synthetic keys](../../architecture/DISTRIBUTED_RUNTIME.md).

### How do I start MCP?

Install the MCP extra and use the product CLI:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

See [MCP setup](../MCP_SETUP_GUIDE.md) for direct module entry points,
capability inspection, and P2P notes (`--no-p2p`).

### MCP (or a container) is “healthy” but tools fail

Check:

1. Capability/MCP manifest tools list (not only HTTP 200 or import OK).
2. Whether the process has the same extras and env as your interactive shell.
3. Whether P2P or remote providers were expected but never configured.

Compose healthchecks that only `import ipfs_accelerate_py` prove **import
liveness**, not tool registration or model readiness. See
[Deployment health evidence](../deployment/README.md#6-health-recovery-and-what-counts-as-evidence).

### Peers are online but tasks never complete

1. Re-run baseline with P2P disabled to confirm local execution.
2. Verify shared token / auth configuration.
3. Inspect queue lease/claim/heartbeat state (mutable DuckDB coordination), not
   only peer connectivity.
4. Confirm workflow tags: `p2p-only` must not silently succeed via a non-P2P
   path; `p2p-eligible` may degrade when policy allows.

Process liveness alone is not queue health or proof of correct execution.

### Is my data sent to a remote service?

Local execution stays local unless a remote model/provider, IPFS, P2P, or other
network integration is enabled. Review provider and deployment configuration
before processing sensitive material.

---

## CLI and supervisor

### Which CLI commands are current?

Run `ipfs-accelerate --help` for the installed command set. Current top-level
groups commonly include `mcp`, `github`, `copilot`, `copilot-sdk`, `text`,
`audio`, `vision`, `multimodal`, `specialized`, and `models`. Older examples
using `inference`, `hardware`, `workflow`, or `network` as top-level product
groups are not current documentation.

### How do I run the agent supervisor?

The supervisor is an optional **maintainer/operator** surface, not required for
ordinary inference. Start with the
[Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md). Completion and merge
authority depend on validation receipts and policy — **not** on a daemon PID
alone.

---

## Browser support

Browser execution depends on the browser, runtime, and WebNN/WebGPU support on
the target machine. Consult the
[WebNN/WebGPU feature guide](../../features/webnn-webgpu/WEBNN_WEBGPU_README.md)
and test the target browser rather than relying on a hard-coded compatibility
table. Browser acceleration is never part of the CPU/local baseline.

---

## Testing and recovery

Run focused current checks from the
[testing guide](../../development/testing.md):

```bash
python -m pytest test/test_unified_cli_integration.py -q
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
```

When a command fails, capture:

1. First traceback
2. Python executable (`sys.executable`)
3. Installed package identity (`pip show` and/or runtime `__version__`)
4. `get_capabilities(detail=True)` output

That set is more useful than retrying with unrelated optional extras.

### Recovery sequence (bounded)

1. Reduce to CPU/local extras and disable P2P/IPFS/remote providers.
2. Re-verify import + capability report.
3. Re-enable one optional plane at a time with an explicit probe.
4. For coordination failures, prefer fail-closed CID/auth errors over silent
   synthetic success.

---

## Related maintained guides

- [Deployment](../deployment/README.md)
- [Hardware overview](../hardware/overview.md)
- [P2P](../p2p/README.md)
- [Installation](../getting-started/installation.md)
- [MCP setup](../MCP_SETUP_GUIDE.md)
- [Architecture overview](../../architecture/overview.md)

Historical troubleshooting under
`docs/guides/troubleshooting/INSTALLATION_TROUBLESHOOTING_GUIDE.md` and various
infrastructure “fix” notes may help for specialized incidents; they are not
the default current FAQ.
