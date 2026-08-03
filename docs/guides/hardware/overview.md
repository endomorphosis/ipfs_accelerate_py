# Hardware and Runtime Capability Guide

**Status:** Current
**Audience:** Developers and operators selecting devices and diagnosing
accelerator availability
**Scope:** Runtime capability discovery, CPU baseline, optional accelerators
(CUDA, ROCm, OpenVINO, MPS, Qualcomm, WebNN/WebGPU), measurement guidance, and
what does **not** count as hardware proof
**Non-goals:** Vendor driver installation manuals; guaranteeing any GPU or
browser backend; packaging extras named `cuda`/`rocm`/`openvino` (they do not
exist); full deployment process management (see
[deployment README](../deployment/README.md))
**Last verified:** `73fd7229111c0553a42d0f11d2370ba1e6e95a45` (2026-08-03);
`get_capabilities`, hardware detector import path, and install torch helper
paths checked against this tree
**Source anchors:** `ipfs_accelerate_py/ipfs_accelerate.py`
(`get_capabilities`, `hwtest`),
`ipfs_accelerate_py/hf_model_server/hardware/detector.py` (`HardwareDetector`),
`ipfs_accelerate_py/kit/hardware_kit.py`, `install/requirements_torch_*.txt`,
`install/requirements_cuda.txt`, `install/requirements_openvino.txt`,
`install/requirements_apple.txt`, `install/requirements_qualcomm.txt`,
`pyproject.toml` (extras; no dedicated CUDA packaging extra),
`test/hardware/`, `test/hardware_detection/`

Hardware support is **discovered at runtime**. The package can import on a host
without CUDA, ROCm, OpenVINO, MPS, WebNN, WebGPU, or Qualcomm support.
**CPU/local operation is the baseline.** GPU, browser, and vendor accelerators
are optional and must never be documented or operated as if they were always
present.

A driver/package probe is not a substitute for a model smoke test. Process
liveness and synthetic identifiers (for example cache keys that look like
CIDs) are never hardware proof.

---

## 1. Discover capabilities

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print("available accelerators:", (report.get("hardware") or {}).get("available", []))
print("hardware details:", report.get("hardware", {}))
print("hwtest (coarse/internal; may be optimistic):", report.get("hwtest", {}))
print("task_types:", report.get("task_types", []))
print("models registered:", report.get("models", []))
print("mcp counts:", (report.get("mcp") or {}).get("counts"))
PY
```

| Field | Meaning | Caveat |
| --- | --- | --- |
| `hardware` (detail) | Best-effort platform + accelerator map from `HardwareDetector` when importable | May be partial if optional detectors or frameworks are missing |
| `hardware.available` | Names currently flagged available | Not a throughput SLA |
| `hwtest` | Coarse internal flags | Can be optimistic; **do not** treat alone as proof |
| `models` / endpoints | Registered handlers in this process | Empty until endpoints are configured |
| `mcp` | Manifest summary when a server instance is visible | Absence is normal without MCP |

The report is JSON-friendly and intended for CLI, MCP, and planners. It is a
**discovery snapshot**, not a lease, receipt, or health certificate.

---

## 2. CPU baseline (always plan for this path)

CPU is the supported default for install verification, CI-friendly smoke tests,
and hosts without accelerators.

```bash
# Optional: bound numerical library threads before importing the model stack
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python - <<'PY'
from ipfs_accelerate_py import ipfs_accelerate_py

# Prefer a small, already-cached or mock-friendly path in your environment.
# Device selection must match what the provider actually supports.
accelerator = ipfs_accelerate_py({}, {})
# Example shape only — model availability is environment-specific:
# result = accelerator.run_model("bert-base-uncased", {"input_ids": [[101, 2023, 2003, 102]]}, device="cpu")
print("cpu path is the baseline; select models present in your environment")
PY
```

### ARM and architecture notes

Do not assume an ARM machine provides the same kernels or model/provider
coverage as x86. Record architecture, Python version, package versions, model
id, batch shape, and device in any performance report. See
`install/requirements_*.txt` for architecture-oriented helper requirement sets;
they are inputs, not guarantees.

---

## 3. Optional accelerators

There are **no** packaging extras named `cuda`, `openvino`, or `rocm`. Those
backends depend on host drivers and framework wheels. See
[installation](../getting-started/installation.md) for torch CUDA wheel
selection and helper scripts under `install/` and `scripts/`.

### CUDA

Check the driver-visible device **and** the PyTorch build before selecting
`device="cuda"`:

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Only after both checks pass, run a model smoke on `device="cuda"`. The model
provider and task must also support that device. A green `nvidia-smi` with a
CPU-only torch wheel still means **no CUDA path**.

Helper requirement files (optional): `install/requirements_cuda.txt`,
`install/requirements_torch_cu124.txt`,
`install/requirements_torch_cu130_nightly.txt`.

### ROCm, OpenVINO, MPS, Qualcomm

| Backend | Typical host signal | Notes |
| --- | --- | --- |
| **ROCm** | ROCm-capable GPU + PyTorch ROCm wheel | Not the CUDA wheel; follow vendor docs |
| **OpenVINO** | OpenVINO runtime + compatible model path | See `install/requirements_openvino.txt` |
| **MPS** | Apple Silicon + compatible PyTorch | Only on supported Apple hardware |
| **Qualcomm** | Vendor stack present | See `install/requirements_qualcomm.txt` and hardware_detection tests |

Install the vendor runtime first, then re-run `get_capabilities(detail=True)`.
Absence of a backend is a **capability report**, not a package defect.

### WebNN and WebGPU (browser)

Browser acceleration is a **separate runtime** from Python device selection.
Install the `webnn` extra and browser tooling only when you need that path, then
follow
[WebNN/WebGPU integration](../../features/webnn-webgpu/WEBNN_WEBGPU_README.md).
Support varies by browser, driver, and flags. Do not promise browser
acceleration in deployment profiles that never open a browser.

---

## 4. Throughput and memory

Measure the workload you intend to deploy. Variables that matter:

- model identity and weights location (local vs download)
- sequence / image / audio size and batch size
- precision and device
- warm-up count, concurrency, and cache state
- host CPU/RAM limits and whether other processes share the accelerator

Avoid copying benchmark numbers from historical reports without their commit
SHA and hardware context.

For the agent supervisor, parallelism is admission-controlled by CPU, memory,
disk, provider capacity, task conflicts, dependencies, and leases. Increasing a
lane count does not necessarily increase throughput. See the
[Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md).

---

## 5. What is not hardware health or proof

| Signal | Not sufficient because |
| --- | --- |
| Package import | Optional detectors and wheels may be missing |
| PID or service restart | Process can live without a usable device |
| Docker healthcheck that only imports | Same as import liveness |
| `hwtest` true flags alone | May be coarse or optimistic |
| Synthetic content keys / cache tokens | Unrelated to device capability; not multiformats proof |
| Historical “8 platforms” marketing text | Environment-specific; verify on the host |

**Authoritative operator sequence:** capability report → framework device check
→ small model smoke on the intended device → only then plan capacity.

---

## 6. Focused hardware tests (optional)

When you have the matching extras and devices:

```bash
python -m pytest test/hardware_detection -q
python -m pytest test/hardware -q
```

Skip or deselect tests that require GPUs, browsers, or vendor stacks that are
not present. Missing optional hardware should **report** as unavailable, not be
hidden as a silent pass.

---

## Related references

- [Installation](../getting-started/installation.md)
- [Deployment](../deployment/README.md)
- [Inference runtime](../../architecture/INFERENCE_RUNTIME.md)
- [Troubleshooting FAQ](../troubleshooting/faq.md)
- [Testing guide](../../development/testing.md)
