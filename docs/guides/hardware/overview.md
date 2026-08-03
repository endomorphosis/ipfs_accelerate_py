# Hardware and Runtime Capability Guide

**Status:** Current
**Owner:** package maintainers
**Audience:** Developers and operators selecting devices and diagnosing
accelerator availability
**Scope:** Runtime capability discovery, CPU baseline, optional accelerators
(CUDA, ROCm, OpenVINO, MPS, Qualcomm, WebNN/WebGPU), measurement guidance, and
what does **not** count as hardware proof
**Non-goals:** Vendor driver installation manuals; guaranteeing any GPU or
browser backend; packaging extras named `cuda`/`rocm`/`openvino` (they do not
exist); full deployment process management (see
[deployment README](../deployment/README.md))
**Last-verified:** 2026-08-03 @ `d5f3aa5c6`; core constructor behavior, direct
hardware detector, full capability report, and install helper paths checked
against this tree
**Sources:** `ipfs_accelerate_py/__init__.py` (`IPFS_ACCEL_SKIP_CORE`,
`get_instance`); `ipfs_accelerate_py/ipfs_accelerate.py`
(`get_capabilities`, `hwtest`),
`ipfs_accelerate_py/hf_model_server/hardware/detector.py` (`HardwareDetector`),
`ipfs_accelerate_py/kit/hardware_kit.py`, `install/requirements_torch_*.txt`,
`install/requirements_cuda.txt`, `install/requirements_openvino.txt`,
`install/requirements_apple.txt`, `install/requirements_qualcomm.txt`,
`pyproject.toml` (extras; no dedicated CUDA packaging extra),
`test/hardware/`, `test/hardware_detection/`
**Freshness triggers:** core constructor, hardware detector, capability-report,
backend extra, vendor helper, or hardware-test changes

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

For a bounded hardware-only probe, skip the heavy core and call the direct
detector:

```bash
IPFS_ACCEL_SKIP_CORE=1 python - <<'PY'
from ipfs_accelerate_py.hf_model_server.hardware.detector import HardwareDetector

detector = HardwareDetector()
print("available accelerators:", detector.get_available_hardware())
for name, capability in detector.capabilities.items():
    print(name, capability)
PY
```

| Field | Meaning | Caveat |
| --- | --- | --- |
| `HardwareDetector.capabilities` | Best-effort CPU/platform and optional accelerator map | May be partial if detectors or frameworks are missing |
| `HardwareDetector.get_available_hardware()` | Names currently flagged available | Not a throughput SLA or model proof |

The detector output is a **discovery snapshot**, not a lease, receipt, or
health certificate.

The broader `get_instance().get_capabilities(detail=True)` report also includes
`hwtest`, registered models/endpoints, task types, and MCP counts. It is **not**
a side-effect-free probe: `get_instance()` constructs the core runtime, which
initializes the storage wrapper and API adapters and can touch files, contact
configured storage/IPFS endpoints, or attempt optional daemon initialization.
Run it only where those effects are permitted. No side-effect-free top-level
equivalent for the complete report is proven at this revision.

---

## 2. CPU baseline (always plan for this path)

CPU is the supported default for install verification, CI-friendly smoke tests,
and hosts without accelerators.

```bash
# Optional: bound numerical library threads before importing the model stack
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

IPFS_ACCEL_SKIP_CORE=1 python - <<'PY'
from ipfs_accelerate_py.hf_model_server.hardware.detector import HardwareDetector

detector = HardwareDetector()
assert detector.is_available("cpu")
print("CPU detected; run a separate already-local model smoke for workload proof")
PY
```

Do not instantiate the full core merely to prove this baseline: construction
has the side effects described above. For inference evidence, separately run a
small model that is already present locally, with the provider and device
explicitly pinned to CPU. Model availability is environment-specific, so this
guide does not invent a universally cached model command.

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

Install the vendor runtime first, then re-run the direct `HardwareDetector`
probe. Run the full `get_capabilities(detail=True)` snapshot only when its
constructor effects are acceptable. Absence of a backend is a **capability
report**, not a package defect.

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

**Authoritative operator sequence:** direct hardware report → framework device
check → small model smoke on the intended device → only then plan capacity.

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
