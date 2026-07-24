# Hardware and Runtime Capability Guide

Hardware support is discovered at runtime. The package can import on a host
without CUDA, ROCm, OpenVINO, MPS, WebNN, WebGPU, or Qualcomm support, and a
driver/package probe is not a substitute for a model smoke test.

## Discover capabilities

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print("available accelerators:", report.get("hardware", {}).get("available", []))
print("hardware details:", report.get("hardware", {}))
PY
```

The report is JSON-friendly and may be partial when optional detectors are not
installed. It also reports registered models/endpoints and MCP capabilities.

## CUDA

Check both the driver-visible device and the PyTorch build:

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

Install a compatible PyTorch CUDA wheel using the [installation guide](../getting-started/installation.md).
For a model operation, choose `device="cuda"` only after the check passes:

```python
from ipfs_accelerate_py import ipfs_accelerate_py

accelerator = ipfs_accelerate_py({}, {})
result = accelerator.run_model(
    "bert-base-uncased",
    {"input_ids": [[101, 2023, 2003, 102]]},
    device="cuda",
)
```

The model provider and model task must also support the selected device.

## CPU and ARM

CPU is the baseline and is useful for deterministic smoke tests. Threading is
normally controlled by the numerical libraries and the workload. Set their
environment variables before importing the model stack when you need a bounded
process:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

Do not assume that an ARM machine provides the same kernels or model/provider
coverage as x86. Record architecture, Python version, package versions, model,
and batch shape in performance reports.

## ROCm, OpenVINO, and MPS

These backends require their upstream runtime and a compatible model/provider.
Install the runtime according to its vendor documentation, then inspect the
capability report. MPS is available only on supported Apple hardware and a
compatible PyTorch build. ROCm uses the PyTorch ROCm distribution rather than
the CUDA wheel.

## WebNN and WebGPU

Browser acceleration is a separate runtime from Python device selection. Install
the `webnn` extra and browser tooling, then follow
[WebNN/WebGPU integration](../../features/webnn-webgpu/WEBNN_WEBGPU_README.md).
Browser support varies by browser, driver, and enabled flags.

## Throughput and memory

Measure the workload you intend to deploy. Useful variables include model,
sequence/image/audio size, batch size, precision, device, warm-up count,
concurrency, and cache state. Avoid copying benchmark numbers from historical
reports without their commit and hardware context.

For the agent supervisor, parallelism is admission-controlled by CPU, memory,
disk, provider capacity, task conflicts, dependencies, and leases. Increasing a
lane count does not necessarily increase throughput; inspect scheduler metrics
and provider capacity first. See the
[Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md).
