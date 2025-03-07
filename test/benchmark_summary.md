# Benchmark Results Summary

Generated: 2025-03-06 19:36:37

## Hardware Platforms

| ID | Type | Description | Simulated? | Simulation Reason |
|---|---|---|---|---|
| 1 | cpu | CPU | ❌ No | N/A |
| 2 | webgpu | WebGPU Device | ❌ No | N/A |
| 3 | webnn | WebNN Device | ❌ No | N/A |
| 4 | rocm | AMD GPU | ✅ Yes | Hardware not available |

## Recent Performance Results

| Model | Family | Hardware | Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) | Simulated? | Timestamp |
|---|---|---|---|---|---|---|---|---|
| prajjwal1/bert-tiny | embedding | cpu | 1 | 1.69 | 592.65 | 150.00 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | cpu | 2 | 3.05 | 655.74 | 160.00 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | cpu | 4 | 5.82 | 687.28 | 175.00 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | cpu | 8 | 11.23 | 712.38 | 200.00 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | cpu | 16 | 22.10 | 724.16 | 250.00 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | webgpu | 1 | 1.92 | 519.77 | 1.06 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | webgpu | 2 | 2.25 | 888.89 | 1.25 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | webgpu | 4 | 3.10 | 1290.32 | 1.75 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | webgpu | 8 | 4.55 | 1758.24 | 2.50 | ❌ No | 2025-03-06 19:02:52 |
| prajjwal1/bert-tiny | embedding | webgpu | 16 | 7.85 | 2038.22 | 4.20 | ❌ No | 2025-03-06 19:02:52 |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |
| unknown_model | None | cpu | 1 | 0.00 | 0.00 | 0.00 | ❌ No | N/A |

## Summary Statistics

### Results by Hardware Type

| Hardware Type | Result Count |
|---|---|
| cpu | 148 |
| webgpu | 10 |
| rocm | 2 |

### Results by Model Family

| Model Family | Result Count |
|---|---|
| embedding | 20 |
| bert | 4 |

### Simulation Status

| Simulation Status | Result Count |
|---|---|
| Real | 158 |
| Simulated | 2 |
