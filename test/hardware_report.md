# Hardware Report

Generated on 2025-03-06 09:51:11

## Hardware Platforms

|    |   hardware_id | hardware_type   | device_name   | driver_version   | compute_units   |   memory_gb |
|---:|--------------:|:----------------|:--------------|:-----------------|:----------------|------------:|
|  0 |             1 | cpu             | CPU           |                  | <NA>            |         nan |
|  1 |             2 | cuda            | NVIDIA GPU    |                  | <NA>            |         nan |
|  2 |             4 | mps             | Apple Silicon |                  | <NA>            |         nan |
|  3 |             5 | openvino        | Intel CPU/GPU |                  | <NA>            |         nan |
|  4 |             8 | qualcomm        | Qualcomm QNN  |                  | <NA>            |         nan |
|  5 |             3 | rocm            | AMD GPU       |                  | <NA>            |         nan |
|  6 |             7 | webgpu          | WebGPU        |                  | <NA>            |         nan |
|  7 |             6 | webnn           | WebNN         |                  | <NA>            |         nan |

## Memory Analysis

|    | model_name        | hardware_type   |   avg_memory |   min_memory |   max_memory |
|---:|:------------------|:----------------|-------------:|-------------:|-------------:|
|  0 | bert-base-uncased | cuda            |      3943.44 |      3712.58 |      4172.81 |
|  1 | bert-base-uncased | cpu             |      2874.4  |      2874.4  |      2874.4  |
