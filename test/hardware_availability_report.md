# Hardware Availability Report
Generated: 2025-03-06 17:28:41

## Available Hardware

- ✅ **CPU**: Direct CPU detection
- ✅ **CUDA** (Quadro P4000): Direct CUDA device detection via torch
- ✅ **OPENVINO** (CPU): OpenVINO runtime detected with available devices

## Unavailable Hardware

- ❌ **ROCM**: torch.hip is not available
- ❌ **MPS**: Not running on macOS
- ❌ **QNN**: QNN package not installed and Qualcomm AI Stack not found
- ❌ **WEBNN**: WebNN requires a browser environment and is not natively available
- ❌ **WEBGPU**: WebGPU requires a browser environment and is not natively available

## Detailed Information

### CPU (✅ Available)

Detection method: Direct CPU detection

Details:
```json
{
  "platform": "Linux-6.8.0-11-generic-x86_64-with-glibc2.39",
  "processor": "x86_64",
  "cpu_count": 8,
  "architecture": "x86_64",
  "python_version": "3.12.3",
  "model_name": "Intel(R) Xeon(R) CPU E3-1535M v6 @ 3.10GHz"
}
```

### CUDA (✅ Available)

Detection method: Direct CUDA device detection via torch

Details:
```json
{
  "device_count": 1,
  "devices": [
    {
      "name": "Quadro P4000",
      "total_memory_gb": 7.92,
      "compute_capability": "6.1",
      "multi_processor_count": 14
    }
  ],
  "cuda_version": "12.4"
}
```

### MPS (❌ Not Available)

Detection method: Not running on macOS

No additional details available.

### OPENVINO (✅ Available)

Detection method: OpenVINO runtime detected with available devices

Details:
```json
{
  "version": "2025.0.0-17942-1f68be9f594-releases/2025/0",
  "available_devices": [
    "CPU"
  ],
  "device_properties": {
    "CPU": "Intel(R) Xeon(R) CPU E3-1535M v6 @ 3.10GHz"
  }
}
```

### QNN (❌ Not Available)

Detection method: QNN package not installed and Qualcomm AI Stack not found

No additional details available.

### ROCM (❌ Not Available)

Detection method: torch.hip is not available

No additional details available.

### WEBGPU (❌ Not Available)

Detection method: WebGPU requires a browser environment and is not natively available

No additional details available.

### WEBNN (❌ Not Available)

Detection method: WebNN requires a browser environment and is not natively available

No additional details available.
