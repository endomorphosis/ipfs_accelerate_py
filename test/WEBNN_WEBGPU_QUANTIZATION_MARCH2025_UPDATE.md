# WebNN and WebGPU Quantization - March 2025 Update

## Summary of Enhancements

The March 2025 update introduces significant improvements to our WebNN and WebGPU quantization implementation:

1. **Enhanced WebNN Precision Options**:
   - Added experimental 4-bit support for WebNN
   - Implemented detailed error reporting for low precision
   - Maintained backward compatibility with 8-bit fallback in standard mode

2. **Two Operating Modes for WebNN**:
   - **Standard Mode**: Automatically upgrades 4-bit/2-bit requests to 8-bit (silent fallback)
   - **Experimental Mode**: Attempts true 4-bit/2-bit and reports detailed errors

3. **Improved Testing Infrastructure**:
   - New comprehensive testing script for all precision levels
   - Real browser implementation using Selenium
   - Detailed metrics collection and reporting

4. **Documentation Updates**:
   - Clarified WebNN and WebGPU capabilities
   - Added examples for all precision levels
   - Documented browser-specific considerations

## WebNN Precision Modes Comparison

| Feature | Standard Mode | Experimental Mode |
|---------|--------------|-------------------|
| 8-bit Support | ✅ Native | ✅ Native |
| 4-bit Support | ⚠️ Upgrades to 8-bit | ⚠️ Attempts true 4-bit |
| 2-bit Support | ⚠️ Upgrades to 8-bit | ⚠️ Attempts true 2-bit |
| Error Handling | Silent fallback | Reports actual errors |
| API Flag | N/A (default) | `--experimental-precision` |
| Env Variable | N/A | `WEBNN_EXPERIMENTAL_PRECISION=1` |
| Use Case | Production | Research/debugging |

## Quick Start Guide

The IPFS Accelerate framework now provides the `run_real_webgpu_webnn.py` script as the main entrypoint for running WebNN and WebGPU tests with different precision levels.

### Running WebNN Tests:

```bash
# Standard Mode (8-bit with fallback)
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 8
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 4  # Auto-upgrades to 8-bit

# Experimental Mode (with error reporting)
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 4 --experimental-precision
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 2 --experimental-precision

# Run with specific model
python run_real_webgpu_webnn.py --platform webnn --browser edge --model bert-base-uncased --bits 8
```

### Running WebGPU Tests:

```bash
# Standard precision levels
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 16  # FP16 baseline
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 8   # INT8 (50% reduction)
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 4   # INT4 (75% reduction)
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 2   # INT2 (87.5% reduction)

# Mixed precision (higher bits for critical layers)
python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits 4 --mixed-precision

# Firefox with optimized audio compute shaders
python run_real_webgpu_webnn.py --platform webgpu --browser firefox --model whisper-tiny --bits 4
```

### Testing Both Platforms:

```bash
# Test both WebNN and WebGPU with default settings
python run_real_webgpu_webnn.py --platform both --browser chrome

# Test both platforms with specific bit precision
python run_real_webgpu_webnn.py --platform both --browser edge --bits 8
```

## API Usage

The WebNN and WebGPU implementations can be accessed through our unified Python API or directly through the command-line script.

### Importing the Real Implementation:

```python
# Import the real implementation module
import asyncio
from run_real_webgpu_webnn import run_real_webnn_test, run_real_webgpu_test

# Run WebNN test with experimental precision
async def test_webnn():
    result = await run_real_webnn_test(
        model="bert-base-uncased",
        browser="edge",
        bits=4,
        experimental_precision=True,
        mixed_precision=False
    )
    print(f"WebNN test result: {result}")

# Run WebGPU test with different precision levels
async def test_webgpu_precision():
    for bits in [16, 8, 4, 2]:
        result = await run_real_webgpu_test(
            model="bert-base-uncased",
            browser="chrome",
            bits=bits,
            mixed_precision=False
        )
        print(f"{bits}-bit WebGPU result: {result}")

# Run the tests
asyncio.run(test_webnn())
asyncio.run(test_webgpu_precision())
```

### Using Command-Line Tool (Recommended):

The simplest approach is to use the command-line tool:

```bash
# Run with default settings (will use WebGPU on Chrome)
python run_real_webgpu_webnn.py

# Run WebNN with experimental precision
python run_real_webgpu_webnn.py --platform webnn --browser edge --bits 4 --experimental-precision

# Run WebGPU with different precision levels
for BITS in 16 8 4 2; do
  python run_real_webgpu_webnn.py --platform webgpu --browser chrome --bits $BITS
done
```

### Advanced Usage with Custom Options:

```python
import asyncio
import subprocess
import json

# Run the command-line tool with custom options
async def run_custom_test(platform, browser, bits, model="bert-base-uncased", experimental=False):
    cmd = [
        "python", "run_real_webgpu_webnn.py",
        f"--platform={platform}",
        f"--browser={browser}",
        f"--model={model}"
    ]
    
    if bits:
        cmd.append(f"--bits={bits}")
    
    if experimental:
        cmd.append("--experimental-precision")
    
    # Run the command and capture output
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    # Parse the results
    try:
        # Find JSON output in the stdout
        output = stdout.decode()
        start = output.find("{")
        end = output.rfind("}") + 1
        if start >= 0 and end > start:
            result_json = output[start:end]
            return json.loads(result_json)
        else:
            return {"status": "error", "message": "No JSON result found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Example usage
asyncio.run(run_custom_test("webnn", "edge", 4, experimental=True))
```

## Browser-Specific Considerations

1. **Chrome**:
   - Good support for WebGPU at all precision levels
   - WebNN supported from Chrome 122+
   - Best performance with 4-bit WebGPU

2. **Edge**:
   - Best WebNN support across browsers
   - Good WebGPU support at all precision levels
   - Recommended for WebNN testing

3. **Firefox**:
   - No WebNN support
   - Excellent WebGPU support, especially for audio models
   - ~20% better performance for audio models with compute shaders
   - Recommended for 4-bit WebGPU audio processing

4. **Safari**:
   - Limited WebGPU support
   - Some WebNN support
   - Best with 8-bit precision
   - Not recommended for 2-bit testing

## Troubleshooting

### Common WebNN Issues in Experimental Mode

When using WebNN with experimental low precision:

1. **Unsupported Operation**: Browser may report operation not supported with requested precision
   ```
   Error: Failed to execute 'buildSync' on 'MLGraphBuilder': The operator 'matmul' is not supported with the requested precision
   ```

2. **Out of Resources**: Browser may report insufficient resources for operation
   ```
   Error: Failed to execute operation: Out of resources
   ```

3. **Implementation Limitations**: Browser may report specific implementation limitations
   ```
   Error: INT4 precision not supported on this hardware
   ```

### WebGPU Issues

When using WebGPU with ultra-low precision:

1. **Shader Compilation Errors**: May occur with 2-bit precision
   ```
   Error: Shader compilation failed: INT2 packing not supported
   ```

2. **Memory Errors**: Common with large models and low precision
   ```
   Error: Out of memory when allocating buffer
   ```

## Technical Implementation Details

1. **WebNN Experimental Mode**:
   - Implemented in `fixed_web_platform/webnn_implementation.py`
   - Uses environment variable `WEBNN_EXPERIMENTAL_PRECISION=1`
   - Removes automatic precision upgrade logic
   - Preserves original error messages from browser

2. **WebGPU Low Precision Implementation**:
   - Uses custom compute shaders for each precision level
   - Specialized kernels for matrix multiplication
   - Memory-efficient packing for 2-bit and 4-bit values
   - Firefox-optimized compute shaders for audio models

3. **Mixed Precision Implementation**:
   - Identifies critical model components (attention, embedding layers)
   - Uses higher precision (8-bit) for critical components
   - Uses lower precision (2-bit/4-bit) for less sensitive layers
   - Dynamic adjustment based on hardware capabilities

## Further Development

Future enhancements planned for WebNN and WebGPU quantization:

1. **WebNN-specific optimizations**: Custom kernels for known browser implementations
2. **Safari-specific WebGPU support**: Better ultra-low precision on Safari
3. **Firefox audio optimizations**: Enhanced compute shaders for audio models
4. **Adaptive precision selection**: Dynamic precision based on hardware compatibility
5. **Cross-browser compatibility layer**: Unified API across all browsers