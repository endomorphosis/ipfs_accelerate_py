"""
Test file for webgpu platform.

This file contains tests for the webgpu platform,
including device detection, computation, and webgpu-specific capabilities.
Generated from HardwareTestTemplate.
"""

import os
import pytest
import logging
import time
from typing import Dict, List, Any, Optional

# Import common utilities
from common.hardware_detection import detect_hardware, setup_platform

# WebGPU-specific imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
except ImportError:
    pass

from common.fixtures import webgpu_browser

# Hardware-specific fixtures
@pytest.fixture
def webgpu_test_page(temp_dir):
    """Create a test HTML page for webgpu tests."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebGPU Test</title>
        <script>
            async function runTest() {
                const resultElement = document.getElementById('result');
                try {
                    // Check for webgpu support
                    if ('webgpu' === 'webgpu') {
                        if (!navigator.gpu) {
                            resultElement.textContent = 'WebGPU not supported';
                            return;
                        }
                        const adapter = await navigator.gpu.requestAdapter();
                        if (!adapter) {
                            resultElement.textContent = 'Couldn\\'t request WebGPU adapter';
                            return;
                        }
                        const device = await adapter.requestDevice();
                        resultElement.textContent = 'WebGPU device created successfully';
                    } else if ('webgpu' === 'webnn') {
                        if (!('ml' in navigator)) {
                            resultElement.textContent = 'WebNN not supported';
                            return;
                        }
                        const context = await navigator.ml.createContext();
                        if (!context) {
                            resultElement.textContent = 'Couldn\\'t create WebNN context';
                            return;
                        }
                        resultElement.textContent = 'WebNN context created successfully';
                    }
                } catch (error) {
                    resultElement.textContent = `Error: ${error.message}`;
                }
            }
            
            window.onload = runTest;
        </script>
    </head>
    <body>
        <h1>WebGPU Test</h1>
        <div id="result">Testing...</div>
    </body>
    </html>
    """
    
    file_path = os.path.join(temp_dir, 'test_page.html')
    with open(file_path, 'w') as f:
        f.write(html_content)
    
    return file_path

@pytest.fixture
def webgpu_matmul_page(temp_dir):
    """Create a test HTML page for WebGPU matrix multiplication."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebGPU Matrix Multiplication Test</title>
        <script>
            async function runMatrixMultiplication() {
                const resultElement = document.getElementById('result');
                const benchmarkElement = document.getElementById('benchmark');
                
                try {
                    // Check for WebGPU support
                    if (!navigator.gpu) {
                        resultElement.textContent = 'WebGPU not supported';
                        return;
                    }
                    
                    // Request adapter and device
                    const adapter = await navigator.gpu.requestAdapter();
                    if (!adapter) {
                        resultElement.textContent = 'Couldn\\'t request WebGPU adapter';
                        return;
                    }
                    const device = await adapter.requestDevice();
                    resultElement.textContent = 'WebGPU device created successfully';
                    
                    // Matrix dimensions
                    const matrixSize = 1024;
                    
                    // Create matrices with random data
                    const matrixA = new Float32Array(matrixSize * matrixSize);
                    const matrixB = new Float32Array(matrixSize * matrixSize);
                    const resultMatrix = new Float32Array(matrixSize * matrixSize);
                    
                    // Fill matrices with random values
                    for (let i = 0; i < matrixA.length; i++) {
                        matrixA[i] = Math.random();
                        matrixB[i] = Math.random();
                    }
                    
                    // Create buffers
                    const matrixABuffer = device.createBuffer({
                        size: matrixA.byteLength,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    });
                    
                    const matrixBBuffer = device.createBuffer({
                        size: matrixB.byteLength,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    });
                    
                    const resultBuffer = device.createBuffer({
                        size: resultMatrix.byteLength,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                    });
                    
                    // Write data to buffers
                    device.queue.writeBuffer(matrixABuffer, 0, matrixA);
                    device.queue.writeBuffer(matrixBBuffer, 0, matrixB);
                    
                    // Create compute pipeline
                    const computeShaderModule = device.createShaderModule({
                        code: `
                            @group(0) @binding(0) var<storage, read> matrixA : array<f32>;
                            @group(0) @binding(1) var<storage, read> matrixB : array<f32>;
                            @group(0) @binding(2) var<storage, read_write> resultMatrix : array<f32>;
                            
                            @compute @workgroup_size(8, 8, 1)
                            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                                let dimension = ${matrixSize}u;
                                let row = global_id.x;
                                let col = global_id.y;
                                
                                if (row >= dimension || col >= dimension) {
                                    return;
                                }
                                
                                var sum = 0.0;
                                for (var i = 0u; i < dimension; i = i + 1u) {
                                    sum = sum + matrixA[row * dimension + i] * matrixB[i * dimension + col];
                                }
                                
                                resultMatrix[row * dimension + col] = sum;
                            }
                        `
                    });
                    
                    const computePipeline = device.createComputePipeline({
                        layout: 'auto',
                        compute: {
                            module: computeShaderModule,
                            entryPoint: 'main',
                        },
                    });
                    
                    // Create bind group
                    const bindGroup = device.createBindGroup({
                        layout: computePipeline.getBindGroupLayout(0),
                        entries: [
                            {
                                binding: 0,
                                resource: { buffer: matrixABuffer },
                            },
                            {
                                binding: 1,
                                resource: { buffer: matrixBBuffer },
                            },
                            {
                                binding: 2,
                                resource: { buffer: resultBuffer },
                            },
                        ],
                    });
                    
                    // Warm-up runs
                    for (let i = 0; i < 3; i++) {
                        const commandEncoder = device.createCommandEncoder();
                        const computePass = commandEncoder.beginComputePass();
                        computePass.setPipeline(computePipeline);
                        computePass.setBindGroup(0, bindGroup);
                        computePass.dispatchWorkgroups(Math.ceil(matrixSize / 8), Math.ceil(matrixSize / 8));
                        computePass.end();
                        device.queue.submit([commandEncoder.finish()]);
                        await device.queue.onSubmittedWorkDone();
                    }
                    
                    // Benchmark
                    const iterations = 5;
                    const startTime = performance.now();
                    
                    for (let i = 0; i < iterations; i++) {
                        const commandEncoder = device.createCommandEncoder();
                        const computePass = commandEncoder.beginComputePass();
                        computePass.setPipeline(computePipeline);
                        computePass.setBindGroup(0, bindGroup);
                        computePass.dispatchWorkgroups(Math.ceil(matrixSize / 8), Math.ceil(matrixSize / 8));
                        computePass.end();
                        device.queue.submit([commandEncoder.finish()]);
                        await device.queue.onSubmittedWorkDone();
                    }
                    
                    const endTime = performance.now();
                    const duration = (endTime - startTime) / iterations;
                    
                    benchmarkElement.textContent = `Matrix multiplication (${matrixSize}x${matrixSize}) took ${duration.toFixed(2)} ms`;
                    
                    // Verify a sample of the computation
                    const readBuffer = device.createBuffer({
                        size: resultMatrix.byteLength,
                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                    });
                    
                    const commandEncoder = device.createCommandEncoder();
                    commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultMatrix.byteLength);
                    device.queue.submit([commandEncoder.finish()]);
                    
                    await readBuffer.mapAsync(GPUMapMode.READ);
                    const result = new Float32Array(readBuffer.getMappedRange());
                    
                    // Compute a checksum for verification
                    let checksum = 0;
                    for (let i = 0; i < 10; i++) {
                        checksum += result[i];
                    }
                    
                    document.getElementById('checksum').textContent = `Result checksum: ${checksum.toFixed(6)}`;
                    readBuffer.unmap();
                    
                } catch (error) {
                    resultElement.textContent = `Error: ${error.message}`;
                }
            }
            
            window.onload = runMatrixMultiplication;
        </script>
    </head>
    <body>
        <h1>WebGPU Matrix Multiplication Test</h1>
        <div id="result">Testing...</div>
        <div id="benchmark">Benchmarking...</div>
        <div id="checksum">Checksum: N/A</div>
    </body>
    </html>
    """
    
    file_path = os.path.join(temp_dir, 'webgpu_matmul_test.html')
    with open(file_path, 'w') as f:
        f.write(html_content)
    
    return file_path

class TestWebgpuMatmul:
    """
    Tests for webgpu platform.
    """
    
    @pytest.mark.webgpu
    def test_webgpu_available(self):
        """Test WebGPU availability."""
        hardware_info = detect_hardware()
        assert hardware_info['platforms']['webgpu']['available']
    
    @pytest.mark.webgpu
    def test_webgpu_browser_launch(self, webgpu_browser):
        """Test WebGPU browser launch."""
        assert webgpu_browser is not None
    
    @pytest.mark.webgpu
    def test_webgpu_device_creation(self, webgpu_browser, webgpu_test_page):
        """Test WebGPU device creation."""
        webgpu_browser.get(f"file://{webgpu_test_page}")
        time.sleep(2)  # Allow time for JavaScript to execute
        result_element = webgpu_browser.find_element(By.ID, 'result')
        assert result_element.text == 'WebGPU device created successfully'
    
    @pytest.mark.webgpu
    def test_webgpu_matmul_computation(self, webgpu_browser, webgpu_matmul_page):
        """Test WebGPU matrix multiplication computation."""
        webgpu_browser.get(f"file://{webgpu_matmul_page}")
        time.sleep(10)  # Allow time for the computation to complete
        
        # Check if the computation was successful
        result_element = webgpu_browser.find_element(By.ID, 'result')
        assert result_element.text == 'WebGPU device created successfully', f"WebGPU device creation failed: {result_element.text}"
        
        # Check if benchmark ran
        benchmark_element = webgpu_browser.find_element(By.ID, 'benchmark')
        assert "Matrix multiplication" in benchmark_element.text, f"Benchmark did not run: {benchmark_element.text}"
        
        # Check if we have a checksum (indicating computation completed)
        checksum_element = webgpu_browser.find_element(By.ID, 'checksum')
        assert "Result checksum:" in checksum_element.text, f"Computation did not complete: {checksum_element.text}"
        
        # Log the benchmark result
        logging.info(benchmark_element.text)
        
        # Extract and log the performance time
        import re
        match = re.search(r'took (\d+\.\d+) ms', benchmark_element.text)
        if match:
            duration_ms = float(match.group(1))
            logging.info(f"WebGPU MatMul duration: {duration_ms} ms")
            
            # Performance assertion (adjust threshold as needed)
            assert duration_ms < 10000, f"WebGPU MatMul performance too slow: {duration_ms} ms"