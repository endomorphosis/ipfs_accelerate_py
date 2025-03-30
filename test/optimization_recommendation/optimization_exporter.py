#!/usr/bin/env python3
"""
Optimization Recommendation Exporter

This module exports hardware optimization recommendations to deployable configuration files.
It generates ready-to-use configuration files and implementation scripts based on optimization
recommendations for various frameworks and hardware platforms.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import zipfile
import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, BinaryIO

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optimization_exporter")

# Import optimization client
try:
    from test.optimization_recommendation.optimization_client import OptimizationClient
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Optimization client not available")
    OPTIMIZATION_AVAILABLE = False

class OptimizationExporter:
    """Exporter for hardware optimization recommendations."""
    
    def __init__(
        self,
        output_dir: str = "./optimization_configs",
        benchmark_db_path: str = "benchmark_db.duckdb",
        api_url: str = "http://localhost:8080",
        api_key: Optional[str] = None
    ):
        """
        Initialize the optimization exporter.
        
        Args:
            output_dir: Directory to store generated configuration files
            benchmark_db_path: Path to benchmark database
            api_url: API base URL
            api_key: Optional API key
        """
        self.output_dir = Path(output_dir)
        self.benchmark_db_path = benchmark_db_path
        self.api_url = api_url
        self.api_key = api_key
        
        # Create client
        self.client = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.client = OptimizationClient(
                    benchmark_db_path=benchmark_db_path,
                    predictive_api_url=api_url,
                    api_key=api_key
                )
                logger.info("Optimization client initialized")
            except Exception as e:
                logger.error(f"Error initializing optimization client: {e}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Template paths
        self.template_dir = Path(__file__).parent / "templates"
        
        # Load implementation templates
        self.implementation_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load implementation templates for different platforms and frameworks.
        
        Returns:
            Dictionary of implementation templates
        """
        templates = {
            "pytorch": {},
            "tensorflow": {},
            "onnx": {},
            "webgpu": {},
            "webnn": {},
            "openvino": {}
        }
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.template_dir, exist_ok=True)
        
        # PyTorch templates
        templates["pytorch"]["mixed_precision"] = """
# PyTorch Mixed Precision Implementation
import torch

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

# Forward pass with mixed precision
def forward_with_mixed_precision(model, inputs):
    with autocast():
        outputs = model(**inputs)
    return outputs

# Training with mixed precision
def train_step_with_mixed_precision(model, optimizer, inputs, labels):
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(**inputs)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()
"""
        
        templates["pytorch"]["quantization"] = """
# PyTorch Quantization Implementation
import torch
import torch.quantization

# Prepare model for quantization
def prepare_model_for_quantization(model, qconfig='fbgemm'):
    # Set qconfig
    if qconfig == 'fbgemm':
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    elif qconfig == 'qnnpack':
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Prepare model
    model_prepared = torch.quantization.prepare(model)
    return model_prepared

# Quantize model after calibration
def quantize_model(model_prepared):
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized

# Example calibration function
def calibrate_model(model_prepared, calibration_data_loader):
    with torch.no_grad():
        for inputs, _ in calibration_data_loader:
            model_prepared(**inputs)
    
    return model_prepared
"""
        
        templates["pytorch"]["tensorrt"] = """
# PyTorch TensorRT Integration
import torch
import torch_tensorrt

# Convert model to TensorRT
def convert_to_tensorrt(model, input_shapes, precision='fp16', workspace_size=8, dynamic_shapes=False):
    # Define input shapes
    inputs = [
        torch_tensorrt.Input(
            shape=shape,
            dtype=torch.float32,
            name=f"input_{i}"
        )
        for i, shape in enumerate(input_shapes)
    ]
    
    # Configure conversion
    enabled_precisions = {torch.float, torch.half if precision == 'fp16' else None}
    enabled_precisions = {p for p in enabled_precisions if p is not None}
    
    # Convert to TensorRT
    trt_model = torch_tensorrt.compile(
        model,
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        workspace_size=workspace_size * (1 << 30)  # Convert GB to bytes
    )
    
    return trt_model

# Save TensorRT model
def save_tensorrt_model(model, path):
    torch.save(model.state_dict(), path)
    return path
"""
        
        # TensorFlow templates
        templates["tensorflow"]["mixed_precision"] = """
# TensorFlow Mixed Precision Implementation
import tensorflow as tf

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Example model creation with mixed precision
def create_mixed_precision_model(model_fn, *args, **kwargs):
    # Create model with mixed precision
    with tf.keras.mixed_precision.policy('mixed_float16'):
        model = model_fn(*args, **kwargs)
    
    # Configure optimizer with loss scaling
    optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    return model, optimizer
"""
        
        templates["tensorflow"]["quantization"] = """
# TensorFlow Quantization Implementation
import tensorflow as tf

# Post-training quantization
def quantize_model(model, representative_dataset, quantization_mode='int8'):
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set quantization parameters
    if quantization_mode == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif quantization_mode == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    # Convert model
    quantized_model = converter.convert()
    return quantized_model

# Save quantized model
def save_quantized_model(quantized_model, path):
    with open(path, 'wb') as f:
        f.write(quantized_model)
    return path
"""
        
        templates["tensorflow"]["tensorrt"] = """
# TensorFlow TensorRT Integration
import tensorflow as tf
import tensorrt as trt
from tensorflow.python.compiler.tensorrt import trt_convert as trt_convert

# Convert model to TensorRT
def convert_to_tensorrt(model, precision='FP16', workspace_size=8, max_batch_size=1):
    # Get SavedModel directory
    saved_model_dir = getattr(model, 'saved_model_dir', None)
    if saved_model_dir is None:
        raise ValueError("Model must be a SavedModel or have saved_model_dir attribute")
    
    # Set precision
    precision_mode = trt_convert.TrtPrecisionMode.FP16 if precision == 'FP16' else trt_convert.TrtPrecisionMode.FP32
    
    # Create converter
    converter = trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        precision_mode=precision_mode,
        max_workspace_size_bytes=workspace_size * (1 << 30),  # Convert GB to bytes
        max_batch_size=max_batch_size
    )
    
    # Convert model
    converter.convert()
    
    return converter

# Save TensorRT model
def save_tensorrt_model(converter, output_dir):
    converter.save(output_dir)
    return output_dir
"""
        
        # WebGPU templates
        templates["webgpu"]["shader_optimization"] = """
// WebGPU Shader Optimization

// Create optimized pipeline with precompiled shaders
async function createOptimizedPipeline(device, shaderModule, pipelineLayout, vertexState, fragmentState) {
  // Cache key for the pipeline
  const cacheKey = `pipeline:${shaderModule.label}:${pipelineLayout.label}`;
  
  // Check cache
  const cachedPipeline = await loadPipelineFromCache(cacheKey);
  if (cachedPipeline) {
    return cachedPipeline;
  }
  
  // Create the pipeline
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: vertexState,
    fragment: fragmentState,
  });
  
  // Cache the pipeline
  await cachePipeline(cacheKey, pipeline);
  
  return pipeline;
}

// Utility to load pipeline from cache
async function loadPipelineFromCache(cacheKey) {
  try {
    const cache = await caches.open('webgpu-pipelines');
    const response = await cache.match(cacheKey);
    if (response) {
      const data = await response.json();
      // Recreate pipeline from cache data
      // This is a placeholder - actual implementation would depend on the caching strategy
      return recreatePipelineFromCache(data);
    }
  } catch (error) {
    console.error('Error loading pipeline from cache:', error);
  }
  return null;
}

// Utility to cache pipeline
async function cachePipeline(cacheKey, pipeline) {
  try {
    const cache = await caches.open('webgpu-pipelines');
    // Serialize pipeline for caching
    const data = serializePipeline(pipeline);
    await cache.put(cacheKey, new Response(JSON.stringify(data)));
  } catch (error) {
    console.error('Error caching pipeline:', error);
  }
}

// Placeholder functions that would need real implementations
function serializePipeline(pipeline) {
  // Extract serializable data from pipeline
  return {
    label: pipeline.label,
    // Additional pipeline data for recreation
  };
}

function recreatePipelineFromCache(data) {
  // Recreate pipeline from cached data
  // This would need actual implementation based on your engine
  return null;
}

// Optimized compute shader dispatch with workgroup size optimization
function dispatchOptimizedCompute(device, pipeline, bindGroup, workgroupCountX, workgroupCountY, workgroupCountZ) {
  // Determine optimal workgroup size based on device limits
  const maxWorkgroupSize = Math.min(256, device.limits.maxComputeWorkgroupSizeX);
  const optimalWorkgroupSize = determineOptimalWorkgroupSize(maxWorkgroupSize);
  
  // Create pass and dispatch
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
  pass.end();
  
  return encoder.finish();
}

// Determine optimal workgroup size
function determineOptimalWorkgroupSize(maxSize) {
  // For most GPUs, workgroup sizes that are multiples of 32 or 64 tend to be efficient
  // This is a simple heuristic - benchmarking would provide better results
  if (maxSize >= 256) return 256;
  if (maxSize >= 128) return 128;
  if (maxSize >= 64) return 64;
  return Math.max(32, maxSize);
}
"""
        
        templates["webgpu"]["tensor_ops"] = """
// WebGPU Tensor Operations with FP16

// Define shader for mixed precision matrix multiplication
const matmulShader = `
@group(0) @binding(0) var<storage, read> inputA: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> inputB: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f16>>;

struct Dimensions {
  M: u32,
  N: u32,
  K: u32,
  strideA: u32,
  strideB: u32,
  strideC: u32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  if (row >= dims.M || col >= dims.N) {
    return;
  }
  
  var sum = vec4<f16>(0.0, 0.0, 0.0, 0.0);
  
  for (var k: u32 = 0; k < dims.K; k += 4) {
    let aVec = inputA[row * dims.strideA + k / 4];
    let bVec = inputB[col * dims.strideB + k / 4];
    
    // Dot product using f16
    sum = sum + vec4<f16>(
      aVec.x * bVec.x,
      aVec.y * bVec.y,
      aVec.z * bVec.z,
      aVec.w * bVec.w
    );
  }
  
  output[row * dims.strideC + col / 4] = sum;
}`;

// Function to create optimized tensor operations pipeline
async function createTensorOpsPipeline(device) {
  // Check if the device supports f16 storage
  const supportsF16 = device.features.has('shader-f16');
  
  if (!supportsF16) {
    console.warn("Device does not support f16 storage, falling back to f32");
    return createFallbackPipeline(device);
  }
  
  // Create shader module
  const shaderModule = device.createShaderModule({
    code: matmulShader,
  });
  
  // Create pipeline
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });
  
  return {
    pipeline,
    supportsF16,
  };
}

// Function to convert tensor to f16 format
function convertToF16Format(tensor, device) {
  // Get tensor data
  const data = tensor.data;
  
  // Create f16 buffer
  const f16Buffer = new Uint16Array(data.length);
  
  // Convert f32 to f16
  for (let i = 0; i < data.length; i++) {
    // This is a simplified conversion - a proper implementation would use a library
    f16Buffer[i] = convertF32ToF16(data[i]);
  }
  
  // Create WebGPU buffer
  const buffer = device.createBuffer({
    size: f16Buffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  
  // Copy data to buffer
  new Uint16Array(buffer.getMappedRange()).set(f16Buffer);
  buffer.unmap();
  
  return buffer;
}

// Placeholder for f32 to f16 conversion
function convertF32ToF16(value) {
  // This is a placeholder - actual conversion would require bit manipulation
  return value;
}

// Fallback to f32 pipeline when f16 is not supported
function createFallbackPipeline(device) {
  // Implementation would be similar but using f32 instead of f16
  console.log("Using fallback f32 pipeline");
  return null;
}
"""
        
        # WebNN templates
        templates["webnn"]["graph_optimization"] = """
// WebNN Graph Optimization

// Create optimized neural network graph
async function createOptimizedGraph(builder, optimizer = 'latency') {
  // Set graph optimization options
  const options = {
    powerPreference: 'default',
    optimizationHint: optimizer, // 'latency', 'memory' or 'power'
  };
  
  // Apply optimization to builder
  builder.optimizationHint = options.optimizationHint;
  builder.powerPreference = options.powerPreference;
  
  return builder;
}

// Create and optimize a convolution operation
function createOptimizedConv2d(builder, input, filters, options = {}) {
  // Default options
  const defaultOptions = {
    padding: [0, 0, 0, 0],
    strides: [1, 1],
    dilations: [1, 1],
    groups: 1,
    activation: null,
    layout: 'nchw',
    outputLayout: 'auto',
    fusedActivation: false,
  };
  
  // Merge options
  const mergedOptions = { ...defaultOptions, ...options };
  
  // Create convolution
  let conv = builder.conv2d(input, filters, mergedOptions);
  
  // Apply fused activation if requested
  if (mergedOptions.fusedActivation && mergedOptions.activation) {
    if (mergedOptions.activation === 'relu') {
      conv = builder.relu(conv);
    } else if (mergedOptions.activation === 'sigmoid') {
      conv = builder.sigmoid(conv);
    } else if (mergedOptions.activation === 'tanh') {
      conv = builder.tanh(conv);
    }
  }
  
  return conv;
}

// Create optimized model execution
async function executeOptimizedGraph(builder, graph, inputs, device = 'preferred') {
  try {
    // Get compute device
    const context = navigator.ml?.getContext() || new MLContext();
    const computeDevice = await context.getDeviceByType(device);
    
    if (!computeDevice) {
      console.warn(`Device '${device}' not available, falling back to default`);
    }
    
    // Compile graph
    const compiledGraph = await context.compile(builder, graph, {
      device: computeDevice || 'default'
    });
    
    // Execute graph
    const outputs = await compiledGraph.compute(inputs);
    
    return outputs;
  } catch (error) {
    console.error('Error executing graph:', error);
    throw error;
  }
}

// Cache compiled models for faster initialization
const compiledModelCache = new Map();

async function getOrCompileModel(builder, graph, deviceType = 'preferred', cacheKey) {
  // Check cache
  if (cacheKey && compiledModelCache.has(cacheKey)) {
    return compiledModelCache.get(cacheKey);
  }
  
  // Get context and device
  const context = navigator.ml?.getContext() || new MLContext();
  const device = await context.getDeviceByType(deviceType);
  
  // Compile graph
  const compiledGraph = await context.compile(builder, graph, {
    device: device || 'default'
  });
  
  // Cache compiled graph
  if (cacheKey) {
    compiledModelCache.set(cacheKey, compiledGraph);
  }
  
  return compiledGraph;
}
"""
        
        # OpenVINO templates
        templates["openvino"]["quantization"] = """
# OpenVINO Quantization Implementation
import openvino as ov
from openvino.runtime import Core
from openvino.tools.pot import compress_quantize_weights
from openvino.tools.pot import IEEngine
from openvino.tools.pot import load_model, save_model
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline

# Quantize model with INT8
def quantize_model_int8(model_path, dataset, output_path):
    # Load model
    model = load_model(model_path)
    
    # Create engine for metric calculation
    engine = IEEngine(config={}, data_loader=dataset)
    
    # Quantize model
    compression_pipeline = create_pipeline('quantization', engine=engine)
    compressed_model = compression_pipeline.run(model=model)
    
    # Save model
    save_model(compressed_model, output_path)
    
    return output_path

# Create representative dataset
def create_calibration_dataset(data_dir, batch_size=1, input_shape=None):
    # This is a placeholder - need to be customized for your data
    class CalibrationDataset:
        def __init__(self, data_dir, batch_size=1, input_shape=None):
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.input_shape = input_shape
            # Initialize dataset
        
        def __iter__(self):
            # Yield data samples for calibration
            for i in range(100):  # Use 100 samples for calibration
                yield self._get_sample(i)
        
        def _get_sample(self, index):
            # Return a sample for calibration
            # This needs to be implemented based on your data
            return {"input_name": None}  # Placeholder
    
    return CalibrationDataset(data_dir, batch_size, input_shape)

# Optimize model with streams
def optimize_model_with_streams(model_path, num_streams=None, cpu_cores=None):
    # Initialize OpenVINO Core
    core = Core()
    
    # Set number of streams
    if num_streams is None:
        # Auto-detect based on CPU cores
        import multiprocessing
        total_cores = cpu_cores or multiprocessing.cpu_count()
        num_streams = max(1, total_cores - 1)
    
    # Read model
    model = core.read_model(model_path)
    
    # Configure streams
    config = {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "NUM_STREAMS": str(num_streams),
        "INFERENCE_NUM_THREADS": str(num_streams)
    }
    
    # Compile model
    compiled_model = core.compile_model(model, "CPU", config)
    
    return compiled_model
"""
        
        templates["openvino"]["streams"] = """
# OpenVINO Streams Optimization for Throughput
import openvino as ov
from openvino.runtime import Core
import multiprocessing
import time
import threading
from queue import Queue
from typing import List, Dict, Any, Callable

# Create multi-stream inference engine
class MultiStreamInference:
    def __init__(self, model_path: str, num_streams: int = None, device: str = "CPU"):
        """
        Initialize multi-stream inference engine.
        
        Args:
            model_path: Path to model
            num_streams: Number of inference streams (default: auto-detect)
            device: Device to use (default: CPU)
        """
        self.model_path = model_path
        self.device = device
        
        # Auto-detect number of streams if not specified
        if num_streams is None:
            cpu_count = multiprocessing.cpu_count()
            self.num_streams = max(1, cpu_count - 1)
        else:
            self.num_streams = num_streams
        
        # Initialize OpenVINO Core
        self.core = Core()
        
        # Configure streams
        self.config = {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": str(self.num_streams),
            "INFERENCE_NUM_THREADS": str(self.num_streams)
        }
        
        # Read model
        self.model = self.core.read_model(model_path)
        
        # Get input and output names
        self.input_names = [node.get_any_name() for node in self.model.inputs]
        self.output_names = [node.get_any_name() for node in self.model.outputs]
        
        # Compile model
        self.compiled_model = self.core.compile_model(self.model, self.device, self.config)
        
        # Create infer requests
        self.infer_requests = [self.compiled_model.create_infer_request() for _ in range(self.num_streams)]
        
        # Task queue and results
        self.task_queue = Queue()
        self.result_callbacks = {}
        self.next_task_id = 0
        
        # Start worker threads
        self.workers = []
        self.stop_event = threading.Event()
        for i in range(self.num_streams):
            worker = threading.Thread(target=self._worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker_thread(self, worker_id: int):
        """Worker thread for processing inference requests."""
        infer_request = self.infer_requests[worker_id]
        
        while not self.stop_event.is_set():
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                task_id, inputs, callback = task
                
                # Set inputs
                for input_name, input_data in inputs.items():
                    infer_request.set_tensor(input_name, input_data)
                
                # Run inference
                infer_request.infer()
                
                # Get results
                results = {}
                for output_name in self.output_names:
                    results[output_name] = infer_request.get_output_tensor(output_name).data
                
                # Call callback
                if callback:
                    callback(task_id, results)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Error in worker {worker_id}: {e}")
    
    def async_infer(self, inputs: Dict[str, Any], callback: Callable = None) -> int:
        """
        Schedule asynchronous inference.
        
        Args:
            inputs: Dictionary of input tensors
            callback: Callback function to call with results
            
        Returns:
            Task ID
        """
        task_id = self.next_task_id
        self.next_task_id += 1
        
        self.task_queue.put((task_id, inputs, callback))
        
        return task_id
    
    def stop(self):
        """Stop all worker threads."""
        self.stop_event.set()
        
        # Add None tasks to unblock workers
        for _ in range(self.num_streams):
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()

# Example usage function
def run_multi_stream_inference(model_path, input_data_list, num_streams=None):
    """
    Run inference on multiple inputs using multi-stream inference.
    
    Args:
        model_path: Path to model
        input_data_list: List of input data dictionaries
        num_streams: Number of inference streams (default: auto-detect)
        
    Returns:
        List of inference results
    """
    # Create engine
    engine = MultiStreamInference(model_path, num_streams)
    
    # Results storage
    results = {}
    results_ready = threading.Event()
    results_count = 0
    expected_count = len(input_data_list)
    
    # Callback function
    def callback(task_id, result):
        nonlocal results_count
        results[task_id] = result
        results_count += 1
        if results_count >= expected_count:
            results_ready.set()
    
    # Schedule all inference requests
    for input_data in input_data_list:
        engine.async_infer(input_data, callback)
    
    # Wait for all results
    results_ready.wait()
    
    # Stop engine
    engine.stop()
    
    # Return results in order
    return [results[i] for i in range(expected_count)]
"""
        
        # Create template files
        for framework, templates_dict in templates.items():
            framework_dir = self.template_dir / framework
            os.makedirs(framework_dir, exist_ok=True)
            
            for name, content in templates_dict.items():
                file_path = framework_dir / f"{name}.py"
                if not file_path.exists():
                    with open(file_path, "w") as f:
                        f.write(content.strip())
        
        return templates
    
    def export_optimization(
        self,
        model_name: str,
        hardware_platform: str,
        recommendation_name: Optional[str] = None,
        output_format: str = "all",
        output_dir: Optional[str] = None,
        model_family: Optional[str] = None,
        batch_size: Optional[int] = None,
        precision: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export optimization recommendation to deployable configuration.
        
        Args:
            model_name: Model name
            hardware_platform: Hardware platform
            recommendation_name: Optional specific recommendation name
            output_format: Output format (python, json, yaml, script, all)
            output_dir: Optional custom output directory
            model_family: Optional model family
            batch_size: Optional batch size
            precision: Optional precision
            
        Returns:
            Dictionary with export results
        """
        if not self.client:
            return {"error": "Optimization client not available"}
        
        # Get recommendations
        recommendations = self.client.get_recommendations(
            model_name=model_name,
            hardware_platform=hardware_platform,
            model_family=model_family,
            batch_size=batch_size,
            current_precision=precision
        )
        
        if "error" in recommendations or not recommendations.get("recommendations", []):
            return {"error": "No recommendations found"}
        
        # Set output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_dir / f"{model_name}_{hardware_platform}"
        
        os.makedirs(output_path, exist_ok=True)
        
        # Filter recommendations if specific name provided
        recs = recommendations["recommendations"]
        if recommendation_name:
            recs = [r for r in recs if r["name"].lower() == recommendation_name.lower()]
            if not recs:
                return {"error": f"Recommendation '{recommendation_name}' not found"}
        
        # Export each recommendation
        exported_files = []
        
        for i, rec in enumerate(recs):
            # Create directory for recommendation
            rec_name = rec["name"].lower().replace(" ", "_")
            rec_dir = output_path / rec_name
            os.makedirs(rec_dir, exist_ok=True)
            
            # Create implementation file
            impl_file = self._create_implementation_file(rec, hardware_platform, model_name, rec_dir)
            if impl_file:
                exported_files.append(impl_file)
            
            # Create config files in different formats
            if output_format in ["json", "all"]:
                json_file = rec_dir / f"{rec_name}_config.json"
                with open(json_file, "w") as f:
                    json.dump(self._create_config_dict(rec), f, indent=2)
                exported_files.append(str(json_file))
            
            if output_format in ["yaml", "all"]:
                yaml_file = rec_dir / f"{rec_name}_config.yaml"
                with open(yaml_file, "w") as f:
                    yaml.dump(self._create_config_dict(rec), f, default_flow_style=False)
                exported_files.append(str(yaml_file))
            
            if output_format in ["python", "all"]:
                py_file = rec_dir / f"{rec_name}_config.py"
                with open(py_file, "w") as f:
                    f.write(self._create_python_config(rec))
                exported_files.append(str(py_file))
            
            # Create README
            readme_file = rec_dir / "README.md"
            with open(readme_file, "w") as f:
                f.write(self._create_readme(rec, hardware_platform, model_name))
            exported_files.append(str(readme_file))
        
        return {
            "model_name": model_name,
            "hardware_platform": hardware_platform,
            "recommendations_exported": len(recs),
            "exported_files": exported_files,
            "base_directory": str(output_path)
        }
    
    def _create_config_dict(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create configuration dictionary from recommendation.
        
        Args:
            recommendation: Recommendation dictionary
            
        Returns:
            Configuration dictionary
        """
        config = {
            "name": recommendation.get("name"),
            "description": recommendation.get("description"),
            "configuration": recommendation.get("configuration", {}),
            "expected_improvements": recommendation.get("expected_improvements", {}),
            "confidence": recommendation.get("confidence", 0)
        }
        
        return config
    
    def _create_python_config(self, recommendation: Dict[str, Any]) -> str:
        """
        Create Python configuration file content.
        
        Args:
            recommendation: Recommendation dictionary
            
        Returns:
            Python configuration content
        """
        config_dict = self._create_config_dict(recommendation)
        
        # Convert dictionary to Python code
        lines = [
            "#!/usr/bin/env python3",
            "\"\"\"",
            f"Configuration for {recommendation.get('name')}",
            "",
            f"{recommendation.get('description')}",
            "\"\"\"",
            "",
            "# Configuration settings",
            "CONFIG = {"
        ]
        
        for key, value in config_dict.items():
            if isinstance(value, dict):
                lines.append(f"    \"{key}\": {{")
                for k, v in value.items():
                    lines.append(f"        \"{k}\": {repr(v)},")
                lines.append("    },")
            else:
                lines.append(f"    \"{key}\": {repr(value)},")
        
        lines.append("}")
        lines.append("")
        lines.append("# Example usage")
        lines.append("if __name__ == \"__main__\":")
        lines.append("    print(f\"Configuration for {CONFIG['name']}:\")")
        lines.append("    for key, value in CONFIG[\"configuration\"].items():")
        lines.append("        print(f\"  {key}: {value}\")")
        lines.append("    ")
        lines.append("    print(\"\\nExpected improvements:\")")
        lines.append("    improvements = CONFIG[\"expected_improvements\"]")
        lines.append("    print(f\"  Throughput: +{improvements['throughput_improvement']*100:.1f}%\")")
        lines.append("    print(f\"  Latency: -{improvements['latency_reduction']*100:.1f}%\")")
        lines.append("    print(f\"  Memory: -{improvements['memory_reduction']*100:.1f}%\")")
        
        return "\n".join(lines)
    
    def _create_readme(
        self, 
        recommendation: Dict[str, Any],
        hardware_platform: str,
        model_name: str
    ) -> str:
        """
        Create README file content.
        
        Args:
            recommendation: Recommendation dictionary
            hardware_platform: Hardware platform
            model_name: Model name
            
        Returns:
            README content
        """
        # Format improvements
        improvements = recommendation.get("expected_improvements", {})
        throughput_imp = improvements.get("throughput_improvement", 0) * 100
        latency_red = improvements.get("latency_reduction", 0) * 100
        memory_red = improvements.get("memory_reduction", 0) * 100
        
        # Current metrics
        current_metrics = recommendation.get("current_metrics", {})
        current_throughput = current_metrics.get("throughput", "N/A")
        current_latency = current_metrics.get("latency", "N/A")
        current_memory = current_metrics.get("memory_usage", "N/A")
        
        # Improved metrics
        improved_throughput = improvements.get("improved_throughput", "N/A")
        improved_latency = improvements.get("improved_latency", "N/A")
        improved_memory = improvements.get("improved_memory", "N/A")
        
        # Configuration parameters
        config_params = recommendation.get("configuration", {})
        config_lines = []
        for key, value in config_params.items():
            config_lines.append(f"- **{key}**: {value}")
        
        # Create README content
        content = [
            f"# {recommendation.get('name')} for {model_name}",
            "",
            f"## Description",
            "",
            f"{recommendation.get('description')}",
            "",
            f"## Hardware Platform",
            "",
            f"- **Platform**: {hardware_platform}",
            "",
            f"## Configuration Parameters",
            ""
        ]
        
        # Add configuration parameters
        content.extend(config_lines)
        
        # Add expected improvements
        content.extend([
            "",
            "## Expected Improvements",
            "",
            f"- **Throughput**: +{throughput_imp:.1f}% ({current_throughput} → {improved_throughput} items/sec)",
            f"- **Latency**: -{latency_red:.1f}% ({current_latency} → {improved_latency} ms)",
            f"- **Memory Usage**: -{memory_red:.1f}% ({current_memory} → {improved_memory} MB)",
            "",
            f"## Confidence Score",
            "",
            f"- **Confidence**: {recommendation.get('confidence', 0) * 100:.1f}%",
            "",
            f"## Implementation Guide",
            "",
            f"{recommendation.get('implementation', 'No implementation guide available.')}",
            "",
            f"## Files",
            "",
            "- `config.json`: Configuration in JSON format",
            "- `config.yaml`: Configuration in YAML format",
            "- `config.py`: Configuration in Python format",
            "- `implementation.py`: Implementation script for this optimization",
            "",
            f"## Usage",
            "",
            "1. Install required dependencies",
            "2. Configure your environment using the provided configuration files",
            "3. Run the implementation script",
            "",
            f"```bash",
            f"python implementation.py --model {model_name} --config config.json",
            f"```"
        ])
        
        return "\n".join(content)
    
    def _create_implementation_file(
        self, 
        recommendation: Dict[str, Any],
        hardware_platform: str,
        model_name: str,
        output_dir: Path
    ) -> Optional[str]:
        """
        Create implementation file for the recommendation.
        
        Args:
            recommendation: Recommendation dictionary
            hardware_platform: Hardware platform
            model_name: Model name
            output_dir: Output directory
            
        Returns:
            Path to implementation file or None
        """
        # Map recommendation name to template
        templates_map = {
            # PyTorch templates
            "mixed precision": ("pytorch", "mixed_precision"),
            "quantization": ("pytorch", "quantization"),
            "tensorrt integration": ("pytorch", "tensorrt"),
            
            # TensorFlow templates
            "tf mixed precision": ("tensorflow", "mixed_precision"),
            "tf quantization": ("tensorflow", "quantization"),
            "tf tensorrt": ("tensorflow", "tensorrt"),
            
            # WebGPU templates
            "shader precompilation": ("webgpu", "shader_optimization"),
            "webgpu compute optimization": ("webgpu", "tensor_ops"),
            
            # WebNN templates
            "webnn graph optimization": ("webnn", "graph_optimization"),
            
            # OpenVINO templates
            "int8 quantization": ("openvino", "quantization"),
            "openvino streams": ("openvino", "streams")
        }
        
        # Try to find matching template
        template_key = None
        for key in templates_map.keys():
            if key in recommendation["name"].lower():
                template_key = key
                break
        
        # Default based on hardware platform
        if not template_key:
            if hardware_platform == "cuda":
                template_key = "mixed precision"
            elif hardware_platform == "cpu":
                template_key = "quantization"
            elif hardware_platform == "openvino":
                template_key = "openvino streams"
            elif hardware_platform == "webgpu":
                template_key = "shader precompilation"
            elif hardware_platform == "webnn":
                template_key = "webnn graph optimization"
            else:
                return None
        
        # Get template
        framework, template_name = templates_map[template_key]
        template_path = self.template_dir / framework / f"{template_name}.py"
        
        if not template_path.exists():
            return None
        
        # Read template
        with open(template_path, "r") as f:
            template_content = f.read()
        
        # Create implementation file
        implementation_file = output_dir / "implementation.py"
        
        with open(implementation_file, "w") as f:
            f.write(f"#!/usr/bin/env python3\n")
            f.write(f"\"\"\"\n")
            f.write(f"Implementation for {recommendation['name']} - {model_name} on {hardware_platform}\n")
            f.write(f"\n")
            f.write(f"{recommendation['description']}\n")
            f.write(f"\"\"\"\n\n")
            
            # Add imports
            f.write("import os\n")
            f.write("import sys\n")
            f.write("import json\n")
            f.write("import argparse\n")
            f.write("from pathlib import Path\n\n")
            
            # Add template content
            f.write(template_content)
            
            # Add main function
            f.write("\n\n")
            f.write("def main():\n")
            f.write("    \"\"\"Main entry point.\"\"\"\n")
            f.write("    parser = argparse.ArgumentParser(description=\"Implementation for ")
            f.write(f"{recommendation['name']}\")\n")
            f.write("    parser.add_argument(\"--model\", type=str, default=")
            f.write(f"\"{model_name}\", help=\"Model name or path\")\n")
            f.write("    parser.add_argument(\"--config\", type=str, default=\"config.json\", ")
            f.write("help=\"Configuration file\")\n")
            f.write("    parser.add_argument(\"--output\", type=str, default=\"optimized_model\", ")
            f.write("help=\"Output path\")\n")
            f.write("    args = parser.parse_args()\n\n")
            
            f.write("    # Load configuration\n")
            f.write("    with open(args.config, 'r') as f:\n")
            f.write("        config = json.load(f)\n\n")
            
            f.write("    print(f\"Applying {config['name']} to {args.model}...\")\n\n")
            
            # Add framework-specific implementation
            if framework == "pytorch":
                f.write("    # Example implementation (modify as needed)\n")
                f.write("    import torch\n")
                f.write("    # Load model (implement based on your model)\n")
                f.write("    # model = torch.load(args.model)\n\n")
                
                if template_name == "mixed_precision":
                    f.write("    # Apply mixed precision\n")
                    f.write("    # scaler = torch.cuda.amp.GradScaler()\n")
                    f.write("    # with torch.cuda.amp.autocast():\n")
                    f.write("    #     # Run model with mixed precision\n")
                    f.write("    #     pass\n")
                elif template_name == "quantization":
                    f.write("    # Apply quantization\n")
                    f.write("    # model_prepared = prepare_model_for_quantization(model)\n")
                    f.write("    # model_quantized = quantize_model(model_prepared)\n")
                elif template_name == "tensorrt":
                    f.write("    # Convert to TensorRT\n")
                    f.write("    # trt_model = convert_to_tensorrt(model, input_shapes=[(1, 3, 224, 224)])\n")
            
            elif framework == "tensorflow":
                f.write("    # Example implementation (modify as needed)\n")
                f.write("    import tensorflow as tf\n")
                f.write("    # Load model (implement based on your model)\n")
                f.write("    # model = tf.keras.models.load_model(args.model)\n\n")
                
                if template_name == "mixed_precision":
                    f.write("    # Apply mixed precision\n")
                    f.write("    # tf.keras.mixed_precision.set_global_policy('mixed_float16')\n")
                    f.write("    # model, optimizer = create_mixed_precision_model(model)\n")
                elif template_name == "quantization":
                    f.write("    # Apply quantization\n")
                    f.write("    # def representative_dataset():\n")
                    f.write("    #     # Implement based on your data\n")
                    f.write("    #     pass\n")
                    f.write("    # quantized_model = quantize_model(model, representative_dataset)\n")
            
            elif framework == "openvino":
                f.write("    # Example implementation (modify as needed)\n")
                f.write("    import openvino as ov\n")
                f.write("    # Convert model to OpenVINO format if needed\n")
                f.write("    # model_path = args.model\n\n")
                
                if template_name == "quantization":
                    f.write("    # Apply INT8 quantization\n")
                    f.write("    # calibration_dataset = create_calibration_dataset('data/')\n")
                    f.write("    # quantized_path = quantize_model_int8(model_path, calibration_dataset, args.output)\n")
                elif template_name == "streams":
                    f.write("    # Setup multi-stream inference\n")
                    f.write("    # engine = MultiStreamInference(model_path)\n")
                    f.write("    # Define input data (example)\n")
                    f.write("    # input_data_list = [{'input': input_tensor} for _ in range(10)]\n")
                    f.write("    # results = run_multi_stream_inference(model_path, input_data_list)\n")
            
            elif framework in ["webgpu", "webnn"]:
                f.write("    print(\"Web platform optimizations should be implemented in JavaScript.\")\n")
                f.write("    print(\"Please refer to the template for JavaScript implementation.\")\n")
            
            # Complete main function
            f.write("\n    print(\"Optimization complete!\")\n\n")
            f.write("if __name__ == \"__main__\":\n")
            f.write("    main()\n")
        
        # Make file executable
        os.chmod(implementation_file, 0o755)
        
        return str(implementation_file)
    
    def export_batch_optimizations(
        self,
        recommendations_report: Dict[str, Any],
        output_dir: Optional[str] = None,
        output_format: str = "all"
    ) -> Dict[str, Any]:
        """
        Export batch optimizations from a recommendations report.
        
        Args:
            recommendations_report: Report with multiple recommendations
            output_dir: Optional custom output directory
            output_format: Output format (python, json, yaml, script, all)
            
        Returns:
            Dictionary with export results
        """
        if not recommendations_report or "top_recommendations" not in recommendations_report:
            return {"error": "Invalid recommendations report"}
        
        # Set output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"batch_export_{timestamp}"
        
        os.makedirs(output_path, exist_ok=True)
        
        # Export each recommendation
        results = {
            "exported_count": 0,
            "export_details": {},
            "output_directory": str(output_path)
        }
        
        for rec_item in recommendations_report["top_recommendations"]:
            model_name = rec_item["model_name"]
            hardware_platform = rec_item["hardware_platform"]
            recommendation = rec_item["recommendation"]
            
            # Create model directory
            model_dir = output_path / f"{model_name}_{hardware_platform}"
            
            # Export recommendation
            export_result = self.export_optimization(
                model_name=model_name,
                hardware_platform=hardware_platform,
                recommendation_name=recommendation["name"],
                output_format=output_format,
                output_dir=str(model_dir)
            )
            
            if "error" not in export_result:
                results["exported_count"] += 1
                key = f"{model_name}_{hardware_platform}"
                results["export_details"][key] = export_result
        
        return results
    
    def create_archive(
        self,
        export_result: Dict[str, Any],
        archive_format: str = "zip"
    ) -> Optional[BytesIO]:
        """
        Create an archive of exported optimization files.
        
        Args:
            export_result: Result from export_optimization or export_batch_optimizations
            archive_format: Archive format (currently only 'zip' is supported)
            
        Returns:
            BytesIO object containing the archive data, or None if creation failed
        """
        if "error" in export_result:
            return None
        
        # Check if export result has files
        if "exported_files" not in export_result and "export_details" not in export_result:
            return None
        
        # Create in-memory file
        archive_buffer = BytesIO()
        
        try:
            # Create ZIP archive
            with zipfile.ZipFile(
                archive_buffer, 
                mode='w', 
                compression=zipfile.ZIP_DEFLATED, 
                compresslevel=9
            ) as archive:
                # Single export
                if "exported_files" in export_result:
                    base_dir = Path(export_result.get("base_directory", ""))
                    for file_path in export_result["exported_files"]:
                        # Get relative path for zip structure
                        file_path = Path(file_path)
                        if base_dir and str(base_dir) in str(file_path):
                            arc_name = str(file_path.relative_to(base_dir))
                        else:
                            arc_name = file_path.name
                        
                        # Add file to archive
                        if file_path.exists():
                            archive.write(file_path, arcname=arc_name)
                
                # Batch export
                elif "export_details" in export_result:
                    for model_hw, details in export_result["export_details"].items():
                        if "exported_files" in details:
                            base_dir = Path(details.get("base_directory", ""))
                            for file_path in details["exported_files"]:
                                # Get relative path for zip structure
                                file_path = Path(file_path)
                                if base_dir and str(base_dir) in str(file_path):
                                    # Use model_hw as the parent directory
                                    arc_name = f"{model_hw}/{file_path.relative_to(base_dir)}"
                                else:
                                    arc_name = f"{model_hw}/{file_path.name}"
                                
                                # Add file to archive
                                if file_path.exists():
                                    archive.write(file_path, arcname=arc_name)
            
            # Reset buffer position
            archive_buffer.seek(0)
            return archive_buffer
        
        except Exception as e:
            logger.error(f"Error creating archive: {e}")
            return None
    
    def get_archive_filename(
        self,
        export_result: Dict[str, Any],
        archive_format: str = "zip"
    ) -> str:
        """
        Generate an appropriate filename for the archive.
        
        Args:
            export_result: Result from export_optimization or export_batch_optimizations
            archive_format: Archive format (currently only 'zip' is supported)
            
        Returns:
            Suggested filename for the archive
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Single export
        if "exported_files" in export_result:
            model_name = export_result.get("model_name", "model")
            hardware = export_result.get("hardware_platform", "hw")
            filename = f"{model_name}_{hardware}_optimizations_{timestamp}.{archive_format}"
        
        # Batch export
        elif "export_details" in export_result:
            exported_count = export_result.get("exported_count", 0)
            filename = f"batch_optimizations_{exported_count}_models_{timestamp}.{archive_format}"
        
        else:
            filename = f"optimizations_{timestamp}.{archive_format}"
        
        return filename
    
    def close(self):
        """Close connections."""
        if self.client:
            self.client.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimization Recommendation Exporter")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--hardware", type=str, help="Hardware platform")
    parser.add_argument("--recommendation", type=str, help="Specific recommendation name")
    parser.add_argument("--benchmark-db", type=str, default="benchmark_db.duckdb", 
                      help="Path to benchmark database")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080",
                      help="URL of the Unified API Server")
    parser.add_argument("--api-key", type=str, help="API key for authenticated endpoints")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--output-format", type=str, choices=["python", "json", "yaml", "script", "all"],
                      default="all", help="Output format")
    parser.add_argument("--batch", type=str, help="Path to recommendations report JSON for batch export")
    parser.add_argument("--create-archive", action="store_true", help="Create ZIP archive of exported files")
    parser.add_argument("--archive-path", type=str, help="Path to save archive file (default: current directory)")
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = OptimizationExporter(
        output_dir=args.output_dir or "./optimization_configs",
        benchmark_db_path=args.benchmark_db,
        api_url=args.api_url,
        api_key=args.api_key
    )
    
    try:
        # Batch export
        if args.batch:
            try:
                with open(args.batch, 'r') as f:
                    report = json.load(f)
                
                result = exporter.export_batch_optimizations(
                    recommendations_report=report,
                    output_dir=args.output_dir,
                    output_format=args.output_format
                )
                
                if "error" in result:
                    print(f"Error exporting batch optimizations: {result['error']}")
                    return 1
                
                print(f"Exported {result['exported_count']} optimizations to {result['output_directory']}")
                
                # Create archive if requested
                if args.create_archive:
                    archive_data = exporter.create_archive(result)
                    if archive_data:
                        archive_filename = exporter.get_archive_filename(result)
                        archive_path = args.archive_path or "."
                        full_path = os.path.join(archive_path, archive_filename)
                        
                        with open(full_path, "wb") as f:
                            f.write(archive_data.getvalue())
                        
                        print(f"Created archive: {full_path}")
                    else:
                        print("Failed to create archive")
                
                return 0
            
            except Exception as e:
                print(f"Error reading recommendations report: {e}")
                return 1
        
        # Single export
        if not args.model or not args.hardware:
            print("ERROR: --model and --hardware are required for single export")
            parser.print_help()
            return 1
        
        result = exporter.export_optimization(
            model_name=args.model,
            hardware_platform=args.hardware,
            recommendation_name=args.recommendation,
            output_format=args.output_format,
            output_dir=args.output_dir
        )
        
        if "error" in result:
            print(f"Error exporting optimization: {result['error']}")
            return 1
        
        print(f"Exported {result['recommendations_exported']} optimization(s) to {result['base_directory']}")
        print(f"Files created:")
        for file in result['exported_files']:
            print(f"  - {file}")
        
        # Create archive if requested
        if args.create_archive:
            archive_data = exporter.create_archive(result)
            if archive_data:
                archive_filename = exporter.get_archive_filename(result)
                archive_path = args.archive_path or "."
                full_path = os.path.join(archive_path, archive_filename)
                
                with open(full_path, "wb") as f:
                    f.write(archive_data.getvalue())
                
                print(f"Created archive: {full_path}")
            else:
                print("Failed to create archive")
        
        return 0
    
    finally:
        exporter.close()

if __name__ == "__main__":
    sys.exit(main())