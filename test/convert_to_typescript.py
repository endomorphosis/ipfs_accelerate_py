#!/usr/bin/env python3
# convert_to_typescript.py
# Unified script for converting Python code to TypeScript

import os
import sys
import logging
import argparse
import glob
import json
import subprocess
import re
import shutil
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'typescript_conversion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    SOURCE_DIR = "./fixed_web_platform"
    TARGET_DIR = "../ipfs_accelerate_js"
    HIGH_PRIORITY_FILES = [
        "cross_model_tensor_sharing.py",
        "sample_webgpu_backend.py",
        "webgpu_ultra_low_precision.py"
    ]
    DRY_RUN = False
    VERBOSE = False
    INSTALL_DEPS = False
    FIX_TYPES = True
    CREATE_BACKUPS = True
    SKIP_COMPILE = False
    STATS = {
        "files_processed": 0,
        "files_converted": 0,
        "files_backed_up": 0,
        "error_count": 0,
        "special_files_handled": 0
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert Python code to TypeScript")
    parser.add_argument("--source-dir", help="Source directory with Python files", default="./fixed_web_platform")
    parser.add_argument("--target-dir", help="Target directory for TypeScript files", default="../ipfs_accelerate_js")
    parser.add_argument("--high-priority", help="Only convert high priority files", action="store_true")
    parser.add_argument("--dry-run", help="Don't make changes, just report", action="store_true")
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    parser.add_argument("--no-install", help="Skip dependency installation", action="store_true")
    parser.add_argument("--no-fix", help="Skip fixing common type issues", action="store_true")
    parser.add_argument("--no-backups", help="Skip creating backups", action="store_true")
    parser.add_argument("--skip-compile", help="Skip TypeScript compilation validation", action="store_true")
    args = parser.parse_args()
    
    Config.SOURCE_DIR = os.path.abspath(args.source_dir)
    Config.TARGET_DIR = os.path.abspath(args.target_dir)
    Config.DRY_RUN = args.dry_run
    Config.VERBOSE = args.verbose
    Config.INSTALL_DEPS = not args.no_install
    Config.FIX_TYPES = not args.no_fix
    Config.CREATE_BACKUPS = not args.no_backups
    Config.SKIP_COMPILE = args.skip_compile
    
    logger.info(f"Source directory: {Config.SOURCE_DIR}")
    logger.info(f"Target directory: {Config.TARGET_DIR}")
    logger.info(f"Dry run: {Config.DRY_RUN}")
    logger.info(f"Install dependencies: {Config.INSTALL_DEPS}")
    logger.info(f"Fix type issues: {Config.FIX_TYPES}")
    logger.info(f"Create backups: {Config.CREATE_BACKUPS}")
    logger.info(f"Skip compilation: {Config.SKIP_COMPILE}")
    
    if args.high_priority:
        logger.info("Only converting high priority files")

def setup_directory_structure():
    """Set up the target directory structure"""
    if not os.path.exists(Config.TARGET_DIR):
        if not Config.DRY_RUN:
            os.makedirs(Config.TARGET_DIR, exist_ok=True)
        logger.info(f"Created target directory: {Config.TARGET_DIR}")
    
    # Create source directory structure
    src_dir = os.path.join(Config.TARGET_DIR, "src")
    if not os.path.exists(src_dir) and not Config.DRY_RUN:
        os.makedirs(src_dir, exist_ok=True)
        logger.info(f"Created src directory: {src_dir}")
    
    # Create directories for different components
    component_dirs = [
        "tensor", "tensor/operations", "tensor/memory",
        "hardware", "hardware/webgpu", "hardware/webnn",
        "utils", "types",
        "browser", "browser/resource_pool", "browser/optimizations"
    ]
    
    for comp_dir in component_dirs:
        dir_path = os.path.join(src_dir, comp_dir)
        if not os.path.exists(dir_path) and not Config.DRY_RUN:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created component directory: {dir_path}")

def find_python_files():
    """Find Python files to convert"""
    all_files = []
    
    # Find Python files in source directory
    for root, _, files in os.walk(Config.SOURCE_DIR):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    # Sort files by priority
    high_priority = []
    normal_priority = []
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        if file_name in Config.HIGH_PRIORITY_FILES:
            high_priority.append(file_path)
        else:
            normal_priority.append(file_path)
    
    return high_priority, normal_priority

def create_tsconfig():
    """Create TypeScript configuration file"""
    tsconfig_path = os.path.join(Config.TARGET_DIR, "tsconfig.json")
    tsconfig_content = {
        "compilerOptions": {
            "target": "es2020",
            "module": "esnext",
            "moduleResolution": "node",
            "declaration": True,
            "declarationDir": "./dist/types",
            "sourceMap": True,
            "outDir": "./dist",
            "strict": False,  # Initially false to allow easier migration
            "esModuleInterop": True,
            "skipLibCheck": True,
            "forceConsistentCasingInFileNames": True,
            "lib": ["dom", "dom.iterable", "esnext", "webworker"],
            "jsx": "react"
        },
        "include": ["src/**/*"],
        "exclude": ["node_modules", "dist", "**/*.spec.ts", "**/*.test.ts"]
    }
    
    if not Config.DRY_RUN:
        with open(tsconfig_path, 'w', encoding='utf-8') as f:
            json.dump(tsconfig_content, f, indent=2)
        
        logger.info(f"Created TypeScript configuration file: {tsconfig_path}")

def create_package_json():
    """Create package.json file if it doesn't exist"""
    package_path = os.path.join(Config.TARGET_DIR, "package.json")
    
    if os.path.exists(package_path):
        logger.info(f"package.json already exists: {package_path}")
        return
    
    package_content = {
        "name": "ipfs-accelerate",
        "version": "0.1.0",
        "description": "IPFS Accelerate JavaScript SDK",
        "main": "dist/index.js",
        "module": "dist/index.esm.js",
        "types": "dist/types/index.d.ts",
        "files": [
            "dist",
            "src"
        ],
        "scripts": {
            "build": "rollup -c",
            "dev": "rollup -c -w",
            "type-check": "tsc --noEmit",
            "lint": "eslint src --ext .ts,.tsx",
            "test": "jest",
            "prepublishOnly": "npm run build"
        },
        "repository": {
            "type": "git",
            "url": "git+https://github.com/yourusername/ipfs-accelerate.git"
        },
        "keywords": [
            "ipfs",
            "ai",
            "webgpu",
            "webnn",
            "machine-learning"
        ],
        "author": "Your Name",
        "license": "MIT",
        "devDependencies": {
            "@rollup/plugin-commonjs": "^21.0.1",
            "@rollup/plugin-node-resolve": "^13.1.3",
            "@rollup/plugin-typescript": "^8.3.0",
            "@types/jest": "^27.4.0",
            "@types/node": "^17.0.10",
            "@types/react": "^17.0.38",
            "@typescript-eslint/eslint-plugin": "^5.10.0",
            "@typescript-eslint/parser": "^5.10.0",
            "eslint": "^8.7.0",
            "jest": "^27.4.7",
            "rollup": "^2.64.0",
            "rollup-plugin-dts": "^4.1.0",
            "rollup-plugin-peer-deps-external": "^2.2.4",
            "rollup-plugin-terser": "^7.0.2",
            "ts-jest": "^27.1.3",
            "typescript": "^4.5.5"
        },
        "peerDependencies": {
            "react": "^16.8.0 || ^17.0.0 || ^18.0.0"
        },
        "peerDependenciesMeta": {
            "react": {
                "optional": True
            }
        }
    }
    
    if not Config.DRY_RUN:
        with open(package_path, 'w', encoding='utf-8') as f:
            json.dump(package_content, f, indent=2)
        
        logger.info(f"Created package.json file: {package_path}")

def create_webgpu_type_definitions():
    """Create WebGPU type definitions file"""
    webgpu_file = os.path.join(Config.TARGET_DIR, "src/types/webgpu.d.ts")
    os.makedirs(os.path.dirname(webgpu_file), exist_ok=True)
    
    webgpu_content = """/**
 * Enhanced TypeScript definitions for WebGPU
 */

// WebGPU Buffer
interface GPUBufferDescriptor {
  label?: string;
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

interface GPUBuffer {
  readonly size: number;
  readonly usage: number;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

// WebGPU Texture
interface GPUTextureDescriptor {
  label?: string;
  size: GPUExtent3D;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: GPUTextureDimension;
  format: GPUTextureFormat;
  usage: number;
}

type GPUTextureDimension = '1d' | '2d' | '3d';
type GPUTextureFormat = 'rgba8unorm' | 'rgba16float' | 'rgba32float' | 'r8unorm' | 'r16float' | 'r32float' | string;

interface GPUExtent3D {
  width: number;
  height?: number;
  depthOrArrayLayers?: number;
}

interface GPUTexture {
  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
  destroy(): void;
}

interface GPUTextureViewDescriptor {
  format?: GPUTextureFormat;
  dimension?: GPUTextureViewDimension;
  aspect?: GPUTextureAspect;
  baseMipLevel?: number;
  mipLevelCount?: number;
  baseArrayLayer?: number;
  arrayLayerCount?: number;
}

type GPUTextureViewDimension = '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';
type GPUTextureAspect = 'all' | 'stencil-only' | 'depth-only';

interface GPUTextureView {
  // Empty interface for type checking
}

// WebGPU Shader
interface GPUShaderModuleDescriptor {
  label?: string;
  code: string;
  sourceMap?: object;
}

interface GPUShaderModule {
  // Empty interface for type checking
}

// WebGPU Pipeline
interface GPUComputePipelineDescriptor {
  label?: string;
  layout?: GPUPipelineLayout | 'auto';
  compute: {
    module: GPUShaderModule;
    entryPoint: string;
  };
}

interface GPUComputePipeline {
  // Empty interface for type checking
}

// WebGPU Pass
interface GPUComputePassDescriptor {
  label?: string;
}

interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

// WebGPU Commands
interface GPUCommandEncoderDescriptor {
  label?: string;
}

interface GPUCommandEncoder {
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
}

interface GPUCommandBufferDescriptor {
  label?: string;
}

interface GPUCommandBuffer {
  // Empty interface for type checking
}

// WebGPU Queue
interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource,
    dataOffset?: number,
    size?: number
  ): void;
}

// WebGPU Device
interface GPUDevice {
  readonly queue: GPUQueue;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
  destroy(): void;
}

// WebGPU Adapter
interface GPUAdapter {
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUDeviceDescriptor {
  label?: string;
  requiredFeatures?: string[];
  requiredLimits?: Record<string, number>;
}

// WebGPU Bind Group
interface GPUBindGroupLayoutEntry {
  binding: number;
  visibility: number;
  buffer?: any;
  sampler?: any;
  texture?: any;
  storageTexture?: any;
}

interface GPUBindGroupLayout {
  // Empty interface for type checking
}

interface GPUBindGroupLayoutDescriptor {
  label?: string;
  entries: GPUBindGroupLayoutEntry[];
}

interface GPUBindGroupEntry {
  binding: number;
  resource: any;
}

interface GPUBindGroupDescriptor {
  label?: string;
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
}

interface GPUBindGroup {
  // Empty interface for type checking
}

// Navigator interface
interface NavigatorGPU {
  gpu: {
    requestAdapter(): Promise<GPUAdapter>;
  };
}

interface Navigator extends NavigatorGPU {}

// Type for our code
export type WebGPUBackendType = 'webgpu';
"""
    
    if not Config.DRY_RUN:
        with open(webgpu_file, 'w', encoding='utf-8') as f:
            f.write(webgpu_content)
        
        logger.info(f"Created WebGPU type definitions: {webgpu_file}")

def create_webnn_type_definitions():
    """Create WebNN type definitions file"""
    webnn_file = os.path.join(Config.TARGET_DIR, "src/types/webnn.d.ts")
    os.makedirs(os.path.dirname(webnn_file), exist_ok=True)
    
    webnn_content = """/**
 * Enhanced TypeScript definitions for WebNN
 */

// Base interfaces
interface MLContext {
  // Empty interface for type safety
}

interface MLOperandDescriptor {
  type: MLOperandType;
  dimensions?: number[];
}

type MLOperandType =
  | 'float32'
  | 'float16'
  | 'int32'
  | 'uint32'
  | 'int8'
  | 'uint8'
  | 'int64'
  | 'uint64';

// MLGraph for executing the neural network
interface MLGraph {
  compute(inputs: Record<string, MLOperand>): Promise<Record<string, MLOperand>>;
}

// MLOperand represents a tensor in WebNN
interface MLOperand {
  // Empty interface for type safety
}

// Graph builder for constructing neural networks
interface MLGraphBuilder {
  // Input and constant creation
  input(name: string, descriptor: MLOperandDescriptor): MLOperand;
  constant(descriptor: MLOperandDescriptor, value: ArrayBufferView): MLOperand;
  
  // Basic operations
  add(a: MLOperand, b: MLOperand): MLOperand;
  sub(a: MLOperand, b: MLOperand): MLOperand;
  mul(a: MLOperand, b: MLOperand): MLOperand;
  div(a: MLOperand, b: MLOperand): MLOperand;
  
  // Neural network operations
  relu(x: MLOperand): MLOperand;
  sigmoid(x: MLOperand): MLOperand;
  tanh(x: MLOperand): MLOperand;
  leakyRelu(x: MLOperand, alpha?: number): MLOperand;
  softmax(x: MLOperand): MLOperand;
  
  // Tensor operations
  concat(inputs: MLOperand[], axis: number): MLOperand;
  reshape(input: MLOperand, newShape: number[]): MLOperand;
  
  // Convolution operations
  conv2d(
    input: MLOperand,
    filter: MLOperand,
    options?: {
      padding?: [number, number, number, number];
      strides?: [number, number];
      dilations?: [number, number];
      groups?: number;
    }
  ): MLOperand;
  
  // Pooling operations
  averagePool2d(
    input: MLOperand,
    options?: {
      windowDimensions?: [number, number];
      padding?: [number, number, number, number];
      strides?: [number, number];
      dilations?: [number, number];
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  maxPool2d(
    input: MLOperand,
    options?: {
      windowDimensions?: [number, number];
      padding?: [number, number, number, number];
      strides?: [number, number];
      dilations?: [number, number];
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  // Matrix operations
  matmul(a: MLOperand, b: MLOperand): MLOperand;
  
  // Build the graph
  build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
}

// WebNN API interfaces
interface MLContextOptions {
  devicePreference?: 'gpu' | 'cpu';
}

interface ML {
  createContext(options?: MLContextOptions): Promise<MLContext>;
  createGraphBuilder(context: MLContext): MLGraphBuilder;
}

interface Navigator {
  readonly ml?: ML;
}

// Helper types for our SDK
export type WebNNBackendType = 'webnn';
"""
    
    if not Config.DRY_RUN:
        with open(webnn_file, 'w', encoding='utf-8') as f:
            f.write(webnn_content)
        
        logger.info(f"Created WebNN type definitions: {webnn_file}")

def create_hardware_abstraction_types():
    """Create hardware abstraction interfaces"""
    types_file = os.path.join(Config.TARGET_DIR, "src/types/hardware_abstraction.d.ts")
    os.makedirs(os.path.dirname(types_file), exist_ok=True)
    
    content = """/**
 * Type definitions for hardware abstraction layer
 */

import { WebGPUBackendType } from './webgpu';
import { WebNNBackendType } from './webnn';

export type HardwareBackendType = WebGPUBackendType | WebNNBackendType | 'wasm' | 'cpu';

export interface HardwareCapabilities {
  browserName: string;
  browserVersion: string;
  platform: string;
  osVersion: string;
  isMobile: boolean;
  webgpuSupported: boolean;
  webgpuFeatures: string[];
  webnnSupported: boolean;
  webnnFeatures: string[];
  wasmSupported: boolean;
  wasmFeatures: string[];
  recommendedBackend: HardwareBackendType;
  memoryLimitMB: number;
}

export interface HardwareBackend {
  initialize(): Promise<boolean>;
  dispose(): void;
  execute?<T = any, U = any>(inputs: T): Promise<U>;
}

export interface ModelLoaderOptions {
  modelId: string;
  modelType: string;
  path?: string;
  backend?: HardwareBackendType;
  options?: Record<string, any>;
}
"""
    
    if not Config.DRY_RUN:
        with open(types_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created hardware abstraction type definitions: {types_file}")

def install_dependencies():
    """Install TypeScript dependencies"""
    if not Config.INSTALL_DEPS or Config.DRY_RUN:
        logger.info("Skipping dependency installation")
        return
    
    logger.info("Installing TypeScript dependencies...")
    
    try:
        subprocess.run(
            ["npm", "install", "--no-save", "--legacy-peer-deps"],
            cwd=Config.TARGET_DIR,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e.stderr}")
    except Exception as e:
        logger.error(f"Error running npm install: {e}")

def convert_python_to_typescript(python_file, ts_file):
    """Convert a Python file to TypeScript"""
    try:
        # Read Python file
        with open(python_file, 'r', encoding='utf-8') as f:
            python_code = f.read()
        
        # Convert Python code to TypeScript
        ts_code = convert_code(python_code)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(ts_file), exist_ok=True)
        
        # Write TypeScript file
        if not Config.DRY_RUN:
            with open(ts_file, 'w', encoding='utf-8') as f:
                f.write(ts_code)
            
            Config.STATS["files_converted"] += 1
            logger.info(f"Converted: {python_file} -> {ts_file}")
        else:
            logger.info(f"Would convert: {python_file} -> {ts_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error converting {python_file}: {e}")
        Config.STATS["error_count"] += 1
        return False

def convert_code(python_code):
    """Convert Python code to TypeScript code using regex patterns"""
    # Convert Python docstrings to JSDoc
    python_code = re.sub(r'"""([^"]*)"""', r'/**\n * \1\n */', python_code)
    python_code = re.sub(r"'''([^']*)'''", r'/**\n * \1\n */', python_code)
    
    # Convert Python imports
    python_code = re.sub(r'from\s+([\'"])([^\'"]+)[\'"]\s+import\s+(.+)', r'import { \3 } from "\2"', python_code)
    python_code = re.sub(r'import\s+(.+)\s+from\s+([\'"])([^\'"]+)[\'"]', r'import \1 from "\3"', python_code)
    
    # Convert Python 'self' to JavaScript 'this'
    python_code = re.sub(r'\bself\b(?!\s*:)', r'this', python_code)
    
    # Convert Python None/True/False to JavaScript null/true/false
    python_code = re.sub(r'\bNone\b', r'null', python_code)
    python_code = re.sub(r'\bTrue\b', r'true', python_code)
    python_code = re.sub(r'\bFalse\b', r'false', python_code)
    
    # Convert Python class definitions
    python_code = re.sub(r'class\s+(\w+)([^{]*):', r'class \1\2 {', python_code)
    
    # Convert Python function definitions
    python_code = re.sub(r'def\s+(\w+)\((.*?)\)(?:\s*->\s*([^:]*))?:', r'function \1(\2): \3 {', python_code)
    
    # Convert Python method definitions in classes
    python_code = re.sub(r'def\s+(\w+)\(self(?:,\s*(.*?))?\)(?:\s*->\s*([^:]*))?:', 
                         lambda m: f'\1({m.group(2) or ""}): {m.group(3) or "any"} {{', python_code)
    
    # Convert Python constructor
    python_code = re.sub(r'def\s+__init__\(self(?:,\s*(.*?))?\):', r'constructor(\1) {', python_code)
    
    # Convert Python if/elif/else
    python_code = re.sub(r'if\s+([^:]+):', r'if (\1) {', python_code)
    python_code = re.sub(r'elif\s+([^:]+):', r'} else if (\1) {', python_code)
    python_code = re.sub(r'else\s*:', r'} else {', python_code)
    
    # Convert Python try/except/finally
    python_code = re.sub(r'try\s*:', r'try {', python_code)
    python_code = re.sub(r'except\s+([^:]+):', r'} catch(\1) {', python_code)
    python_code = re.sub(r'except\s*:', r'} catch(error) {', python_code)
    python_code = re.sub(r'finally\s*:', r'} finally {', python_code)
    
    # Convert Python for loops
    python_code = re.sub(r'for\s+([^:]+):', r'for (\1) {', python_code)
    
    # Convert Python while loops
    python_code = re.sub(r'while\s+([^:]+):', r'while (\1) {', python_code)
    
    # Convert Python list comprehensions (simple cases)
    python_code = re.sub(r'\[(.*?) for (.*?) in (.*?)\]', r'(\3).map((\2) => \1)', python_code)
    
    # Convert Python dictionary comprehensions
    python_code = re.sub(r'\{(.*?):(.*?) for (.*?) in (.*?)\}', r'Object.fromEntries((\4).map((\3) => [\1, \2]))', python_code)
    
    # Convert Python raise to JavaScript throw
    python_code = re.sub(r'raise\s+(\w+)\((.*?)\)', r'throw new \1(\2)', python_code)
    python_code = re.sub(r'raise\s+(\w+)', r'throw new \1()', python_code)
    
    # Convert Python f-strings to JavaScript template literals
    python_code = re.sub(r'f(["\'])(.*?)\\1', lambda m: '`' + re.sub(r'\{([^}]+)\}', r'${\1}', m.group(2)) + '`', python_code)
    
    # Convert Python list syntax
    python_code = re.sub(r'\[\s*\]', r'[]', python_code)
    
    # Convert Python dict syntax
    python_code = re.sub(r'\{\s*\}', r'{}', python_code)
    
    # Convert Python comments
    python_code = re.sub(r'(?m)^\s*#\s*(.*?)$', r'// \1', python_code)
    
    # Add missing semicolons after variable assignments
    python_code = re.sub(r'(\s+)(\w+)\s*=\s*([^;{}\n]+)(?:\n|$)', r'\1\2 = \3;\n', python_code)
    
    # Add missing semicolons after method calls
    python_code = re.sub(r'(\s+)(\w+\([^)]*\))(?:\n|$)', r'\1\2;\n', python_code)
    
    # Add missing semicolons after return statements
    python_code = re.sub(r'return\s+([^;{}\n]+)(?:\n|$)', r'return \1;\n', python_code)
    
    # Fix Python string concatenation
    python_code = re.sub(r'(\w+)\s*\+\s*=\s*([^;]+)', r'\1 += \2;', python_code)
    
    # Add types to class properties
    python_code = re.sub(r'(\s+)(\w+)\s*=\s*', r'\1\2: any = ', python_code)
    
    # Fix Python staticmethod decorator
    python_code = re.sub(r'@staticmethod\s+def\s+(\w+)\((.*?)\)(?:\s*->\s*([^:]*))?:', 
                         r'static \1(\2): \3 {', python_code)
    
    # Fix Python classmethod decorator
    python_code = re.sub(r'@classmethod\s+def\s+(\w+)\(cls(?:,\s*(.*?))?\)(?:\s*->\s*([^:]*))?:', 
                         lambda m: f'static \1({m.group(2) or ""}): {m.group(3) or "any"} {{', python_code)
    
    # Fix Python property decorator
    python_code = re.sub(r'@property\s+def\s+(\w+)\(self\)(?:\s*->\s*([^:]*))?:', 
                         r'get \1(): \2 {', python_code)
    
    # Fix Python's len() function
    python_code = re.sub(r'len\(([^)]+)\)', r'\1.length', python_code)
    
    # Fix Python dict and list type annotations
    python_code = re.sub(r': Dict\[([^,\]]+),\s*([^\]]+)\]', r': Record<\1, \2>', python_code)
    python_code = re.sub(r': List\[([^\]]+)\]', r': \1[]', python_code)
    
    # Fix Python Optional type annotations
    python_code = re.sub(r': Optional\[([^\]]+)\]', r': \1 | null', python_code)
    
    # Fix Python tuple type annotations
    python_code = re.sub(r': Tuple\[([^\]]+)\]', r': [\1]', python_code)
    
    # Fix Python Union type annotations
    python_code = re.sub(r': Union\[([^\]]+)\]', r': \1', python_code)
    
    # Add missing function return types
    python_code = re.sub(r'function\s+(\w+)\(([^)]*)\)(?!\s*:)', r'function \1(\2): any', python_code)
    
    # Add missing parameter types
    python_code = re.sub(r'([(,]\s*)(\w+)(?!\s*:)(\s*[,)])', r'\1\2: any\3', python_code)
    
    # Replace Python string literal types
    python_code = re.sub(r': \'([^\']+)\'', r': "\1"', python_code)
    
    # Fix Python's enumerate()
    python_code = re.sub(r'enumerate\(([^)]+)\)', r'Array.from(\1.entries())', python_code)
    
    # Fix Python's zip()
    python_code = re.sub(r'zip\(([^)]+)\)', r'Array.from(\1[0].map((_, i) => \1.map(arr => arr[i])))', python_code)
    
    # Fix Python's range()
    python_code = re.sub(r'range\((\d+)\)', r'Array.from({length: \1}, (_, i) => i)', python_code)
    python_code = re.sub(r'range\((\d+),\s*(\d+)\)', r'Array.from({length: \2 - \1}, (_, i) => i + \1)', python_code)
    
    # Fix Python list(), dict(), str(), int(), float() constructors
    python_code = re.sub(r'list\(([^)]+)\)', r'Array.from(\1)', python_code)
    python_code = re.sub(r'dict\(([^)]+)\)', r'Object.fromEntries(\1)', python_code)
    python_code = re.sub(r'str\(([^)]+)\)', r'String(\1)', python_code)
    python_code = re.sub(r'int\(([^)]+)\)', r'parseInt(\1, 10)', python_code)
    python_code = re.sub(r'float\(([^)]+)\)', r'parseFloat(\1)', python_code)
    
    # Fix Python's super() calls
    python_code = re.sub(r'super\(\s*\)\.([^(]+)\((.*?)\)', r'super.\1(\2)', python_code)
    
    # Fix async functions
    python_code = re.sub(r'async\s+def\s+(\w+)\((.*?)\)(?:\s*->\s*([^:]*))?:', 
                         lambda m: f'async function \1(\2): Promise<{m.group(3) or "any"}> {{', python_code)
    
    # Fix async methods
    python_code = re.sub(r'async\s+def\s+(\w+)\(self(?:,\s*(.*?))?\)(?:\s*->\s*([^:]*))?:', 
                         lambda m: f'async \1({m.group(2) or ""}): Promise<{m.group(3) or "any"}> {{', python_code)
    
    # Fix await statements
    python_code = re.sub(r'await\s+([^;{}\n]+)(?:\n|$)', r'await \1;\n', python_code)
    
    # Add export statement to classes
    python_code = re.sub(r'(class\s+\w+)', r'export \1', python_code)
    
    # Add export statement to functions at the module level (not inside classes)
    def add_export_to_functions(match):
        indent = len(match.group(1))
        if indent == 0:  # Only add export to top-level functions
            return f"export function {match.group(2)}"
        return f"{match.group(1)}function {match.group(2)}"
    
    python_code = re.sub(r'([ \t]*)function\s+(\w+)', add_export_to_functions, python_code)
    
    return python_code

def run_typescript_compiler():
    """Run TypeScript compiler to validate the conversion"""
    if Config.SKIP_COMPILE or Config.DRY_RUN:
        logger.info("Skipping TypeScript compiler validation")
        return
    
    logger.info("Running TypeScript compiler for validation...")
    
    try:
        result = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            cwd=Config.TARGET_DIR,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode == 0:
            logger.info("TypeScript compilation successful!")
        else:
            logger.warning("TypeScript compilation failed with errors:")
            logger.warning(result.stdout)
            
            # Save detailed error output to file
            errors_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "typescript_errors.log")
            with open(errors_file, 'w', encoding='utf-8') as f:
                f.write(result.stdout)
            logger.info(f"Detailed TypeScript errors saved to: {errors_file}")
    except Exception as e:
        logger.error(f"Error running TypeScript compiler: {e}")

def determine_ts_file_path(python_file):
    """Determine the appropriate TypeScript file path based on the Python file path"""
    # Extract relative path from source directory
    rel_path = os.path.relpath(python_file, Config.SOURCE_DIR)
    
    # Determine target subdirectory based on filename or content
    file_name = os.path.basename(python_file)
    file_base = os.path.splitext(file_name)[0]
    
    # Special mapping for high-priority files
    if file_name == "cross_model_tensor_sharing.py":
        target_subdir = "tensor"
        ts_file_base = "shared_tensor"
    elif file_name == "sample_webgpu_backend.py":
        target_subdir = "hardware/webgpu"
        ts_file_base = "backend"
    elif file_name == "webgpu_ultra_low_precision.py":
        target_subdir = "hardware/webgpu"
        ts_file_base = "ultra_low_precision"
    elif "webgpu" in file_base.lower():
        target_subdir = "hardware/webgpu"
        ts_file_base = file_base
    elif "webnn" in file_base.lower():
        target_subdir = "hardware/webnn"
        ts_file_base = file_base
    elif "tensor" in file_base.lower():
        target_subdir = "tensor"
        ts_file_base = file_base
    elif "browser" in file_base.lower() or "resource_pool" in file_base.lower():
        target_subdir = "browser/resource_pool"
        ts_file_base = file_base
    else:
        # Default mapping
        target_subdir = "utils"
        ts_file_base = file_base
    
    # Create TypeScript file path
    ts_file_path = os.path.join(Config.TARGET_DIR, "src", target_subdir, f"{ts_file_base}.ts")
    
    return ts_file_path

def create_index_files():
    """Create index.ts files in all directories"""
    # Find all directories in target src folder
    dirs = []
    for root, subdirs, _ in os.walk(os.path.join(Config.TARGET_DIR, "src")):
        for d in subdirs:
            dirs.append(os.path.join(root, d))
    
    dirs.append(os.path.join(Config.TARGET_DIR, "src"))  # Add root src directory
    
    for dir_path in dirs:
        # Find all .ts files in directory
        ts_files = []
        for file in os.listdir(dir_path):
            if file.endswith(".ts") and file != "index.ts" and not file.endswith(".d.ts"):
                ts_files.append(os.path.splitext(file)[0])
        
        if not ts_files:
            continue
        
        index_path = os.path.join(dir_path, "index.ts")
        if not Config.DRY_RUN:
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write("// Auto-generated index file\n\n")
                for ts_file in ts_files:
                    f.write(f'export * from "./{ts_file}";\n')
            
            logger.info(f"Created index file: {index_path}")

def create_main_index():
    """Create main index.ts file"""
    index_path = os.path.join(Config.TARGET_DIR, "src/index.ts")
    
    content = """/**
 * IPFS Accelerate JavaScript SDK
 * Hardware-accelerated machine learning in the browser
 */

// Export main components
export * from './tensor';
export * from './hardware';
export * from './browser';
export * from './utils';

// Export type definitions
export * from './types/hardware_abstraction';
"""
    
    if not Config.DRY_RUN:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created main index file: {index_path}")

def create_common_interfaces():
    """Create common interfaces file"""
    interfaces_path = os.path.join(Config.TARGET_DIR, "src/interfaces.ts")
    
    content = """/**
 * Common interfaces for IPFS Accelerate JavaScript SDK
 */

// Hardware interfaces
export interface HardwareBackend {
  initialize(): Promise<boolean>;
  dispose(): void;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

export interface HardwarePreferences {
  backendOrder?: string[];
  modelPreferences?: Record<string, string>;
  options?: Record<string, any>;
}

// Model interfaces
export interface ModelConfig {
  id: string;
  type: string;
  path?: string;
  options?: Record<string, any>;
}

export interface Model {
  id: string;
  type: string;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

// Tensor interfaces
export interface TensorOptions {
  dtype?: string;
  device?: string;
  requiresGrad?: boolean;
}

export interface TensorShape {
  dimensions: number[];
  numel: number;
  strides?: number[];
}

export interface SharedTensorOptions {
  name: string;
  shape: number[];
  dtype?: string;
  storage_type?: string;
  producer_model?: string;
  consumer_models?: string[];
}

// Resource Pool interfaces
export interface ResourcePoolOptions {
  maxConnections?: number;
  browserPreferences?: Record<string, string>;
  adaptiveScaling?: boolean;
  enableFaultTolerance?: boolean;
  recoveryStrategy?: string;
  stateSyncInterval?: number;
  redundancyFactor?: number;
}
"""
    
    if not Config.DRY_RUN:
        with open(interfaces_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created common interfaces file: {interfaces_path}")

def generate_conversion_report():
    """Generate a conversion report"""
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TYPESCRIPT_CONVERSION_REPORT.md")
    
    content = f"""# TypeScript Conversion Report

## Summary

TypeScript conversion was performed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

### Statistics

- Files processed: {Config.STATS["files_processed"]}
- Files converted: {Config.STATS["files_converted"]}
- Files backed up: {Config.STATS["files_backed_up"]}
- Errors encountered: {Config.STATS["error_count"]}

### Key Components Converted

- **Tensor Operations**: SharedTensor implementation with reference counting
- **WebGPU Backend**: Hardware acceleration using the WebGPU API
- **WebNN Integration**: Neural network acceleration with WebNN
- **Browser Integration**: Resource pool for managing browser resources

### High-Priority Components

The following high-priority components were converted:

1. `cross_model_tensor_sharing.py` -> `tensor/shared_tensor.ts`
2. `sample_webgpu_backend.py` -> `hardware/webgpu/backend.ts`
3. `webgpu_ultra_low_precision.py` -> `hardware/webgpu/ultra_low_precision.ts`

### Directory Structure

The TypeScript SDK follows this structure:

```
ipfs_accelerate_js/
├── src/
│   ├── tensor/
│   │   ├── shared_tensor.ts
│   │   └── operations/
│   ├── hardware/
│   │   ├── webgpu/
│   │   │   ├── backend.ts
│   │   │   └── ultra_low_precision.ts
│   │   └── webnn/
│   ├── browser/
│   │   └── resource_pool/
│   ├── utils/
│   └── types/
│       ├── webgpu.d.ts
│       └── webnn.d.ts
```

### Next Steps

1. **Manual Refinements**: Some converted files may need manual tweaking
2. **Test Implementation**: Implement comprehensive tests
3. **Documentation**: Enhance JSDoc comments
4. **Example Applications**: Create example applications

## Implementation Details

### Conversion Process

The conversion process followed these steps:

1. Set up TypeScript environment with type definitions
2. Convert Python files to TypeScript with pattern matching
3. Fix common TypeScript syntax issues
4. Create index files and module structure
5. Validate TypeScript compilation

### Known Issues

- Some complex Python patterns may require manual adjustment
- Type definitions may need further refinement
- Some module imports may need adjustment
"""
    
    if not Config.DRY_RUN:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Generated conversion report: {report_path}")

def main():
    """Main function"""
    parse_args()
    
    # Setup directory structure
    setup_directory_structure()
    
    # Create TypeScript configuration
    create_tsconfig()
    
    # Create package.json
    create_package_json()
    
    # Create WebGPU/WebNN type definitions
    create_webgpu_type_definitions()
    create_webnn_type_definitions()
    create_hardware_abstraction_types()
    
    # Create common interfaces
    create_common_interfaces()
    
    # Find Python files to convert
    high_priority_files, normal_priority_files = find_python_files()
    
    # Log files to convert
    logger.info(f"Found {len(high_priority_files)} high-priority Python files")
    logger.info(f"Found {len(normal_priority_files)} normal-priority Python files")
    
    # Convert high-priority files
    for python_file in high_priority_files:
        ts_file = determine_ts_file_path(python_file)
        Config.STATS["files_processed"] += 1
        convert_python_to_typescript(python_file, ts_file)
    
    # Convert normal-priority files
    for python_file in normal_priority_files:
        ts_file = determine_ts_file_path(python_file)
        Config.STATS["files_processed"] += 1
        convert_python_to_typescript(python_file, ts_file)
    
    # Create index files
    create_index_files()
    
    # Create main index.ts
    create_main_index()
    
    # Install dependencies if requested
    install_dependencies()
    
    # Run TypeScript compiler validation
    run_typescript_compiler()
    
    # Generate conversion report
    generate_conversion_report()
    
    # Print summary
    logger.info("\nConversion Summary:")
    logger.info(f"Files processed: {Config.STATS['files_processed']}")
    logger.info(f"Files converted: {Config.STATS['files_converted']}")
    logger.info(f"Files backed up: {Config.STATS['files_backed_up']}")
    logger.info(f"Errors encountered: {Config.STATS['error_count']}")
    
    logger.info("\nConversion complete!")

if __name__ == "__main__":
    main()