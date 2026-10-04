#!/usr/bin/env python3
# Python to JavaScript/TypeScript Converter for IPFS Accelerate
# This script enhances the WebGPU/WebNN migration by converting Python code to JavaScript/TypeScript

import os
import sys
import re
import glob
import json
import shutil
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'ipfs_accelerate_js_py_converter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Global configuration
class Config:
    DRY_RUN = False
    FORCE = False
    SOURCE_DIR = None
    TARGET_DIR = None
    LOG_FILE = None
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    FIXED_WEB_PLATFORM_DIR = None
    ENABLE_VERBOSE = False
    # Directories to exclude from search
    EXCLUDE_DIRS = [
        "transformers_docs_built",
        "archive",
        "__pycache__",
        "node_modules",
        ".git",
        "huggingface_doc_builder"
    ]
    MIGRATION_STATS = {
        "files_processed": 0,
        "files_converted": 0,
        "conversion_failures": 0,
        "empty_files_created": 0,
        "copied_files": 0,
        "webgpu_files": 0,
        "webnn_files": 0,
        "wgsl_shaders": 0,
    }
    CONVERTED_FILES = set()

def setup_config(args):
    """Initialize global configuration based on command line arguments"""
    Config.DRY_RUN = args.dry_run
    Config.FORCE = args.force
    Config.ENABLE_VERBOSE = args.verbose
    
    # Set source directory (default to current working directory)
    Config.SOURCE_DIR = os.path.abspath(os.getcwd())
    
    # Set parent directory
    parent_dir = os.path.dirname(Config.SOURCE_DIR)
    
    # Set target directory
    if args.target_dir:
        Config.TARGET_DIR = os.path.abspath(args.target_dir)
    else:
        Config.TARGET_DIR = os.path.join(parent_dir, "ipfs_accelerate_js")
    
    # Find fixed_web_platform directory
    fixed_web_platform = os.path.join(parent_dir, "fixed_web_platform")
    if os.path.isdir(fixed_web_platform):
        Config.FIXED_WEB_PLATFORM_DIR = fixed_web_platform
    else:
        logger.warning(f"Could not find fixed_web_platform directory at {fixed_web_platform}")
    
    # Set log file
    Config.LOG_FILE = os.path.join(parent_dir, f"ipfs_accelerate_js_py_converter_{Config.TIMESTAMP}.log")
    
    logger.info(f"Configuration initialized:")
    logger.info(f"  Source directory: {Config.SOURCE_DIR}")
    logger.info(f"  Target directory: {Config.TARGET_DIR}")
    logger.info(f"  Fixed web platform directory: {Config.FIXED_WEB_PLATFORM_DIR}")
    logger.info(f"  Dry run: {Config.DRY_RUN}")
    logger.info(f"  Force: {Config.FORCE}")
    logger.info(f"  Log file: {Config.LOG_FILE}")

# File type detection and mapping
class FileTypes:
    PYTHON = 'python'
    TYPESCRIPT = 'typescript'
    JAVASCRIPT = 'javascript'
    WGSL = 'wgsl'
    HTML = 'html'
    CSS = 'css'
    MARKDOWN = 'markdown'
    JSON = 'json'
    UNKNOWN = 'unknown'

    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """Detect file type based on extension and content"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Check by extension
        if ext in ('.py'):
            return FileTypes.PYTHON
        elif ext in ('.ts', '.tsx'):
            return FileTypes.TYPESCRIPT
        elif ext in ('.js', '.jsx'):
            return FileTypes.JAVASCRIPT
        elif ext in ('.wgsl'):
            return FileTypes.WGSL
        elif ext in ('.html', '.htm'):
            return FileTypes.HTML
        elif ext in ('.css'):
            return FileTypes.CSS
        elif ext in ('.md', '.markdown'):
            return FileTypes.MARKDOWN
        elif ext in ('.json'):
            return FileTypes.JSON
            
        # If no extension match, check content for specific patterns
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(4096)  # Read first 4KB
                
                if re.search(r'import\s+[\w\s,{}]+\s+from', content) or re.search(r'export\s+(class|function|const)', content):
                    return FileTypes.TYPESCRIPT
                elif 'function' in content or 'var ' in content or 'let ' in content or 'const ' in content:
                    return FileTypes.JAVASCRIPT
                elif 'def ' in content or 'class ' in content or 'import ' in content and '#' in content:
                    return FileTypes.PYTHON
                elif '@stage' in content or '@compute' in content or 'fn main' in content:
                    return FileTypes.WGSL
                elif '<!DOCTYPE html>' in content or '<html' in content:
                    return FileTypes.HTML
                elif '{' in content and '}' in content and '"' in content:
                    try:
                        json.loads(content)
                        return FileTypes.JSON
                    except:
                        pass
        except Exception as e:
            if Config.ENABLE_VERBOSE:
                logger.debug(f"Error reading {file_path}: {e}")
        
        return FileTypes.UNKNOWN

    @staticmethod
    def get_output_extension(file_type: str) -> str:
        """Get appropriate file extension for output files"""
        if file_type == FileTypes.PYTHON:
            return '.ts'  # Convert Python to TypeScript
        elif file_type == FileTypes.TYPESCRIPT:
            return '.ts'
        elif file_type == FileTypes.JAVASCRIPT:
            return '.js'
        elif file_type == FileTypes.WGSL:
            return '.wgsl'
        elif file_type == FileTypes.HTML:
            return '.html'
        elif file_type == FileTypes.CSS:
            return '.css'
        elif file_type == FileTypes.MARKDOWN:
            return '.md'
        elif file_type == FileTypes.JSON:
            return '.json'
        else:
            return '.txt'

# Python to TypeScript converter
class PyToTsConverter:
    # Pattern mapping from Python to TypeScript
    PATTERN_MAP = [
        # Import statements with better handling of paths and relative imports
        (r'import\s+(\w+)', r'import * as $1'),
        (r'from\s+(\w+)\s+import\s+\{([^}]+)\}', r'import { $2 } from "$1"'),
        (r'from\s+(\w+)\s+import\s+(.+)', r'import { $2 } from "$1"'),
        (r'from\s+\.(\w+)\s+import\s+(.+)', r'import { $2 } from "./$1"'),
        (r'from\s+\.\.\s+import\s+(.+)', r'import { $1 } from ".."'),
        (r'class\s+(\w+)(?:\((\w+)\))?:', r'class $1 extends $2 {'),
        (r'class\s+(\w+):', r'class $1 {'),
        (r'def\s+__init__\s*\(self(?:,\s*([^)]+))?\):', r'constructor($1) {'),
        (r'def\s+(\w+)\s*\((self)(?:,\s*([^)]+))?\)\s*->\s*(\w+):', r'$1($3): $4 {'),
        (r'def\s+(\w+)\s*\((self)(?:,\s*([^)]+))?\):', r'$1($3) {'),
        (r'async\s+def\s+(\w+)\s*\((self)(?:,\s*([^)]+))?\)\s*->\s*(\w+):', r'async $1($3): Promise<$4> {'),
        (r'async\s+def\s+(\w+)\s*\((self)(?:,\s*([^)]+))?\):', r'async $1($3): Promise<any> {'),
        (r'@staticmethod\s+def\s+(\w+)\s*\(([^)]*)\)\s*->\s*(\w+):', r'static $1($2): $3 {'),
        (r'@staticmethod\s+def\s+(\w+)\s*\(([^)]*)\):', r'static $1($2): any {'),
        (r'@property\s+def\s+(\w+)\s*\(self\)\s*->\s*(\w+):', r'get $1(): $2 {'),
        (r'@property\s+def\s+(\w+)\s*\(self\):', r'get $1(): any {'),
        (r'@(\w+)\.setter\s+def\s+\1\s*\(self,\s*(\w+)(?:\s*:\s*([^)]+))?\):', r'set $1($2: $3) {'),
        (r'(\w+):\s*str', r'$1: string'),
        (r'(\w+):\s*int', r'$1: number'),
        (r'(\w+):\s*float', r'$1: number'),
        (r'(\w+):\s*bool', r'$1: boolean'),
        (r'(\w+):\s*List\[(\w+)\]', r'$1: $2[]'),
        (r'(\w+):\s*List\[([\w\[\]\.]+)\]', r'$1: $2[]'),
        (r'(\w+):\s*Dict\[(\w+),\s*(\w+)\]', r'$1: Record<$2, $3>'),
        (r'(\w+):\s*Dict\[([\w\[\]\.]+),\s*([\w\[\]\.]+)\]', r'$1: Record<$2, $3>'),
        (r'(\w+):\s*Optional\[(\w+)\]', r'$1: $2 | null'),
        (r'(\w+):\s*Optional\[([\w\[\]\.]+)\]', r'$1: $2 | null'),
        (r'(\w+):\s*Union\[([^\]]+)\]', r'$1: $2'),
        (r'(\w+):\s*Tuple\[([^\]]+)\]', r'$1: [$2]'),
        (r'(\w+):\s*Any', r'$1: any'),
        (r'(\w+):\s*Callable\[\[([\w\s,]+)\],\s*(\w+)\]', r'$1: ($2) => $3'),
        (r'(\w+):\s*Literal\[([^\]]+)\]', r'$1: $2'),
        (r'self\.', r'this.'),
        (r'super\(\)\.(\w+)', r'super.$1'),
        (r'if\s+(.*?):', r'if ($1) {'),
        (r'elif\s+(.*?):', r'} else if ($1) {'),
        (r'else:', r'} else {'),
        (r'for\s+(\w+)\s+in\s+range\((\w+)\):', r'for (let $1 = 0; $1 < $2; $1++) {'),
        (r'for\s+(\w+)\s+in\s+range\((\w+),\s*(\w+)\):', r'for (let $1 = $2; $1 < $3; $1++) {'),
        (r'for\s+(\w+)\s+in\s+range\((\w+),\s*(\w+),\s*(\w+)\):', r'for (let $1 = $2; $1 < $3; $1 += $4) {'),
        (r'for\s+(\w+)\s+in\s+(\w+):', r'for (const $1 of $2) {'),
        (r'while\s+(.*?):', r'while ($1) {'),
        (r'try:', r'try {'),
        (r'except\s+(\w+)(?:\s+as\s+(\w+))?:', r'} catch($2: $1) {'),
        (r'except:', r'} catch(error) {'),
        (r'finally:', r'} finally {'),
        (r'(\w+)\.append\((.*?)\)', r'$1.push($2)'),
        (r'\[(.*?) for (.*?) in (.*?)\]', r'$3.map(($2) => $1)'),
        (r'\[(.*?) for (.*?) in (.*?) if (.*?)\]', r'$3.filter(($2) => $4).map(($2) => $1)'),
        (r'(\w+)\.items\(\)', r'Object.entries($1)'),
        (r'(\w+)\.keys\(\)', r'Object.keys($1)'),
        (r'(\w+)\.values\(\)', r'Object.values($1)'),
        (r'(\w+)\.get\((.*?), (.*?)\)', r'$1[$2] ?? $3'),
        (r'(\w+)\.get\((.*?)\)', r'$1[$2]'),
        (r' and ', r' && '),
        (r' or ', r' || '),
        (r'not ', r'!'),
        (r'None', r'null'),
        (r'True', r'true'),
        (r'False', r'false'),
        (r'f[\'"](.+?)[\'"]', r'`$1`'),
        (r'{([^{}]+?)}', r'${$1}'),
        (r'#\s*(.*?)$', r'// $1'),
        (r'print\((.*?)\)', r'console.log($1)'),
        (r'async\s+def', r'async'),
        (r'await\s+', r'await '),
        (r'return\s+(.*)', r'return $1;'),
        (r'return;', r'return;'),
        (r'assert\s+(.*?)(,\s*[\'"](.+)[\'"])?', r'if (!($1)) { throw new Error($3 || "Assertion failed"); }'),
        (r'navigator\.gpu\.request_adapter', r'navigator.gpu.requestAdapter'),
        (r'create_buffer', r'createBuffer'),
        (r'create_compute_pipeline', r'createComputePipeline'),
        (r'create_shader_module', r'createShaderModule'),
        (r'create_bind_group', r'createBindGroup'),
        (r'create_command_encoder', r'createCommandEncoder'),
        (r'begin_compute_pass', r'beginComputePass'),
        (r'set_pipeline', r'setPipeline'),
        (r'set_bind_group', r'setBindGroup'),
        (r'dispatch_workgroups', r'dispatchWorkgroups'),
        (r'submit_command_buffer', r'submitCommandBuffer'),
        (r'navigator\.ml\.create_context', r'navigator.ml.createContext'),
        (r'create_graph_builder', r'createGraphBuilder'),
        (r'create_graph', r'createGraph'),
        (r'create_model', r'createModel'),
        (r'build_graph', r'buildGraph'),
        (r'create_operand', r'createOperand'),
        (r'str\((.*?)\)', r'String($1)'),
        (r'len\((.*?)\)', r'$1.length'),
        (r'isinstance\((.*?), (.*?)\)', r'$1 instanceof $2'),
        (r'(\w+)\.split\((.*?)\)', r'$1.split($2)'),
        (r'(\w+)\.strip\(\)', r'$1.trim()'),
        (r'(\w+)\.lower\(\)', r'$1.toLowerCase()'),
        (r'(\w+)\.upper\(\)', r'$1.toUpperCase()'),
        (r'(\w+)\.startswith\((.*?)\)', r'$1.startsWith($2)'),
        (r'(\w+)\.endswith\((.*?)\)', r'$1.endsWith($2)'),
        (r'(\w+)\.replace\((.*?), (.*?)\)', r'$1.replace($2, $3)'),
        (r'(\w+)\.join\((.*?)\)', r'$2.join($1)'),
    ]', r'$1: $2[]'),
        (r'(\w+):\s*Dict\[(\w+),\s*(\w+)\]', r'$1: Record<$2, $3>'),
        (r'(\w+):\s*Optional\[(\w+)\]', r'$1: $2 | null'),
        (r'(\w+):\s*Union\[([^\]]+)\]', r'$1: $2'),
        
        # Function definitions
        (r'def\s+(\w+)\s*\((.*?)\)\s*->\s*(\w+):', r'$1($2): $3 {'),
        (r'def\s+(\w+)\s*\((.*?)\):', r'$1($2) {'),
        (r'self\.', r'this.'),
        
        # Control structures
        (r'if\s+(.*?):', r'if ($1) {'),
        (r'elif\s+(.*?):', r'} else if ($1) {'),
        (r'else:', r'} else {'),
        (r'for\s+(\w+)\s+in\s+range\((\w+)\):', r'for (let $1 = 0; $1 < $2; $1++) {'),
        (r'for\s+(\w+)\s+in\s+(\w+):', r'for (const $1 of $2) {'),
        (r'while\s+(.*?):', r'while ($1) {'),
        (r'try:', r'try {'),
        (r'except\s+(\w+)(?:\s+as\s+(\w+))?:', r'} catch($2: $1) {'),
        (r'except:', r'} catch(error) {'),
        (r'finally:', r'} finally {'),
        
        # List operations
        (r'(\w+)\.append\((.*?)\)', r'$1.push($2)'),
        (r'\[(.*?) for (.*?) in (.*?)\]', r'$3.map(($2) => $1)'),
        
        # Dictionary operations
        (r'(\w+)\.items\(\)', r'Object.entries($1)'),
        (r'(\w+)\.keys\(\)', r'Object.keys($1)'),
        (r'(\w+)\.values\(\)', r'Object.values($1)'),
        
        # Boolean operators
        (r' and ', r' && '),
        (r' or ', r' || '),
        (r'not ', r'!'),
        
        # None/null
        (r'None', r'null'),
        (r'True', r'true'),
        (r'False', r'false'),
        
        # f-strings
        (r'f[\'"](.+?)[\'"]', r'`$1`'),
        (r'{([^{}]+?)}', r'${$1}'),
        
        # Comments
        (r'#\s*(.*?)$', r'// $1'),
        
        # Print statements
        (r'print\((.*?)\)', r'console.log($1)'),
        
        # Async/await
        (r'async\s+def', r'async'),
        (r'await\s+', r'await '),
        
        # WebGPU specific conversions
        (r'navigator\.gpu\.request_adapter', r'navigator.gpu.requestAdapter'),
        (r'request_device', r'requestDevice'),
        (r'create_buffer', r'createBuffer'),
        (r'create_compute_pipeline', r'createComputePipeline'),
        (r'create_shader_module', r'createShaderModule'),
        (r'set_pipeline', r'setPipeline'),
        (r'set_bind_group', r'setBindGroup'),
        (r'dispatch_workgroups', r'dispatchWorkgroups'),
        
        # WebNN specific conversions
        (r'navigator\.ml', r'navigator.ml'),
        (r'create_context', r'createContext'),
        (r'create_graph', r'createGraph'),
        (r'create_model', r'createModel'),
        (r'build_graph', r'buildGraph'),
    ]
    
    # WebGPU/WebNN specific class conversions
    CLASS_CONVERSIONS = {
        'WebGPUBackend': {
            'signature': 'class WebGPUBackend implements HardwareBackend',
            'methods': {
                'initialize': 'async initialize(): Promise<boolean> {\n    try {\n      // Request adapter from navigator.gpu\n      this.adapter = await navigator.gpu.requestAdapter();\n      \n      if (!this.adapter) {\n        console.error("WebGPU not supported or disabled");\n        return false;\n      }\n      \n      // Request device from adapter\n      this.device = await this.adapter.requestDevice();\n      \n      if (!this.device) {\n        console.error("Failed to get WebGPU device");\n        return false;\n      }\n      \n      this.initialized = true;\n      return true;\n    } catch (error) {\n      console.error("WebGPU initialization error:", error);\n      return false;\n    }\n  }',
                'createBuffer': 'createBuffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer | null {\n    if (!this.device) {\n      console.error("WebGPU device not initialized");\n      return null;\n    }\n    \n    try {\n      return this.device.createBuffer({\n        size,\n        usage,\n        mappedAtCreation: false\n      });\n    } catch (error) {\n      console.error("Error creating WebGPU buffer:", error);\n      return null;\n    }\n  }',
                'createComputePipeline': 'async createComputePipeline(shader: string): Promise<GPUComputePipeline | null> {\n    if (!this.device) {\n      console.error("WebGPU device not initialized");\n      return null;\n    }\n    \n    try {\n      const shaderModule = this.device.createShaderModule({\n        code: shader\n      });\n      \n      return this.device.createComputePipeline({\n        layout: \'auto\',\n        compute: {\n          module: shaderModule,\n          entryPoint: \'main\'\n        }\n      });\n    } catch (error) {\n      console.error("Error creating compute pipeline:", error);\n      return null;\n    }\n  }',
                'runCompute': 'async runCompute(\n    pipeline: GPUComputePipeline,\n    bindings: GPUBindGroup[],\n    workgroups: [number, number?, number?]\n  ): Promise<void> {\n    if (!this.device) {\n      throw new Error("WebGPU device not initialized");\n    }\n    \n    try {\n      const commandEncoder = this.device.createCommandEncoder();\n      const passEncoder = commandEncoder.beginComputePass();\n      \n      passEncoder.setPipeline(pipeline);\n      \n      for (let i = 0; i < bindings.length; i++) {\n        passEncoder.setBindGroup(i, bindings[i]);\n      }\n      \n      const [x, y = 1, z = 1] = workgroups;\n      passEncoder.dispatchWorkgroups(x, y, z);\n      passEncoder.end();\n      \n      const commandBuffer = commandEncoder.finish();\n      this.device.queue.submit([commandBuffer]);\n      \n      // Wait for GPU to complete\n      await this.device.queue.onSubmittedWorkDone();\n    } catch (error) {\n      console.error("Error running compute operation:", error);\n      throw error;\n    }\n  }',
                'destroy': 'destroy(): void {\n    if (this.adapter) {\n      // Clean up any resources\n      this.adapter = null;\n    }\n    \n    if (this.device) {\n      // Clean up device\n      this.device = null;\n    }\n    \n    this.initialized = false;\n  }',
            },
            'properties': {
                'device': 'device: GPUDevice | null = null',
                'adapter': 'adapter: GPUAdapter | null = null',
                'initialized': 'initialized: boolean = false',
                'features': 'features: Set<string> = new Set()',
                'limits': 'limits: Record<string, number> = {}',
            }
        },
        'WebNNBackend': {
            'signature': 'class WebNNBackend implements HardwareBackend',
            'methods': {
                'initialize': 'async initialize(): Promise<boolean> {\n    try {\n      if (!navigator.ml) {\n        console.error("WebNN not supported in this browser");\n        return false;\n      }\n      \n      this.context = navigator.ml.createContext();\n      \n      if (!this.context) {\n        console.error("Failed to create WebNN context");\n        return false;\n      }\n      \n      this.initialized = true;\n      return true;\n    } catch (error) {\n      console.error("WebNN initialization error:", error);\n      return false;\n    }\n  }',
                'createGraphBuilder': 'createGraphBuilder(): MLGraphBuilder | null {\n    if (!this.context) {\n      console.error("WebNN context not initialized");\n      return null;\n    }\n    \n    try {\n      return new MLGraphBuilder(this.context);\n    } catch (error) {\n      console.error("Error creating graph builder:", error);\n      return null;\n    }\n  }',
                'buildGraph': 'async buildGraph(\n    graphBuilder: MLGraphBuilder,\n    outputs: Record<string, MLOperand>\n  ): Promise<MLGraph | null> {\n    if (!this.initialized) {\n      console.error("WebNN not initialized");\n      return null;\n    }\n    \n    try {\n      return await graphBuilder.build(outputs);\n    } catch (error) {\n      console.error("Error building graph:", error);\n      return null;\n    }\n  }',
                'runInference': 'async runInference(\n    graph: MLGraph,\n    inputs: Record<string, MLOperand>\n  ): Promise<Record<string, MLOperand>> {\n    if (!this.initialized) {\n      throw new Error("WebNN not initialized");\n    }\n    \n    try {\n      return graph.compute(inputs);\n    } catch (error) {\n      console.error("Error running inference:", error);\n      throw error;\n    }\n  }',
                'destroy': 'destroy(): void {\n    this.context = null;\n    this.initialized = false;\n  }',
            },
            'properties': {
                'context': 'context: MLContext | null = null',
                'initialized': 'initialized: boolean = false',
                'capabilities': 'capabilities: string[] = []',
            }
        },
        'HardwareAbstraction': {
            'signature': 'class HardwareAbstraction',
            'methods': {
                'initialize': 'async initialize(): Promise<boolean> {\n    try {\n      // Initialize hardware detection\n      const hardwareDetection = new HardwareDetection();\n      const capabilities = await hardwareDetection.detectCapabilities();\n      \n      // Initialize backends based on available hardware\n      if (capabilities.webgpu) {\n        const webgpuBackend = new WebGPUBackend();\n        const success = await webgpuBackend.initialize();\n        if (success) {\n          this.backends.set(\'webgpu\', webgpuBackend);\n        }\n      }\n      \n      if (capabilities.webnn) {\n        const webnnBackend = new WebNNBackend();\n        const success = await webnnBackend.initialize();\n        if (success) {\n          this.backends.set(\'webnn\', webnnBackend);\n        }\n      }\n      \n      // Always add CPU backend as fallback\n      const cpuBackend = new CPUBackend();\n      await cpuBackend.initialize();\n      this.backends.set(\'cpu\', cpuBackend);\n      \n      // Apply hardware preferences\n      this.applyPreferences();\n      \n      return this.backends.size > 0;\n    } catch (error) {\n      console.error("Error initializing hardware abstraction:", error);\n      return false;\n    }\n  }',
                'getBestBackend': 'getBestBackend(modelType: string): HardwareBackend {\n    // Check if we have a preference for this model type\n    if (\n      this.preferences &&\n      this.preferences.modelPreferences &&\n      this.preferences.modelPreferences[modelType]\n    ) {\n      const preferredBackends = this.preferences.modelPreferences[modelType];\n      \n      // Try each preferred backend in order\n      for (const backendName of preferredBackends) {\n        if (this.backends.has(backendName)) {\n          return this.backends.get(backendName)!;\n        }\n      }\n    }\n    \n    // Fallback to order of preference: WebGPU > WebNN > CPU\n    if (this.backends.has(\'webgpu\')) return this.backends.get(\'webgpu\')!;\n    if (this.backends.has(\'webnn\')) return this.backends.get(\'webnn\')!;\n    \n    // Always have CPU as fallback\n    return this.backends.get(\'cpu\')!;\n  }',
                'runModel': 'async runModel<T = any, U = any>(model: Model, inputs: T): Promise<U> {\n    const backend = this.getBestBackend(model.type);\n    return model.execute(inputs, backend);\n  }',
                'destroy': 'destroy(): void {\n    // Destroy all backends\n    for (const backend of this.backends.values()) {\n      backend.destroy();\n    }\n    \n    this.backends.clear();\n  }',
                'applyPreferences': 'private applyPreferences(): void {\n    // Apply any hardware preferences from configuration\n    if (this.preferences && this.preferences.backendOrder) {\n      // Reorder backends based on preferences\n      this.backendOrder = this.preferences.backendOrder.filter(\n        backend => this.backends.has(backend)\n      );\n    } else {\n      // Default order: WebGPU > WebNN > CPU\n      this.backendOrder = [\'webgpu\', \'webnn\', \'cpu\'].filter(\n        backend => this.backends.has(backend)\n      );\n    }\n  }',
            },
            'properties': {
                'backends': 'backends: Map<string, HardwareBackend> = new Map()',
                'preferences': 'preferences: HardwarePreferences',
                'backendOrder': 'backendOrder: string[] = []',
            }
        },
    },
            'properties': {
                'device': 'device: GPUDevice | null = null',
                'adapter': 'adapter: GPUAdapter | null = null',
                'initialized': 'initialized: boolean = false'
            }
        },
        'WebNNBackend': {
            'signature': 'class WebNNBackend implements HardwareBackend',
            'methods': {
                'initialize': 'async initialize(): Promise<boolean>',
                'createContext': 'createContext(): MLContext',
                'buildGraph': 'buildGraph(graphBuilder: MLGraphBuilder): MLGraph',
                'runInference': 'async runInference(graph: MLGraph, inputs: Record<string, MLOperand>): Promise<Record<string, MLOperand>>',
                'destroy': 'destroy(): void'
            },
            'properties': {
                'context': 'context: MLContext | null = null',
                'initialized': 'initialized: boolean = false'
            }
        },
        'HardwareAbstraction': {
            'signature': 'class HardwareAbstraction',
            'methods': {
                'initialize': 'async initialize(): Promise<boolean>',
                'getBestBackend': 'getBestBackend(modelType: string): HardwareBackend',
                'runModel': 'async runModel(model: Model, inputs: any): Promise<any>',
                'destroy': 'destroy(): void'
            },
            'properties': {
                'backends': 'backends: Map<string, HardwareBackend> = new Map()',
                'preferences': 'preferences: HardwarePreferences'
            }
        }
    }
    
    @staticmethod
    def convert_py_to_ts(content: str, filename: str) -> str:
        """Convert Python code to TypeScript"""
        # First, check if this is a known class that has a special conversion template
        class_match = re.search(r'class\s+(\w+)', content)
        if class_match:
            class_name = class_match.group(1)
            if class_name in PyToTsConverter.CLASS_CONVERSIONS:
                logger.info(f"Using specialized conversion template for {class_name}")
                return PyToTsConverter._generate_class_from_template(class_name, content)
        
        # Regular expression-based conversion for other files
        result = content
        
        # Clean up indentation (4 spaces to 2 spaces)
        lines = result.split('\n')
        for i, line in enumerate(lines):
            indent_match = re.match(r'^(\s+)', line)
            if indent_match:
                indent = indent_match.group(1)
                # Convert 4-space or tab indentation to 2-space
                if '\t' in indent:
                    # Replace tabs with 2 spaces
                    lines[i] = line.replace('\t', '  ')
                else:
                    # Calculate how many 2-space indents we need
                    indent_level = len(indent) // 4
                    lines[i] = '  ' * indent_level + line.lstrip()
        
        result = '\n'.join(lines)
        
        # Apply all pattern conversions
        for pattern, replacement in PyToTsConverter.PATTERN_MAP:
            result = re.sub(pattern, replacement, result)
        
        # Add closing braces for blocks (a bit more complex)
        result = PyToTsConverter._add_closing_braces(result)
        
        # Add TypeScript header
        header = '/**\n'
        header += f' * Converted from Python: {os.path.basename(filename)}\n'
        header += f' * Conversion date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        header += ' * This file was automatically converted from Python to TypeScript.\n'
        header += ' * Conversion fidelity might not be 100%, please manual review recommended.\n'
        header += ' */\n\n'
        
        # Add TypeScript interfaces
        interfaces = PyToTsConverter._extract_interfaces(content)
        
        # If this is a WebGPU or WebNN file, add appropriate imports
        imports = ""
        if "webgpu" in filename.lower() or "gpu" in content.lower():
            imports += '// WebGPU related imports\n'
            imports += 'import { HardwareBackend } from "../hardware_abstraction";\n\n'
            Config.MIGRATION_STATS["webgpu_files"] += 1
        elif "webnn" in filename.lower() or "ml" in content.lower():
            imports += '// WebNN related imports\n'
            imports += 'import { HardwareBackend } from "../hardware_abstraction";\n\n'
            Config.MIGRATION_STATS["webnn_files"] += 1
        
        return header + imports + interfaces + result
    
    @staticmethod
        @staticmethod
    def _extract_interfaces(content: str) -> str:
        """Extract and convert Python type annotations to TypeScript interfaces"""
        interfaces = ""
        
        # Add standard interfaces for common types
        interfaces += """interface HardwareBackend {
  initialize(): Promise<boolean>;
  destroy(): void;
}

interface HardwarePreferences {
  backendOrder?: string[];
  modelPreferences?: Record<string, string[]>;
  options?: Record<string, any>;
}

interface ModelConfig {
  id: string;
  type: string;
  path?: string;
  options?: Record<string, any>;
}

interface Model {
  id: string;
  type: string;
  execute<T = any, U = any>(inputs: T, backend: HardwareBackend): Promise<U>;
}

"""
        
        # Look for type definitions at module level
        type_defs = re.findall(r'(\w+)\s*=\s*(?:Dict|List|Tuple|Optional|Union)\[([^\]]+)\]', content)
        for name, types in type_defs:
            # Convert Python types to TypeScript
            ts_types = types.replace("str", "string").replace("int", "number").replace("float", "number")
            ts_types = ts_types.replace("bool", "boolean").replace("Any", "any")
            ts_types = re.sub(r'List\[([^\]]+)\]', r'$1[]', ts_types)
            ts_types = re.sub(r'Dict\[([^,]+),\s*([^\]]+)\]', r'Record<$1, $2>', ts_types)
            ts_types = re.sub(r'Optional\[([^\]]+)\]', r'$1 | null', ts_types)
            ts_types = re.sub(r'Tuple\[([^\]]+)\]', r'[$1]', ts_types)
            
            interfaces += f'export type {name} = {ts_types};\n'
        
        # Look for class properties with type hints
        class_props = re.findall(r'self\.(\w+)(?:\s*:\s*([^=\n]+))?(?:\s*=\s*([^#\n]+))?', content)
        interface_name = ""
        
        # Try to extract class name for better interface naming
        class_match = re.search(r'class\s+(\w+)', content)
        if class_match:
            interface_name = f"{class_match.group(1)}Props"
        else:
            interface_name = "Props"
        
        if class_props:
            interfaces += f'\nexport interface {interface_name} {\n'
            
            for prop_match in class_props:
                prop_name = prop_match[0]
                prop_type = prop_match[1] if len(prop_match) > 1 and prop_match[1] else "any"
                
                # Convert Python type to TypeScript
                ts_type = prop_type.strip()
                ts_type = re.sub(r'str', 'string', ts_type)
                ts_type = re.sub(r'int|float', 'number', ts_type)
                ts_type = re.sub(r'bool', 'boolean', ts_type)
                ts_type = re.sub(r'List\[([\w]+)\]', r'$1[]', ts_type)
                ts_type = re.sub(r'Dict\[([\w]+),\s*([\w]+)\]', r'Record<$1, $2>', ts_type)
                ts_type = re.sub(r'Optional\[([\w]+)\]', r'$1 | null', ts_type)
                ts_type = re.sub(r'Any', 'any', ts_type)
                
                interfaces += f'  {prop_name}: {ts_type};\n'
            interfaces += '}\n\n'
        
        # Look for method parameters with type hints
        methods = re.finditer(r'def\s+(\w+)\s*\(self(?:,\s*([^)]+))?\)(?:\s*->\s*([^:]+))?:', content)
        for method in methods:
            method_name = method.group(1)
            params = method.group(2)
            return_type = method.group(3)
            
            if params:
                param_interface = f'\ninterface {method_name.capitalize()}Params {\n'
                has_params = False
                
                for param in params.split(','):
                    param = param.strip()
                    if ':' in param:
                        has_params = True
                        param_name, param_type = param.split(':', 1)
                        param_name = param_name.strip()
                        param_type = param_type.strip()
                        
                        # Convert Python type to TypeScript
                        ts_type = re.sub(r'str', 'string', param_type)
                        ts_type = re.sub(r'int|float', 'number', ts_type)
                        ts_type = re.sub(r'bool', 'boolean', ts_type)
                        ts_type = re.sub(r'List\[([\w]+)\]', r'$1[]', ts_type)
                        ts_type = re.sub(r'Dict\[([\w]+),\s*([\w]+)\]', r'Record<$1, $2>', ts_type)
                        ts_type = re.sub(r'Optional\[([\w]+)\]', r'$1 | null', ts_type)
                        ts_type = re.sub(r'Any', 'any', ts_type)
                        
                        # Handle default values
                        if '=' in param_name:
                            param_parts = param_name.split('=')
                            param_name = param_parts[0].strip()
                            param_interface += f'  {param_name}?: {ts_type};\n'
                        else:
                            param_interface += f'  {param_name}: {ts_type};\n'
                
                if has_params:
                    param_interface += '}\n'
                    interfaces += param_interface
        
        return interfaces
    
    @staticmethod
        @staticmethod
    def _add_closing_braces(content: str) -> str:
        """Add closing braces to match the opening ones with improved handling of nested blocks"""
        lines = content.split('\n')
        result_lines = []
        stack = []
        indent_stack = []
        
        for i, line in enumerate(lines):
            # Track current indentation level
            current_indent = len(re.match(r'^\s*', line).group(0))
            
            # Look for opening braces at the end of a line
            opening_braces = len(re.findall(r'{\s*$', line))
            if opening_braces > 0:
                for _ in range(opening_braces):
                    stack.append(i)
                    indent_stack.append(current_indent)
                result_lines.append(line)
            # Check if next line already has a closing brace that matches
            elif i < len(lines) - 1 and re.match(r'\s*}', lines[i+1]):
                # Next line already has a closing brace, no need to add one
                result_lines.append(line)
            # Check if this is the end of a block by checking indentation
            elif stack and (
                i == len(lines) - 1 or  # Last line
                len(line.strip()) == 0 or  # Empty line
                (current_indent <= indent_stack[-1])  # Outdent
            ):
                # End of a block, add closing brace after the current line
                result_lines.append(line)
                
                # Check how many blocks are ending
                while stack and (i == len(lines) - 1 or current_indent <= indent_stack[-1]):
                    # Get the indentation of the opening brace
                    indent = ' ' * indent_stack[-1]
                    stack.pop()
                    indent_stack.pop()
                    
                    # Add closing brace with the same indentation as the opening
                    if i < len(lines) - 1:  # Not the last line
                        result_lines.append(f"{indent}}}")
            else:
                result_lines.append(line)
        
        # Add any remaining closing braces at the end
        while stack:
            indent = ' ' * indent_stack[-1]
            stack.pop()
            indent_stack.pop()
            result_lines.append(f"{indent}}}")
        
        # Join lines back together
        content = '\n'.join(result_lines)
        
        # Fix common brace issues
        
        # Fix duplicate closing braces
        content = re.sub(r'}\s*}([^}])', r'}\1', content)
        
        # Fix duplicate opening braces
        content = re.sub(r'([^{]){(\s*){', r'\1{\2', content)
        
        # Fix spacing around braces
        content = re.sub(r'{\s*{', r'{', content)
        content = re.sub(r'}\s*}', r'}', content)
        
        # Fix semicolons after braces
        content = re.sub(r'}\s*;', r'}', content)
        
        # Ensure semicolons at the end of statements
        content = re.sub(r'(\w+\s*=\s*[^;{\n]+)\n', r'\1;\n', content)
        
        # Fix missing semicolons after return statements
        content = re.sub(r'return\s+([^;{\n]+)\n', r'return \1;\n', content)
        
        # Fix dangling else (else without preceding if)
        content = re.sub(r'\n(\s*)} else', r'\n\1}\n\1else', content)
        
        # Fix missing braces for single-line if statements
        content = re.sub(r'if\s*\(([^)]+)\)\s*([^{\n][^;\n]+);', r'if (\1) {\n  \2;\n}', content)
        
        # Fix array destructuring syntax that TypeScript doesn't support well
        content = re.sub(r'const\s*\[([^=]+)\]\s*=\s*([^;]+);', r'const _tmp = \2;\nconst \1 = _tmp;', content)
        
        return content
    
    @staticmethod
        @staticmethod
    def _generate_class_from_template(class_name: str, content: str) -> str:
        """Generate TypeScript class from predefined template with enhanced typing"""
        template = PyToTsConverter.CLASS_CONVERSIONS[class_name]
        
        # Add extracted interfaces from content
        interfaces = ""
        class_props = re.findall(r'self\.(\w+)(?:\s*:\s*([^=
]+))?(?:\s*=\s*([^#
]+))?', content)
        if class_props:
            prop_interface = f"interface {class_name}Props {\n"
            for prop_match in class_props:
                prop_name = prop_match[0]
                prop_type = prop_match[1] if len(prop_match) > 1 and prop_match[1] else "any"
                
                # Convert Python type to TypeScript
                ts_type = prop_type.strip()
                ts_type = re.sub(r'str', 'string', ts_type)
                ts_type = re.sub(r'int|float', 'number', ts_type)
                ts_type = re.sub(r'bool', 'boolean', ts_type)
                ts_type = re.sub(r'List\[([\w]+)\]', r'$1[]', ts_type)
                ts_type = re.sub(r'Dict\[([\w]+),\s*([\w]+)\]', r'Record<$1, $2>', ts_type)
                ts_type = re.sub(r'Optional\[([\w]+)\]', r'$1 | null', ts_type)
                ts_type = re.sub(r'Any', 'any', ts_type)
                
                prop_interface += f"  {prop_name}: {ts_type};\n"
            prop_interface += "}\n\n"
            interfaces += prop_interface
        
        # Generate class definition
        result = interfaces + f"{template['signature']} {\n"
        
        # Add properties with TypeScript types
        for prop_name, prop_def in template['properties'].items():
            result += f"  {prop_def};\n"
        
        result += "\n"
        
        # Add constructor with proper parameter types
        constructor_match = re.search(r'def __init__\s*\(self(?:,\s*([^)]+))?\):', content)
        params = ""
        if constructor_match and constructor_match.group(1):
            # Extract parameters and convert to TypeScript
            py_params = constructor_match.group(1).split(',')
            ts_params = []
            
            for param in py_params:
                param = param.strip()
                if ':' in param:
                    # Has type annotation
                    param_name, param_type = param.split(':', 1)
                    param_name = param_name.strip()
                    param_type = param_type.strip()
                    
                    # Convert Python type to TypeScript
                    param_type = re.sub(r'str', 'string', param_type)
                    param_type = re.sub(r'int|float', 'number', param_type)
                    param_type = re.sub(r'bool', 'boolean', param_type)
                    param_type = re.sub(r'List\[([\w]+)\]', r'$1[]', param_type)
                    param_type = re.sub(r'Dict\[([\w]+),\s*([\w]+)\]', r'Record<$1, $2>', param_type)
                    param_type = re.sub(r'Optional\[([\w]+)\]', r'$1 | null', param_type)
                    param_type = re.sub(r'Any', 'any', param_type)
                    
                    # Check for default value
                    if '=' in param_name:
                        param_parts = param_name.split('=', 1)
                        param_name = param_parts[0].strip()
                        default_value = param_parts[1].strip()
                        
                        # Convert Python default values to TypeScript
                        default_value = re.sub(r'None', 'null', default_value)
                        default_value = re.sub(r'True', 'true', default_value)
                        default_value = re.sub(r'False', 'false', default_value)
                        
                        ts_params.append(f"{param_name}: {param_type} = {default_value}")
                    else:
                        ts_params.append(f"{param_name}: {param_type}")
                elif '=' in param:
                    # Has default value but no type
                    param_parts = param.split('=', 1)
                    param_name = param_parts[0].strip()
                    default_value = param_parts[1].strip()
                    
                    # Convert Python default values to TypeScript
                    default_value = re.sub(r'None', 'null', default_value)
                    default_value = re.sub(r'True', 'true', default_value)
                    default_value = re.sub(r'False', 'false', default_value)
                    
                    ts_params.append(f"{param_name}: any = {default_value}")
                else:
                    # No type or default
                    ts_params.append(f"{param.strip()}: any")
            
            params = ", ".join(ts_params)
        
        result += f"  constructor({params}) {\n"
        
        # Extract constructor body from Python __init__ method
        constructor_body = ""
        init_match = re.search(r'def __init__.*?:(.*?)(?=\s+(?:async\s+)?def|$)', content, re.DOTALL)
        if init_match:
            body = init_match.group(1).strip()
            # Convert Python constructor body to TypeScript
            body = re.sub(r'self\.', 'this.', body)
            # Add each line with proper indentation
            for line in body.split('\n'):
                if line.strip():
                    constructor_body += f"    {line.strip()}\n"
        
        # If no body extracted, initialize properties
        if not constructor_body:
            for prop_name, _ in template['properties'].items():
                if prop_name != 'initialized':  # Skip 'initialized' as it's set below
                    constructor_body += f"    this.{prop_name} = {prop_name};\n"
        
        # Always set initialized to false in constructor
        result += f"{constructor_body}    this.initialized = false;\n  }}\n\n"
        
        # Add methods
        for method_name, method_body in template['methods'].items():
            result += f"  {method_body}\n\n"
        
        result += "}\n"
        return result

# File finder and mapper
class FileFinder:
    @staticmethod
    def find_webgpu_webnn_files() -> List[str]:
        """Find all WebGPU/WebNN related files in source and fixed_web_platform directories"""
        all_files = []
        
        # Define patterns to search for
        patterns = [
            "webgpu", "gpu.requestAdapter", "GPUDevice", "GPUBuffer", "GPUCommandEncoder",
            "GPUShaderModule", "GPUComputePipeline", "webnn", "navigator.ml", "MLContext",
            "MLGraph", "MLGraphBuilder", "wgsl", "shader", "computeShader",
            "navigator.gpu", "createTexture", "createBuffer", "tensor", "tensorflow",
            "onnx", "WebWorker", "postMessage", "MessageEvent", "transferControlToOffscreen"
        ]
        
        # Helper function to check if path should be excluded
        def should_exclude(path):
            for exclude_dir in Config.EXCLUDE_DIRS:
                if exclude_dir in path:
                    return True
            return False
        
        # Search in source directory
        logger.info(f"Searching in {Config.SOURCE_DIR} for WebGPU/WebNN files...")
        for root, dirs, files in os.walk(Config.SOURCE_DIR):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in Config.EXCLUDE_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip if in excluded directory
                if should_exclude(file_path):
                    continue
                
                # Get file type
                file_type = FileTypes.detect_file_type(file_path)
                if file_type in [FileTypes.PYTHON, FileTypes.TYPESCRIPT, FileTypes.JAVASCRIPT, FileTypes.WGSL]:
                    # Check if file contains WebGPU/WebNN patterns
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            for pattern in patterns:
                                if pattern.lower() in content.lower():
                                    all_files.append(file_path)
                                    break
                    except Exception as e:
                        if Config.ENABLE_VERBOSE:
                            logger.debug(f"Error reading {file_path}: {e}")
        
        # Search in fixed_web_platform directory
        if Config.FIXED_WEB_PLATFORM_DIR and os.path.isdir(Config.FIXED_WEB_PLATFORM_DIR):
            logger.info(f"Searching in {Config.FIXED_WEB_PLATFORM_DIR} for WebGPU/WebNN files...")
            for root, dirs, files in os.walk(Config.FIXED_WEB_PLATFORM_DIR):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in Config.EXCLUDE_DIRS]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip if in excluded directory
                    if should_exclude(file_path):
                        continue
                    
                    # For fixed_web_platform, we're more selective about which files to include
                    if any(ext in file.lower() for ext in ['.py', '.ts', '.js', '.wgsl', '.html', '.jsx', '.tsx']):
                        all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} WebGPU/WebNN related files")
        return all_files
    
    @staticmethod
    def map_file_to_destination(file_path: str) -> str:
        """Map source file to appropriate destination in the target directory"""
        # Get file type
        file_type = FileTypes.detect_file_type(file_path)
        
        # Get basename of file
        basename = os.path.basename(file_path)
        _, src_ext = os.path.splitext(basename)
        output_ext = FileTypes.get_output_extension(file_type)
        
        # Get relative path if in fixed_web_platform directory
        if Config.FIXED_WEB_PLATFORM_DIR and file_path.startswith(Config.FIXED_WEB_PLATFORM_DIR):
            rel_path = os.path.relpath(file_path, Config.FIXED_WEB_PLATFORM_DIR)
            
            # Map fixed_web_platform subdirectories to target subdirectories
            if rel_path.startswith('unified_framework'):
                if 'webgpu_interface' in rel_path:
                    rel_path = 'src/hardware/backends/webgpu_interface'
                elif 'webnn_interface' in rel_path:
                    rel_path = 'src/hardware/backends/webnn_interface'
                elif 'hardware_abstraction' in rel_path:
                    rel_path = 'src/hardware/hardware_abstraction'
                elif 'model_sharding' in rel_path:
                    rel_path = 'src/model/model_sharding'
                else:
                    rel_path = rel_path.replace('unified_framework', 'src/hardware')
            elif rel_path.startswith('wgsl_shaders'):
                if '/firefox/' in rel_path:
                    rel_path = rel_path.replace('wgsl_shaders', 'src/worker/webgpu/shaders/firefox')
                elif '/chrome/' in rel_path:
                    rel_path = rel_path.replace('wgsl_shaders', 'src/worker/webgpu/shaders/chrome')
                elif '/safari/' in rel_path:
                    rel_path = rel_path.replace('wgsl_shaders', 'src/worker/webgpu/shaders/safari')
                elif '/edge/' in rel_path:
                    rel_path = rel_path.replace('wgsl_shaders', 'src/worker/webgpu/shaders/edge')
                else:
                    rel_path = rel_path.replace('wgsl_shaders', 'src/worker/webgpu/shaders/model_specific')
            elif rel_path.startswith('worker'):
                rel_path = 'src/' + rel_path
            else:
                rel_path = os.path.join('src', rel_path)
            
            # Determine output extension
            _, src_ext = os.path.splitext(file_path)
            output_ext = FileTypes.get_output_extension(file_type)
            
            # If it's a Python file, convert to TypeScript
            if src_ext.lower() == '.py':
                rel_path = os.path.splitext(rel_path)[0] + output_ext
            
            return os.path.join(Config.TARGET_DIR, rel_path)
        
        # Enhanced intelligent mapping based on filename patterns and content
        # WebGPU/WebNN related files
        if "webgpu_backend" in basename.lower() or "gpu_backend" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/hardware/backends/webgpu_backend" + output_ext)
        elif "webnn_backend" in basename.lower() or "nn_backend" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/hardware/backends/webnn_backend" + output_ext)
        elif "hardware_abstraction" in basename.lower() or "hw_abstraction" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/hardware/hardware_abstraction" + output_ext)
        elif "hardware_detection" in basename.lower() or "hw_detection" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/hardware/detection/hardware_detection" + output_ext)
        elif "gpu_detection" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/hardware/detection/gpu_detection" + output_ext)
        elif "model_loader" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/model/model_loader" + output_ext)
        elif "quantization_engine" in basename.lower() or "quant_engine" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/quantization/quantization_engine" + output_ext)
        elif "quantization" in basename.lower() or "quant_" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/quantization/techniques", os.path.splitext(basename)[0] + output_ext)
        elif "ultra_low_precision" in basename.lower() or "ulp" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/quantization/techniques/ultra_low_precision" + output_ext)
        
        # Shader files
        elif basename.endswith(".wgsl"):
            if "firefox" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/firefox", basename)
            elif "chrome" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/chrome", basename)
            elif "safari" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/safari", basename)
            elif "edge" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/edge", basename)
            else:
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/model_specific", basename)
        
        # Example/Demo files
        elif "streaming" in basename.lower() and "webgpu" in basename.lower() and src_ext in [".jsx", ".tsx", ".js", ".html", ".css"]:
            return os.path.join(Config.TARGET_DIR, "examples/browser/streaming", basename)
        elif "demo" in basename.lower() and src_ext in [".jsx", ".tsx", ".js", ".html", ".css"]:
            return os.path.join(Config.TARGET_DIR, "examples/browser/basic", basename)
        elif "example" in basename.lower() and "react" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "examples/browser/react", basename)
        
        # Resource pool files
        elif "resource_pool" in basename.lower() or "resource_bridge" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/browser/resource_pool", os.path.splitext(basename)[0] + output_ext)
        
        # Tensor related files
        elif "tensor_sharing" in basename.lower() or "cross_model_tensor" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/tensor/tensor_sharing" + output_ext)
        elif "tensor" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/tensor", os.path.splitext(basename)[0] + output_ext)
        
        # Storage related files
        elif "storage" in basename.lower() or "indexeddb" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/storage/indexeddb", os.path.splitext(basename)[0] + output_ext)
        
        # React related files
        elif "react" in basename.lower() or "hooks" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/react", os.path.splitext(basename)[0] + output_ext)
        
        # Model specific files
        elif "bert" in basename.lower() or "t5" in basename.lower() or "gpt" in basename.lower() or "llama" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/model/transformers", os.path.splitext(basename)[0] + output_ext)
        elif "vit" in basename.lower() or "vision" in basename.lower() or "clip" in basename.lower() or "detr" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/model/vision", os.path.splitext(basename)[0] + output_ext)
        elif "whisper" in basename.lower() or "clap" in basename.lower() or "audio" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/model/audio", os.path.splitext(basename)[0] + output_ext)
        
        # Test files - move to test directory
        elif basename.startswith("test_"):
            if "webgpu" in basename.lower() or "webnn" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "test/browser", os.path.splitext(basename)[0] + output_ext)
            elif "resource_pool" in basename.lower() or "browser" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "test/browser", os.path.splitext(basename)[0] + output_ext)
            elif "performance" in basename.lower() or "benchmark" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "test/performance", os.path.splitext(basename)[0] + output_ext)
            else:
                return os.path.join(Config.TARGET_DIR, "test/unit", os.path.splitext(basename)[0] + output_ext)
        
        # Optimization related files
        elif "optimization" in basename.lower() or "optimize" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/optimization/techniques", os.path.splitext(basename)[0] + output_ext)
        elif "memory" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/optimization/memory", os.path.splitext(basename)[0] + output_ext)
        
        # Browser and utils related files
        elif "browser" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/browser/optimizations", os.path.splitext(basename)[0] + output_ext)
        elif basename.startswith("utils") or basename.startswith("helpers"):
            return os.path.join(Config.TARGET_DIR, "src/utils", os.path.splitext(basename)[0] + output_ext)
        
        # Template files
        elif "template" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/model/templates", os.path.splitext(basename)[0] + output_ext)
        
        # Configuration files
        elif "config" in basename.lower():
            if src_ext.lower() in [".json", ".yaml", ".yml", ".toml"]:
                return os.path.join(Config.TARGET_DIR, basename)
            else:
                return os.path.join(Config.TARGET_DIR, "src/utils", os.path.splitext(basename)[0] + output_ext)
        
        # Special file types        # Special file types
        elif src_ext.lower() in [".md", ".markdown"]:
            return os.path.join(Config.TARGET_DIR, "docs", basename)
        elif src_ext.lower() in [".json"]:
            return os.path.join(Config.TARGET_DIR, basename)
        elif src_ext.lower() in [".wgsl"]:
            # Handle WGSL files with browser-specific optimizations
            if "firefox" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/firefox", basename)
            elif "chrome" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/chrome", basename)
            elif "safari" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/safari", basename)
            elif "edge" in basename.lower():
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/edge", basename)
            else:
                return os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/model_specific", basename)
        
        # Handle TypeScript declaration files
        elif basename.endswith(".d.ts"):
            return os.path.join(Config.TARGET_DIR, "src/types", basename)
        
        # Special fixes for known files
        elif "webgpu-utils" in basename.lower() or "webgpu_utils" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/utils/browser/webgpu-utils" + output_ext)
        elif "webnn-utils" in basename.lower() or "webnn_utils" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/utils/browser/webnn-utils" + output_ext)
        
        # Improved inference for file destination based on content
        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000)  # Read first 10KB
                    
                    # Check content patterns to determine appropriate directory
                    if "class HardwareAbstraction" in content or "HardwareAbstraction" in content:
                        return os.path.join(Config.TARGET_DIR, "src/hardware/hardware_abstraction" + output_ext)
                    elif "class WebGPUBackend" in content or "WebGPUBackend" in content:
                        return os.path.join(Config.TARGET_DIR, "src/hardware/backends/webgpu_backend" + output_ext)
                    elif "class WebNNBackend" in content or "WebNNBackend" in content:
                        return os.path.join(Config.TARGET_DIR, "src/hardware/backends/webnn_backend" + output_ext)
                    elif "class CPUBackend" in content or "CPUBackend" in content:
                        return os.path.join(Config.TARGET_DIR, "src/hardware/backends/cpu_backend" + output_ext)
                    elif "class ResourcePool" in content or "ResourcePool" in content:
                        return os.path.join(Config.TARGET_DIR, "src/browser/resource_pool/resource_pool" + output_ext)
                    elif "class ModelLoader" in content or "ModelLoader" in content:
                        return os.path.join(Config.TARGET_DIR, "src/model/loaders/model_loader" + output_ext)
                    elif "class QuantizationEngine" in content or "quantization" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/quantization/quantization_engine" + output_ext)
                    elif "class TensorSharing" in content or "TensorSharing" in content:
                        return os.path.join(Config.TARGET_DIR, "src/tensor/tensor_sharing" + output_ext)
                    elif "webgpu" in content.lower() or "gpu.requestAdapter" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/hardware/backends/webgpu_backend" + output_ext)
                    elif "webnn" in content.lower() or "navigator.ml" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/hardware/backends/webnn_backend" + output_ext)
                    elif "bert" in content.lower() or "transformer" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/model/transformers", os.path.splitext(basename)[0] + output_ext)
                    elif "vit" in content.lower() or "vision" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/model/vision", os.path.splitext(basename)[0] + output_ext)
                    elif "whisper" in content.lower() or "clap" in content.lower() or "wav2vec" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/model/audio", os.path.splitext(basename)[0] + output_ext)
                    elif "react" in content.lower() or "component" in content.lower() or "hook" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/react", os.path.splitext(basename)[0] + output_ext)
                    elif "storage" in content.lower() or "indexeddb" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/storage/indexeddb", os.path.splitext(basename)[0] + output_ext)
                    elif "browser" in content.lower() and "detect" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/browser/detection", os.path.splitext(basename)[0] + output_ext)
                    elif "worker" in content.lower() or "offscreen" in content.lower():
                        return os.path.join(Config.TARGET_DIR, "src/worker", os.path.splitext(basename)[0] + output_ext)
                    elif "test" in basename.lower() or "test" in file_path.lower():
                        if "webgpu" in content.lower() or "webnn" in content.lower():
                            return os.path.join(Config.TARGET_DIR, "test/browser", os.path.splitext(basename)[0] + output_ext)
                        elif "unit" in file_path.lower():
                            return os.path.join(Config.TARGET_DIR, "test/unit", os.path.splitext(basename)[0] + output_ext)
                        elif "integration" in file_path.lower():
                            return os.path.join(Config.TARGET_DIR, "test/integration", os.path.splitext(basename)[0] + output_ext)
                        else:
                            return os.path.join(Config.TARGET_DIR, "test", os.path.splitext(basename)[0] + output_ext)
            except Exception as e:
                if Config.ENABLE_VERBOSE:
                    logger.warning(f"Could not analyze content for {file_path}: {e}")
            
            # Default case - place in utils directory
            return os.path.join(Config.TARGET_DIR, "src/utils", os.path.splitext(basename)[0] + output_ext)
        
        # Template files
        elif "template" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/model/templates", os.path.splitext(basename)[0] + output_ext)
        
        # Configuration files
        elif "config" in basename.lower():
            if src_ext.lower() in [".json", ".yaml", ".yml", ".toml"]:
                return os.path.join(Config.TARGET_DIR, basename)
            else:
                return os.path.join(Config.TARGET_DIR, "src/utils", os.path.splitext(basename)[0] + output_ext)
        
        # Special file types
        elif src_ext.lower() in [".md", ".markdown"]:
            return os.path.join(Config.TARGET_DIR, "docs", basename)
        elif src_ext.lower() in [".json"]:
            return os.path.join(Config.TARGET_DIR, basename)
        
        # Special fixes for known files
        elif "webgpu-utils" in basename.lower() or "webgpu_utils" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/utils/browser/webgpu-utils" + output_ext)
        elif "webnn-utils" in basename.lower() or "webnn_utils" in basename.lower():
            return os.path.join(Config.TARGET_DIR, "src/utils/browser/webnn-utils" + output_ext)
        
        # Default case - place in utils directory
        else:
            return os.path.join(Config.TARGET_DIR, "src/utils", os.path.splitext(basename)[0] + output_ext)

class FileProcessor:
    @staticmethod
    def process_file(source_path: str, destination_path: str) -> bool:
        """Process a file based on its type and convert if necessary"""
        # Skip if file already processed
        if source_path in Config.CONVERTED_FILES:
            return True
        
        # Get file type
        file_type = FileTypes.detect_file_type(source_path)
        
        # Create destination directory if needed
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        try:
            # Handle based on file type
            if file_type == FileTypes.PYTHON:
                logger.info(f"Converting Python file: {source_path} -> {destination_path}")
                Config.MIGRATION_STATS["files_processed"] += 1
                
                if not Config.DRY_RUN:
                    with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Convert Python to TypeScript
                    ts_content = PyToTsConverter.convert_py_to_ts(content, source_path)
                    
                    with open(destination_path, 'w', encoding='utf-8') as f:
                        f.write(ts_content)
                    
                    Config.MIGRATION_STATS["files_converted"] += 1
                    Config.CONVERTED_FILES.add(source_path)
                    return True
            elif file_type == FileTypes.WGSL:
                logger.info(f"Copying WGSL shader: {source_path} -> {destination_path}")
                Config.MIGRATION_STATS["files_processed"] += 1
                Config.MIGRATION_STATS["wgsl_shaders"] += 1
                
                if not Config.DRY_RUN:
                    shutil.copy2(source_path, destination_path)
                    Config.MIGRATION_STATS["copied_files"] += 1
                    Config.CONVERTED_FILES.add(source_path)
                    return True
            elif file_type in [FileTypes.TYPESCRIPT, FileTypes.JAVASCRIPT, FileTypes.HTML, FileTypes.CSS]:
                logger.info(f"Copying file: {source_path} -> {destination_path}")
                Config.MIGRATION_STATS["files_processed"] += 1
                
                if not Config.DRY_RUN:
                    # Copy file but fix import paths
                    with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Fix import paths
                    fixed_content = FileProcessor.fix_import_paths(content)
                    
                    with open(destination_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    Config.MIGRATION_STATS["copied_files"] += 1
                    Config.CONVERTED_FILES.add(source_path)
                    return True
            else:
                logger.warning(f"Unknown file type for {source_path}, skipping")
                return False
        except Exception as e:
            logger.error(f"Error processing {source_path}: {e}")
            Config.MIGRATION_STATS["conversion_failures"] += 1
            return False
        
        return True
    
    @staticmethod
    def fix_import_paths(content: str) -> str:
        """Fix import paths in TypeScript/JavaScript files"""
        # Fix relative imports
        fixed_content = re.sub(r'from\s+[\'"]\./(ipfs_accelerate_js_)?([^\'"]+)[\'"]', r'from ".\/\2"', content)
        fixed_content = re.sub(r'import\s+[\'"]\./(ipfs_accelerate_js_)?([^\'"]+)[\'"]', r'import ".\/\2"', fixed_content)
        fixed_content = re.sub(r'require\([\'"]\./(ipfs_accelerate_js_)?([^\'"]+)[\'"]\)', r'require(".\/\2")', fixed_content)
        
        return fixed_content
    
    @staticmethod
    def create_placeholder_for_empty_dirs():
        """Create placeholder files for empty directories"""
        logger.info("Creating placeholder files for empty directories...")
        
        if Config.DRY_RUN:
            logger.info("Dry run: Would create placeholder files in empty directories")
            return
        
        for root, dirs, files in os.walk(os.path.join(Config.TARGET_DIR, "src")):
            if not files and not any(os.path.isfile(os.path.join(root, d)) for d in dirs):
                # Empty directory, create placeholder
                dir_name = os.path.basename(root)
                placeholder_path = os.path.join(root, "index.ts")
                
                logger.info(f"Creating placeholder file: {placeholder_path}")
                
                # Generate placeholder content
                content = f"""/**
 * {dir_name} Module
 * 
 * This module provides functionality for {dir_name}.
 * Implementation pending as part of the WebGPU/WebNN migration.
 * 
 * @module {dir_name}
 */

/**
 * Configuration options for the {dir_name} module
 */
export interface {dir_name.title()}Options {{
  /**
   * Whether to enable debug mode
   */
  debug?: boolean;
  
  /**
   * Custom configuration settings
   */
  config?: Record<string, any>;
}}

/**
 * Main implementation class for the {dir_name} module
 */
export class {dir_name.title()}Manager {{
  private initialized = false;
  private options: {dir_name.title()}Options;
  
  /**
   * Creates a new {dir_name} manager
   * @param options Configuration options
   */
  constructor(options: {dir_name.title()}Options = {{}}) {{
    this.options = {{
      debug: false,
      ...options
    }};
  }}
  
  /**
   * Initializes the {dir_name} manager
   * @returns Promise that resolves when initialization is complete
   */
  async initialize(): Promise<boolean> {{
    // Implementation pending
    this.initialized = true;
    return true;
  }}
  
  /**
   * Checks if the manager is initialized
   */
  isInitialized(): boolean {{
    return this.initialized;
  }}
}}

// Default export
export default {dir_name.title()}Manager;
"""
                with open(placeholder_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                Config.MIGRATION_STATS["empty_files_created"] += 1

def create_base_project_files():
    """Create base project files (package.json, tsconfig.json, etc.)"""
    logger.info("Creating base project files...")
    
    if Config.DRY_RUN:
        logger.info("Dry run: Would create base project files")
        return
    
    # Create package.json
    package_json_path = os.path.join(Config.TARGET_DIR, "package.json")
    if not os.path.exists(package_json_path):
        logger.info(f"Creating {package_json_path}")
        
        package_json = {
            "name": "ipfs-accelerate",
            "version": "0.1.0",
            "description": "IPFS Accelerate JavaScript SDK for web browsers and Node.js",
            "main": "dist/ipfs-accelerate.js",
            "module": "dist/ipfs-accelerate.esm.js",
            "types": "dist/types/index.d.ts",
            "scripts": {
                "build": "rollup -c",
                "dev": "rollup -c -w",
                "test": "jest",
                "lint": "eslint 'src/**/*.{js,ts,tsx}'",
                "docs": "typedoc --out docs/api src/"
            },
            "repository": {
                "type": "git",
                "url": "git+https://github.com/your-org/ipfs-accelerate-js.git"
            },
            "keywords": [
                "webgpu",
                "webnn",
                "machine-learning",
                "ai",
                "hardware-acceleration",
                "browser"
            ],
            "author": "",
            "license": "MIT",
            "bugs": {
                "url": "https://github.com/your-org/ipfs-accelerate-js/issues"
            },
            "homepage": "https://github.com/your-org/ipfs-accelerate-js#readme",
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
                "rollup": "^2.66.1",
                "rollup-plugin-terser": "^7.0.2",
                "ts-jest": "^27.1.3",
                "tslib": "^2.3.1",
                "typedoc": "^0.22.11",
                "typescript": "^4.5.5"
            },
            "dependencies": {
                "comlink": "^4.3.1"
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
        
        with open(package_json_path, 'w', encoding='utf-8') as f:
            json.dump(package_json, f, indent=2)
    
    # Create tsconfig.json
    tsconfig_path = os.path.join(Config.TARGET_DIR, "tsconfig.json")
    if not os.path.exists(tsconfig_path):
        logger.info(f"Creating {tsconfig_path}")
        
        tsconfig = {
            "compilerOptions": {
                "target": "es2020",
                "module": "esnext",
                "moduleResolution": "node",
                "declaration": True,
                "declarationDir": "./dist/types",
                "sourceMap": True,
                "outDir": "./dist",
                "strict": True,
                "esModuleInterop": True,
                "noImplicitAny": True,
                "noImplicitThis": True,
                "strictNullChecks": True,
                "strictFunctionTypes": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True,
                "lib": ["dom", "dom.iterable", "esnext", "webworker"],
                "jsx": "react"
            },
            "include": ["src/**/*"],
            "exclude": ["node_modules", "dist", "examples", "**/*.test.ts"]
        }
        
        with open(tsconfig_path, 'w', encoding='utf-8') as f:
            json.dump(tsconfig, f, indent=2)
    
    # Create README.md
    readme_path = os.path.join(Config.TARGET_DIR, "README.md")
    if not os.path.exists(readme_path):
        logger.info(f"Creating {readme_path}")
        
        readme_content = """# IPFS Accelerate JavaScript SDK

> Hardware-accelerated machine learning for web browsers and Node.js

## Features

- **WebGPU Acceleration**: Utilize browser GPU capabilities for fast inference
- **WebNN Support**: Access neural network acceleration on supported browsers
- **Cross-Browser Compatibility**: Works on Chrome, Firefox, Safari, and Edge
- **React Integration**: Simple hooks for React applications
- **Ultra-Low Precision**: Support for 2-bit to 16-bit quantization
- **P2P Content Distribution**: IPFS-based model distribution
- **Cross-Environment**: Works in browsers and Node.js

## Installation

```bash
npm install ipfs-accelerate
```

## Quick Start

```javascript
import { createAccelerator } from 'ipfs-accelerate';

async function runInference() {
  // Create accelerator with automatic hardware detection
  const accelerator = await createAccelerator({
    autoDetectHardware: true
  });
  
  // Run inference
  const result = await accelerator.accelerate({
    modelId: 'bert-base-uncased',
    modelType: 'text',
    input: 'This is a sample text for embedding.'
  });
  
  console.log(result);
}

runInference();
```

## React Integration

```jsx
import { useAccelerator } from 'ipfs-accelerate/react';

function TextEmbeddingComponent() {
  const { model, loading, error } = useAccelerator({
    modelId: 'bert-base-uncased',
    modelType: 'text'
  });
  
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (model && input) {
      const embedding = await model.embed(input);
      setResult(embedding);
    }
  };
  
  return (
    <div>
      {loading && <p>Loading model...</p>}
      {error && <p>Error: {error.message}</p>}
      {model && (
        <form onSubmit={handleSubmit}>
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            placeholder="Enter text to embed"
          />
          <button type="submit">Generate Embedding</button>
        </form>
      )}
      {result && (
        <pre>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
```

## Documentation

For complete documentation, see the [docs directory](./docs).

## License

MIT
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    # Create rollup.config.js
    rollup_config_path = os.path.join(Config.TARGET_DIR, "rollup.config.js")
    if not os.path.exists(rollup_config_path):
        logger.info(f"Creating {rollup_config_path}")
        
        rollup_config = """import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import { terser } from 'rollup-plugin-terser';
import pkg from './package.json';

export default [
  // Browser-friendly UMD build
  {
    input: 'src/index.ts',
    output: {
      name: 'ipfsAccelerate',
      file: pkg.main,
      format: 'umd',
      sourcemap: true,
      globals: {
        'react': 'React'
      }
    },
    plugins: [
      resolve(),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json' }),
      terser()
    ],
    external: ['react']
  },
  
  // ESM build for modern bundlers
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'es',
      sourcemap: true
    },
    plugins: [
      resolve(),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json' })
    ],
    external: ['react']
  }
];
"""
        with open(rollup_config_path, 'w', encoding='utf-8') as f:
            f.write(rollup_config)

def create_migration_report():
    """Create a detailed migration report"""
    logger.info("Creating migration report...")
    
    if Config.DRY_RUN:
        logger.info("Dry run: Would create migration report")
        return
    
    # Generate report filename
    report_path = os.path.join(Config.TARGET_DIR, f"MIGRATION_REPORT_{Config.TIMESTAMP}.md")
    
    # Generate file counts by extension
    file_counts = {}
    for root, _, files in os.walk(Config.TARGET_DIR):
        for file in files:
            _, ext = os.path.splitext(file)
            ext = ext.lower()
            if ext not in file_counts:
                file_counts[ext] = 0
            file_counts[ext] += 1
    
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# WebGPU/WebNN JavaScript SDK Migration Report\n\n")
        f.write(f"**Migration Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes the results of migrating WebGPU and WebNN implementations ")
        f.write("from Python to a dedicated JavaScript SDK.\n\n")
        
        f.write("## Migration Statistics\n\n")
        f.write(f"- **Files Processed:** {Config.MIGRATION_STATS['files_processed']}\n")
        f.write(f"- **Python Files Converted to TypeScript:** {Config.MIGRATION_STATS['files_converted']}\n")
        f.write(f"- **Files Copied (TS/JS/WGSL):** {Config.MIGRATION_STATS['copied_files']}\n")
        f.write(f"- **Conversion Failures:** {Config.MIGRATION_STATS['conversion_failures']}\n")
        f.write(f"- **Empty Directories with Placeholders:** {Config.MIGRATION_STATS['empty_files_created']}\n")
        f.write(f"- **WebGPU-Related Files:** {Config.MIGRATION_STATS['webgpu_files']}\n")
        f.write(f"- **WebNN-Related Files:** {Config.MIGRATION_STATS['webnn_files']}\n")
        f.write(f"- **WGSL Shaders:** {Config.MIGRATION_STATS['wgsl_shaders']}\n\n")
        
        f.write("## File Distribution by Type\n\n")
        f.write("```\n")
        for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{count:4d} {ext}\n")
        f.write("```\n\n")
        
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        for root, dirs, files in os.walk(Config.TARGET_DIR):
            level = root.replace(Config.TARGET_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            for file in files:
                if file.startswith('.') or file == "MIGRATION_REPORT_":
                    continue
                f.write(f"{indent}  {file}\n")
        f.write("```\n\n")
        
        f.write("## Conversion Process\n\n")
        f.write("The migration script automatically converts Python files to TypeScript using pattern matching ")
        f.write("and specialized templates for WebGPU and WebNN related classes. Key conversions include:\n\n")
        f.write("- Python classes to TypeScript classes\n")
        f.write("- Python type hints to TypeScript type annotations\n")
        f.write("- Python methods to TypeScript methods\n")
        f.write("- WebGPU/WebNN specific API naming conventions\n")
        f.write("- WGSL shader organization by browser target\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. **Install Dependencies:**\n")
        f.write("   ```bash\n")
        f.write(f"   cd {Config.TARGET_DIR}\n")
        f.write("   npm install\n")
        f.write("   ```\n\n")
        
        f.write("2. **Test Compilation:**\n")
        f.write("   ```bash\n")
        f.write("   npm run build\n")
        f.write("   ```\n\n")
        
        f.write("3. **Review Converted Files:**\n")
        f.write("   - Check conversion quality, especially complex Python code\n")
        f.write("   - Complete implementation of placeholder files\n")
        f.write("   - Fix any remaining TypeScript errors\n\n")
        
        f.write("4. **Implement Tests:**\n")
        f.write("   ```bash\n")
        f.write("   npm test\n")
        f.write("   ```\n\n")
        
        f.write("5. **Build Documentation:**\n")
        f.write("   ```bash\n")
        f.write("   npm run docs\n")
        f.write("   ```\n\n")
        
        f.write("## Challenges and Solutions\n\n")
        
        f.write("### Python to TypeScript Conversion\n\n")
        f.write("The primary challenge was converting Python-specific constructs to TypeScript. ")
        f.write("This includes:\n\n")
        f.write("- **Classes and Inheritance:** Different inheritance patterns between languages\n")
        f.write("- **Type Annotations:** Python type hints vs TypeScript type annotations\n")
        f.write("- **Asynchronous Code:** Python async/await patterns vs JavaScript Promises\n")
        f.write("- **Browser APIs:** Python code using browser APIs required special handling\n\n")
        
        f.write("### WebGPU/WebNN Specific Considerations\n\n")
        f.write("WebGPU and WebNN have specific considerations:\n\n")
        f.write("- **API Naming Conventions:** Different method naming between languages\n")
        f.write("- **Browser-Specific Optimizations:** Each browser has unique optimizations\n")
        f.write("- **WGSL Shader Organization:** Organizing shaders by target browser\n")
        f.write("- **Hardware Detection:** Handling hardware capabilities across browsers\n\n")
        
        f.write("## Migration Log\n\n")
        f.write(f"For detailed migration logs, see: `{Config.LOG_FILE}`\n")
    
    logger.info(f"Migration report written to: {report_path}")

def create_directory_structure():
    """Create the base directory structure for the SDK"""
    logger.info("Creating directory structure...")
    
    if Config.DRY_RUN:
        logger.info("Dry run: Would create directory structure")
        return
    
    # Create the main target directory
    os.makedirs(Config.TARGET_DIR, exist_ok=True)
    
    # Define all directories to create
    directories = [
        # Source code directories
        os.path.join(Config.TARGET_DIR, "src/worker/webnn"),
        os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/chrome"),
        os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/firefox"),
        os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/edge"),
        os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/safari"),
        os.path.join(Config.TARGET_DIR, "src/worker/webgpu/shaders/model_specific"),
        os.path.join(Config.TARGET_DIR, "src/worker/webgpu/compute"),
        os.path.join(Config.TARGET_DIR, "src/worker/webgpu/pipeline"),
        os.path.join(Config.TARGET_DIR, "src/worker/wasm"),
        os.path.join(Config.TARGET_DIR, "src/api_backends"),
        os.path.join(Config.TARGET_DIR, "src/hardware/backends"),
        os.path.join(Config.TARGET_DIR, "src/hardware/detection"),
        os.path.join(Config.TARGET_DIR, "src/utils"),
        os.path.join(Config.TARGET_DIR, "src/utils/browser"),
        os.path.join(Config.TARGET_DIR, "src/model"),
        os.path.join(Config.TARGET_DIR, "src/model/transformers"),
        os.path.join(Config.TARGET_DIR, "src/model/loaders"),
        os.path.join(Config.TARGET_DIR, "src/optimization/techniques"),
        os.path.join(Config.TARGET_DIR, "src/optimization/memory"),
        os.path.join(Config.TARGET_DIR, "src/quantization"),
        os.path.join(Config.TARGET_DIR, "src/quantization/techniques"),
        os.path.join(Config.TARGET_DIR, "src/benchmark"),
        os.path.join(Config.TARGET_DIR, "src/storage"),
        os.path.join(Config.TARGET_DIR, "src/storage/indexeddb"),
        os.path.join(Config.TARGET_DIR, "src/react"),
        os.path.join(Config.TARGET_DIR, "src/browser/optimizations"),
        os.path.join(Config.TARGET_DIR, "src/tensor"),
        os.path.join(Config.TARGET_DIR, "src/p2p"),
        
        # Distribution directory
        os.path.join(Config.TARGET_DIR, "dist"),
        
        # Example directories
        os.path.join(Config.TARGET_DIR, "examples/browser/basic"),
        os.path.join(Config.TARGET_DIR, "examples/browser/advanced"),
        os.path.join(Config.TARGET_DIR, "examples/browser/react"),
        os.path.join(Config.TARGET_DIR, "examples/browser/streaming"),
        os.path.join(Config.TARGET_DIR, "examples/node"),
        
        # Test directories
        os.path.join(Config.TARGET_DIR, "test/unit"),
        os.path.join(Config.TARGET_DIR, "test/integration"),
        os.path.join(Config.TARGET_DIR, "test/browser"),
        os.path.join(Config.TARGET_DIR, "test/performance"),
        
        # Documentation directories
        os.path.join(Config.TARGET_DIR, "docs/api"),
        os.path.join(Config.TARGET_DIR, "docs/examples"),
        os.path.join(Config.TARGET_DIR, "docs/guides"),
        os.path.join(Config.TARGET_DIR, "docs/architecture"),
    ]
    
    # Create all directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Directory structure created successfully")

def create_main_index_file():
    """Create the main index.ts file for the SDK"""
    logger.info("Creating main index.ts file...")
    
    if Config.DRY_RUN:
        logger.info("Dry run: Would create main index.ts file")
        return
    
    # Create the main index.ts file
    index_path = os.path.join(Config.TARGET_DIR, "src/index.ts")
    
    if not os.path.exists(index_path):
        logger.info(f"Creating {index_path}")
        
        index_content = """/**
 * IPFS Accelerate JavaScript SDK
 * 
 * The main entry point for the IPFS Accelerate JavaScript SDK.
 * This SDK provides hardware-accelerated machine learning for web browsers and Node.js.
 * 
 * @packageDocumentation
 */

// Hardware acceleration
export * from './hardware/hardware_abstraction';
export * from './hardware/backends/webgpu_backend';
export * from './hardware/backends/webnn_backend';
export * from './hardware/detection/gpu_detection';

// Model loaders
export * from './model/model_loader';
export * from './model/transformers/tensorflow_adapter';
export * from './model/transformers/onnx_adapter';

// Quantization engine
export * from './quantization/quantization_engine';

// Tensor operations
export * from './tensor/tensor_sharing';

// Storage
export * from './storage/indexeddb/storage_manager';

// API backends
export * from './api_backends';

// React integration
import * as React from './react/hooks';
export { React };

/**
 * Create an accelerator instance with the specified options
 * @param options Accelerator options
 * @returns An initialized accelerator instance
 */
export async function createAccelerator(options: any = {}) {
  const { HardwareAbstraction } = await import('./hardware/hardware_abstraction');
  const hardwareAbstraction = new HardwareAbstraction(options);
  await hardwareAbstraction.initialize();
  return hardwareAbstraction;
}

/**
 * Library version
 */
export const VERSION = '0.1.0';
"""
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        logger.info(f"Created main index.ts file")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Python to JavaScript/TypeScript Converter for IPFS Accelerate")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--force", action="store_true", help="Skip confirmation and update existing files")
    parser.add_argument("--target-dir", help="Set custom target directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Setup configuration
    setup_config(args)
    
    # Check if target directory already exists
    if os.path.exists(Config.TARGET_DIR) and not Config.DRY_RUN and not Config.FORCE:
        response = input(f"Directory {Config.TARGET_DIR} already exists. Continue and update existing files? (y/n): ")
        if response.lower() != 'y':
            logger.info("Operation cancelled by user")
            return
    
    # Create directory structure
    create_directory_structure()
    
    # Create base project files
    create_base_project_files()
    
    # Create main index file
    create_main_index_file()
    
    # Find WebGPU/WebNN files
    files = FileFinder.find_webgpu_webnn_files()
    logger.info(f"Found {len(files)} files to process")
    
    # Process files
    for file_path in files:
        dest_path = FileFinder.map_file_to_destination(file_path)
        FileProcessor.process_file(file_path, dest_path)
    
    # Create placeholders for empty directories
    FileProcessor.create_placeholder_for_empty_dirs()
    
    # Create migration report
    create_migration_report()
    
    # Log summary
    logger.info("Migration completed successfully!")
    logger.info(f"Total files processed: {Config.MIGRATION_STATS['files_processed']}")
    logger.info(f"Files converted: {Config.MIGRATION_STATS['files_converted']}")
    logger.info(f"Files copied: {Config.MIGRATION_STATS['copied_files']}")
    logger.info(f"Conversion failures: {Config.MIGRATION_STATS['conversion_failures']}")
    logger.info(f"Empty files created: {Config.MIGRATION_STATS['empty_files_created']}")
    logger.info(f"WebGPU files: {Config.MIGRATION_STATS['webgpu_files']}")
    logger.info(f"WebNN files: {Config.MIGRATION_STATS['webnn_files']}")
    logger.info(f"WGSL shaders: {Config.MIGRATION_STATS['wgsl_shaders']}")

if __name__ == "__main__":
    main()