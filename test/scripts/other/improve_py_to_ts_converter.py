#!/usr/bin/env python3
# improve_py_to_ts_converter.py
# Enhanced Python to TypeScript converter with improved pattern mapping, class templates, and import path resolution

import os
import sys
import re
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'improve_py_to_ts_converter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Global configuration
class Config:
    SOURCE_CONVERTER_PATH = None
    OUTPUT_CONVERTER_PATH = None
    TEST_FILE = None
    APPLY_CHANGES = False
    TEST_MODE = False
    VERBOSE = False

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Improve the Python to TypeScript converter")
    parser.add_argument("--source", default="setup_ipfs_accelerate_js_py_converter.py", 
                        help="Source converter script path")
    parser.add_argument("--output", help="Output path for improved converter (defaults to source with _improved suffix)")
    parser.add_argument("--test-file", help="Test a specific Python file conversion")
    parser.add_argument("--apply", action="store_true", help="Apply changes directly to source file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    Config.SOURCE_CONVERTER_PATH = os.path.abspath(args.source)
    if not os.path.exists(Config.SOURCE_CONVERTER_PATH):
        logger.error(f"Source converter not found: {Config.SOURCE_CONVERTER_PATH}")
        sys.exit(1)
    
    if args.output:
        Config.OUTPUT_CONVERTER_PATH = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(Config.SOURCE_CONVERTER_PATH)
        Config.OUTPUT_CONVERTER_PATH = f"{base}_improved{ext}"
    
    Config.TEST_FILE = args.test_file
    if Config.TEST_FILE and os.path.exists(Config.TEST_FILE):
        Config.TEST_MODE = True
    
    Config.APPLY_CHANGES = args.apply
    Config.VERBOSE = args.verbose
    
    logger.info(f"Source converter: {Config.SOURCE_CONVERTER_PATH}")
    logger.info(f"Output path: {Config.OUTPUT_CONVERTER_PATH}")
    logger.info(f"Test mode: {Config.TEST_MODE}")
    logger.info(f"Apply changes: {Config.APPLY_CHANGES}")

class ConverterImprovements:
    """Contains the improved patterns and templates for the Python to TypeScript converter"""
    
    # Improved pattern mapping for better conversion accuracy
    IMPROVED_PATTERN_MAP = [
        # Import statements with better handling of paths and relative imports
        (r'import\s+(\w+)', r'import * as $1'),
        (r'from\s+(\w+)\s+import\s+\{([^}]+)\}', r'import { $2 } from "$1"'),
        (r'from\s+(\w+)\s+import\s+(.+)', r'import { $2 } from "$1"'),
        (r'from\s+\.(\w+)\s+import\s+(.+)', r'import { $2 } from "./$1"'),
        (r'from\s+\.\.\s+import\s+(.+)', r'import { $1 } from ".."'),
        
        # Class definitions with inheritance and interfaces
        (r'class\s+(\w+)(?:\((\w+)\))?:', r'class $1 extends $2 {'),
        (r'class\s+(\w+):', r'class $1 {'),
        
        # Constructor handling
        (r'def\s+__init__\s*\(self(?:,\s*([^)]+))?\):', r'constructor($1) {'),
        
        # Method definitions with proper return types
        (r'def\s+(\w+)\s*\((self)(?:,\s*([^)]+))?\)\s*->\s*(\w+):', r'$1($3): $4 {'),
        (r'def\s+(\w+)\s*\((self)(?:,\s*([^)]+))?\):', r'$1($3) {'),
        (r'async\s+def\s+(\w+)\s*\((self)(?:,\s*([^)]+))?\)\s*->\s*(\w+):', r'async $1($3): Promise<$4> {'),
        (r'async\s+def\s+(\w+)\s*\((self)(?:,\s*([^)]+))?\):', r'async $1($3): Promise<any> {'),
        
        # Static method handling
        (r'@staticmethod\s+def\s+(\w+)\s*\(([^)]*)\)\s*->\s*(\w+):', r'static $1($2): $3 {'),
        (r'@staticmethod\s+def\s+(\w+)\s*\(([^)]*)\):', r'static $1($2): any {'),
        
        # Property and getter/setter handling
        (r'@property\s+def\s+(\w+)\s*\(self\)\s*->\s*(\w+):', r'get $1(): $2 {'),
        (r'@property\s+def\s+(\w+)\s*\(self\):', r'get $1(): any {'),
        (r'@(\w+)\.setter\s+def\s+\1\s*\(self,\s*(\w+)(?:\s*:\s*([^)]+))?\):', r'set $1($2: $3) {'),
        
        # Type hints with better handling of complex types
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
        
        # Literal type conversion (for enum-like values)
        (r'(\w+):\s*Literal\[([^\]]+)\]', r'$1: $2'),
        
        # Self reference conversion
        (r'self\.', r'this.'),
        (r'super\(\)\.(\w+)', r'super.$1'),
        
        # Control structures with improved handling
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
        
        # List operations
        (r'(\w+)\.append\((.*?)\)', r'$1.push($2)'),
        (r'\[(.*?) for (.*?) in (.*?)\]', r'$3.map(($2) => $1)'),
        (r'\[(.*?) for (.*?) in (.*?) if (.*?)\]', r'$3.filter(($2) => $4).map(($2) => $1)'),
        
        # Dictionary operations
        (r'(\w+)\.items\(\)', r'Object.entries($1)'),
        (r'(\w+)\.keys\(\)', r'Object.keys($1)'),
        (r'(\w+)\.values\(\)', r'Object.values($1)'),
        (r'(\w+)\.get\((.*?), (.*?)\)', r'$1[$2] ?? $3'),
        (r'(\w+)\.get\((.*?)\)', r'$1[$2]'),
        
        # Boolean operators
        (r' and ', r' && '),
        (r' or ', r' || '),
        (r'not ', r'!'),
        
        # None/null
        (r'None', r'null'),
        (r'True', r'true'),
        (r'False', r'false'),
        
        # f-strings with improved handling of expressions
        (r'f[\'"](.+?)[\'"]', r'`$1`'),
        (r'{([^{}]+?)}', r'${$1}'),
        
        # Comments
        (r'#\s*(.*?)$', r'// $1'),
        
        # Print statements
        (r'print\((.*?)\)', r'console.log($1)'),
        
        # Async/await with better Promise handling
        (r'async\s+def', r'async'),
        (r'await\s+', r'await '),
        
        # Return statement with proper semicolon
        (r'return\s+(.*)', r'return $1;'),
        (r'return;', r'return;'),
        
        # Assert statement
        (r'assert\s+(.*?)(,\s*[\'"](.+)[\'"])?', r'if (!($1)) { throw new Error($3 || "Assertion failed"); }'),
        
        # WebGPU specific conversions with camelCase
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
        
        # WebNN specific conversions with camelCase
        (r'navigator\.ml\.create_context', r'navigator.ml.createContext'),
        (r'create_graph_builder', r'createGraphBuilder'),
        (r'create_graph', r'createGraph'),
        (r'create_model', r'createModel'),
        (r'build_graph', r'buildGraph'),
        (r'create_operand', r'createOperand'),
        
        # More Python to JavaScript conversions
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
    ]
    
    # Enhanced WebGPU class template with proper TypeScript typing
    WEBGPU_CLASS_TEMPLATE = {
        'signature': 'class WebGPUBackend implements HardwareBackend',
        'methods': {
            'initialize': '''async initialize(): Promise<boolean> {
    try {
      // Request adapter from navigator.gpu
      this.adapter = await navigator.gpu.requestAdapter();
      
      if (!this.adapter) {
        console.error("WebGPU not supported or disabled");
        return false;
      }
      
      // Request device from adapter
      this.device = await this.adapter.requestDevice();
      
      if (!this.device) {
        console.error("Failed to get WebGPU device");
        return false;
      }
      
      this.initialized = true;
      return true;
    } catch (error) {
      console.error("WebGPU initialization error:", error);
      return false;
    }
  }''',
            'createBuffer': '''createBuffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer | null {
    if (!this.device) {
      console.error("WebGPU device not initialized");
      return null;
    }
    
    try {
      return this.device.createBuffer({
        size,
        usage,
        mappedAtCreation: false
      });
    } catch (error) {
      console.error("Error creating WebGPU buffer:", error);
      return null;
    }
  }''',
            'createComputePipeline': '''async createComputePipeline(shader: string): Promise<GPUComputePipeline | null> {
    if (!this.device) {
      console.error("WebGPU device not initialized");
      return null;
    }
    
    try {
      const shaderModule = this.device.createShaderModule({
        code: shader
      });
      
      return this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });
    } catch (error) {
      console.error("Error creating compute pipeline:", error);
      return null;
    }
  }''',
            'runCompute': '''async runCompute(
    pipeline: GPUComputePipeline,
    bindings: GPUBindGroup[],
    workgroups: [number, number?, number?]
  ): Promise<void> {
    if (!this.device) {
      throw new Error("WebGPU device not initialized");
    }
    
    try {
      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      
      passEncoder.setPipeline(pipeline);
      
      for (let i = 0; i < bindings.length; i++) {
        passEncoder.setBindGroup(i, bindings[i]);
      }
      
      const [x, y = 1, z = 1] = workgroups;
      passEncoder.dispatchWorkgroups(x, y, z);
      passEncoder.end();
      
      const commandBuffer = commandEncoder.finish();
      this.device.queue.submit([commandBuffer]);
      
      // Wait for GPU to complete
      await this.device.queue.onSubmittedWorkDone();
    } catch (error) {
      console.error("Error running compute operation:", error);
      throw error;
    }
  }''',
            'destroy': '''destroy(): void {
    if (this.adapter) {
      // Clean up any resources
      this.adapter = null;
    }
    
    if (this.device) {
      // Clean up device
      this.device = null;
    }
    
    this.initialized = false;
  }'''
        },
        'properties': {
            'device': 'device: GPUDevice | null = null',
            'adapter': 'adapter: GPUAdapter | null = null',
            'initialized': 'initialized: boolean = false',
            'features': 'features: Set<string> = new Set()',
            'limits': 'limits: Record<string, number> = {}'
        }
    }
    
    # Enhanced WebNN class template with proper TypeScript typing
    WEBNN_CLASS_TEMPLATE = {
        'signature': 'class WebNNBackend implements HardwareBackend',
        'methods': {
            'initialize': '''async initialize(): Promise<boolean> {
    try {
      if (!navigator.ml) {
        console.error("WebNN not supported in this browser");
        return false;
      }
      
      this.context = navigator.ml.createContext();
      
      if (!this.context) {
        console.error("Failed to create WebNN context");
        return false;
      }
      
      this.initialized = true;
      return true;
    } catch (error) {
      console.error("WebNN initialization error:", error);
      return false;
    }
  }''',
            'createGraphBuilder': '''createGraphBuilder(): MLGraphBuilder | null {
    if (!this.context) {
      console.error("WebNN context not initialized");
      return null;
    }
    
    try {
      return new MLGraphBuilder(this.context);
    } catch (error) {
      console.error("Error creating graph builder:", error);
      return null;
    }
  }''',
            'buildGraph': '''async buildGraph(
    graphBuilder: MLGraphBuilder,
    outputs: Record<string, MLOperand>
  ): Promise<MLGraph | null> {
    if (!this.initialized) {
      console.error("WebNN not initialized");
      return null;
    }
    
    try {
      return await graphBuilder.build(outputs);
    } catch (error) {
      console.error("Error building graph:", error);
      return null;
    }
  }''',
            'runInference': '''async runInference(
    graph: MLGraph,
    inputs: Record<string, MLOperand>
  ): Promise<Record<string, MLOperand>> {
    if (!this.initialized) {
      throw new Error("WebNN not initialized");
    }
    
    try {
      return graph.compute(inputs);
    } catch (error) {
      console.error("Error running inference:", error);
      throw error;
    }
  }''',
            'destroy': '''destroy(): void {
    this.context = null;
    this.initialized = false;
  }'''
        },
        'properties': {
            'context': 'context: MLContext | null = null',
            'initialized': 'initialized: boolean = false',
            'capabilities': 'capabilities: string[] = []'
        }
    }
    
    # Enhanced HardwareAbstraction class template with proper TypeScript typing
    HARDWARE_ABSTRACTION_TEMPLATE = {
        'signature': 'class HardwareAbstraction',
        'methods': {
            'initialize': '''async initialize(): Promise<boolean> {
    try {
      // Initialize hardware detection
      const hardwareDetection = new HardwareDetection();
      const capabilities = await hardwareDetection.detectCapabilities();
      
      // Initialize backends based on available hardware
      if (capabilities.webgpu) {
        const webgpuBackend = new WebGPUBackend();
        const success = await webgpuBackend.initialize();
        if (success) {
          this.backends.set('webgpu', webgpuBackend);
        }
      }
      
      if (capabilities.webnn) {
        const webnnBackend = new WebNNBackend();
        const success = await webnnBackend.initialize();
        if (success) {
          this.backends.set('webnn', webnnBackend);
        }
      }
      
      // Always add CPU backend as fallback
      const cpuBackend = new CPUBackend();
      await cpuBackend.initialize();
      this.backends.set('cpu', cpuBackend);
      
      // Apply hardware preferences
      this.applyPreferences();
      
      return this.backends.size > 0;
    } catch (error) {
      console.error("Error initializing hardware abstraction:", error);
      return false;
    }
  }''',
            'getBestBackend': '''getBestBackend(modelType: string): HardwareBackend {
    // Check if we have a preference for this model type
    if (
      this.preferences &&
      this.preferences.modelPreferences &&
      this.preferences.modelPreferences[modelType]
    ) {
      const preferredBackends = this.preferences.modelPreferences[modelType];
      
      // Try each preferred backend in order
      for (const backendName of preferredBackends) {
        if (this.backends.has(backendName)) {
          return this.backends.get(backendName)!;
        }
      }
    }
    
    // Fallback to order of preference: WebGPU > WebNN > CPU
    if (this.backends.has('webgpu')) return this.backends.get('webgpu')!;
    if (this.backends.has('webnn')) return this.backends.get('webnn')!;
    
    // Always have CPU as fallback
    return this.backends.get('cpu')!;
  }''',
            'runModel': '''async runModel<T = any, U = any>(model: Model, inputs: T): Promise<U> {
    const backend = this.getBestBackend(model.type);
    return model.execute(inputs, backend);
  }''',
            'destroy': '''destroy(): void {
    // Destroy all backends
    for (const backend of this.backends.values()) {
      backend.destroy();
    }
    
    this.backends.clear();
  }''',
            'applyPreferences': '''private applyPreferences(): void {
    // Apply any hardware preferences from configuration
    if (this.preferences && this.preferences.backendOrder) {
      // Reorder backends based on preferences
      this.backendOrder = this.preferences.backendOrder.filter(
        backend => this.backends.has(backend)
      );
    } else {
      // Default order: WebGPU > WebNN > CPU
      this.backendOrder = ['webgpu', 'webnn', 'cpu'].filter(
        backend => this.backends.has(backend)
      );
    }
  }'''
        },
        'properties': {
            'backends': 'backends: Map<string, HardwareBackend> = new Map()',
            'preferences': 'preferences: HardwarePreferences',
            'backendOrder': 'backendOrder: string[] = []'
        }
    }
    
    # Enhanced class conversions mapping
    IMPROVED_CLASS_CONVERSIONS = {
        'WebGPUBackend': WEBGPU_CLASS_TEMPLATE,
        'WebNNBackend': WEBNN_CLASS_TEMPLATE,
        'HardwareAbstraction': HARDWARE_ABSTRACTION_TEMPLATE,
    }
    
    # TypeScript interfaces for common types
    TS_INTERFACES = {
        'HardwareBackend': '''interface HardwareBackend {
  initialize(): Promise<boolean>;
  destroy(): void;
}''',
        'HardwarePreferences': '''interface HardwarePreferences {
  backendOrder?: string[];
  modelPreferences?: Record<string, string[]>;
  options?: Record<string, any>;
}''',
        'ModelConfig': '''interface ModelConfig {
  id: string;
  type: string;
  path?: string;
  options?: Record<string, any>;
}''',
        'Model': '''interface Model {
  id: string;
  type: string;
  execute<T = any, U = any>(inputs: T, backend: HardwareBackend): Promise<U>;
}'''
    }
    
    @staticmethod
    def create_improved_interfaces(file_content: str) -> str:
        """Extract interfaces from Python types and create TypeScript interfaces"""
        interfaces = []
        
        # Look for type annotations for classes
        class_matches = re.finditer(r'class\s+(\w+)(?:\(([^)]+)\))?:', file_content)
        for match in class_matches:
            class_name = match.group(1)
            # Extract property annotations from the class
            props = {}
            prop_matches = re.finditer(r'self\.(\w+)(?:\s*:\s*([^=\n]+))?(?:\s*=\s*([^#\n]+))?', file_content)
            for prop_match in prop_matches:
                prop_name = prop_match.group(1)
                prop_type = prop_match.group(2)
                prop_default = prop_match.group(3)
                
                if prop_type:
                    # Convert Python type to TypeScript
                    ts_type = prop_type.strip()
                    ts_type = re.sub(r'str', 'string', ts_type)
                    ts_type = re.sub(r'int|float', 'number', ts_type)
                    ts_type = re.sub(r'bool', 'boolean', ts_type)
                    ts_type = re.sub(r'List\[(\w+)\]', r'$1[]', ts_type)
                    ts_type = re.sub(r'Dict\[(\w+),\s*(\w+)\]', r'Record<$1, $2>', ts_type)
                    ts_type = re.sub(r'Optional\[(\w+)\]', r'$1 | null', ts_type)
                    ts_type = re.sub(r'Any', 'any', ts_type)
                    
                    props[prop_name] = ts_type
            
            # If we found properties, create an interface
            if props:
                interface = f"interface {class_name}Props {{\n"
                for prop_name, prop_type in props.items():
                    interface += f"  {prop_name}: {prop_type};\n"
                interface += "}\n\n"
                interfaces.append(interface)
        
        # Add standard interfaces
        for interface_name, interface_def in ConverterImprovements.TS_INTERFACES.items():
            interfaces.append(interface_def + "\n")
        
        return "\n".join(interfaces) if interfaces else ""
    
    @staticmethod
    def fix_import_paths(original_content: str, ts_content: str, file_path: str, target_dir: str) -> str:
        """Fix import paths in the converted TypeScript content"""
        # Extract Python imports
        py_imports = re.findall(r'(?:from|import)\s+([.\w]+)', original_content)
        
        # Extract TypeScript imports
        ts_imports = re.findall(r'import.*?from\s+[\'"]([^\'"]+)[\'"]', ts_content)
        
        # Create a mapping from Python module to TypeScript path
        import_mapping = {}
        for py_import in py_imports:
            # Skip standard lib imports
            if py_import in ('os', 'sys', 're', 'json', 'logging', 'datetime', 'pathlib', 'typing'):
                continue
                
            # Convert Python import to potential TypeScript path
            if '.' in py_import:
                parts = py_import.split('.')
                ts_path = '/'.join(parts)
                import_mapping[py_import] = f'./{ts_path}'
            else:
                # For single module imports, map to ./modulename
                import_mapping[py_import] = f'./{py_import}'
        
        # Fix the imports in TypeScript content
        for py_import, ts_path in import_mapping.items():
            # Replace simple imports
            ts_content = re.sub(
                fr'from\s+[\'"]({py_import})[\'"]',
                f'from "{ts_path}"',
                ts_content
            )
            
            # Handle relative imports that might not match exactly
            for ts_import in ts_imports:
                if py_import in ts_import:
                    ts_content = re.sub(
                        fr'from\s+[\'"]({ts_import})[\'"]',
                        f'from "{ts_path}"',
                        ts_content
                    )
        
        return ts_content

def update_pattern_map(source_content: str) -> str:
    """Update the pattern mapping in the source converter code"""
    # Find PATTERN_MAP in source
    pattern_map_match = re.search(r'PATTERN_MAP\s*=\s*\[(.*?)\]', source_content, re.DOTALL)
    
    if not pattern_map_match:
        logger.error("Could not find PATTERN_MAP in source code")
        return source_content
    
    # Replace with improved pattern map
    improved_pattern_map = "PATTERN_MAP = [\n        # Import statements with better handling of paths and relative imports\n"
    for pattern, replacement in ConverterImprovements.IMPROVED_PATTERN_MAP:
        improved_pattern_map += f"        (r'{pattern}', r'{replacement}'),\n"
    improved_pattern_map += "    ]"
    
    updated_content = source_content.replace(
        pattern_map_match.group(0),
        improved_pattern_map
    )
    
    return updated_content

def update_class_conversions(source_content: str) -> str:
    """Update the class templates in the source converter code"""
    # Find CLASS_CONVERSIONS in source
    class_conv_match = re.search(r'CLASS_CONVERSIONS\s*=\s*\{(.*?)\}', source_content, re.DOTALL)
    
    if not class_conv_match:
        logger.error("Could not find CLASS_CONVERSIONS in source code")
        return source_content
    
    # Create improved class conversions
    improved_class_conversions = "CLASS_CONVERSIONS = {\n"
    
    for class_name, template in ConverterImprovements.IMPROVED_CLASS_CONVERSIONS.items():
        improved_class_conversions += f"        '{class_name}': {{\n"
        improved_class_conversions += f"            'signature': '{template['signature']}',\n"
        
        # Add methods
        improved_class_conversions += "            'methods': {\n"
        for method_name, method_body in template['methods'].items():
            # Escape single quotes in the method body
            method_body_escaped = method_body.replace("'", "\\'")
            # Add proper indentation
            method_body_formatted = method_body_escaped.replace("\n", "\\n")
            improved_class_conversions += f"                '{method_name}': '{method_body_formatted}',\n"
        improved_class_conversions += "            },\n"
        
        # Add properties
        improved_class_conversions += "            'properties': {\n"
        for prop_name, prop_def in template['properties'].items():
            improved_class_conversions += f"                '{prop_name}': '{prop_def}',\n"
        improved_class_conversions += "            }\n"
        
        improved_class_conversions += "        },\n"
    
    improved_class_conversions += "    }"
    
    updated_content = source_content.replace(
        class_conv_match.group(0),
        improved_class_conversions
    )
    
    return updated_content

def enhance_generate_class_method(source_content: str) -> str:
    """Enhance the _generate_class_from_template method"""
    # Find _generate_class_from_template method
    method_match = re.search(r'def\s+_generate_class_from_template.*?return\s+result', source_content, re.DOTALL)
    
    if not method_match:
        logger.error("Could not find _generate_class_from_template method")
        return source_content
    
    # Create improved method
    improved_method = '''    @staticmethod
    def _generate_class_from_template(class_name: str, content: str) -> str:
        """Generate TypeScript class from predefined template with enhanced typing"""
        template = PyToTsConverter.CLASS_CONVERSIONS[class_name]
        
        # Add extracted interfaces from content
        interfaces = ""
        class_props = re.findall(r'self\.(\w+)(?:\s*:\s*([^=\n]+))?(?:\s*=\s*([^#\n]+))?', content)
        if class_props:
            prop_interface = f"interface {class_name}Props {\\n"
            for prop_match in class_props:
                prop_name = prop_match[0]
                prop_type = prop_match[1] if len(prop_match) > 1 and prop_match[1] else "any"
                
                # Convert Python type to TypeScript
                ts_type = prop_type.strip()
                ts_type = re.sub(r'str', 'string', ts_type)
                ts_type = re.sub(r'int|float', 'number', ts_type)
                ts_type = re.sub(r'bool', 'boolean', ts_type)
                ts_type = re.sub(r'List\\[([\\w]+)\\]', r'$1[]', ts_type)
                ts_type = re.sub(r'Dict\\[([\\w]+),\\s*([\\w]+)\\]', r'Record<$1, $2>', ts_type)
                ts_type = re.sub(r'Optional\\[([\\w]+)\\]', r'$1 | null', ts_type)
                ts_type = re.sub(r'Any', 'any', ts_type)
                
                prop_interface += f"  {prop_name}: {ts_type};\\n"
            prop_interface += "}\\n\\n"
            interfaces += prop_interface
        
        # Generate class definition
        result = interfaces + f"{template['signature']} {\\n"
        
        # Add properties with TypeScript types
        for prop_name, prop_def in template['properties'].items():
            result += f"  {prop_def};\\n"
        
        result += "\\n"
        
        # Add constructor with proper parameter types
        constructor_match = re.search(r'def __init__\\s*\\(self(?:,\\s*([^)]+))?\\):', content)
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
                    param_type = re.sub(r'List\\[([\\w]+)\\]', r'$1[]', param_type)
                    param_type = re.sub(r'Dict\\[([\\w]+),\\s*([\\w]+)\\]', r'Record<$1, $2>', param_type)
                    param_type = re.sub(r'Optional\\[([\\w]+)\\]', r'$1 | null', param_type)
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
        
        result += f"  constructor({params}) {\\n"
        
        # Extract constructor body from Python __init__ method
        constructor_body = ""
        init_match = re.search(r'def __init__.*?:(.*?)(?=\\s+(?:async\\s+)?def|$)', content, re.DOTALL)
        if init_match:
            body = init_match.group(1).strip()
            # Convert Python constructor body to TypeScript
            body = re.sub(r'self\\.', 'this.', body)
            # Add each line with proper indentation
            for line in body.split('\\n'):
                if line.strip():
                    constructor_body += f"    {line.strip()}\\n"
        
        # If no body extracted, initialize properties
        if not constructor_body:
            for prop_name, _ in template['properties'].items():
                if prop_name != 'initialized':  # Skip 'initialized' as it's set below
                    constructor_body += f"    this.{prop_name} = {prop_name};\\n"
        
        # Always set initialized to false in constructor
        result += f"{constructor_body}    this.initialized = false;\\n  }}\\n\\n"
        
        # Add methods
        for method_name, method_body in template['methods'].items():
            result += f"  {method_body}\\n\\n"
        
        result += "}\\n"
        return result'''
    
    # Replace the original method
    updated_content = source_content.replace(
        method_match.group(0),
        improved_method
    )
    
    return updated_content

def enhance_extract_interfaces(source_content: str) -> str:
    """Enhance the _extract_interfaces method to generate better TypeScript interfaces"""
    # Find _extract_interfaces method
    method_match = re.search(r'def\s+_extract_interfaces.*?return\s+interfaces', source_content, re.DOTALL)
    
    if not method_match:
        logger.error("Could not find _extract_interfaces method")
        return source_content
    
    # Create improved method
    improved_method = '''    @staticmethod
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
        type_defs = re.findall(r'(\\w+)\\s*=\\s*(?:Dict|List|Tuple|Optional|Union)\\[([^\\]]+)\\]', content)
        for name, types in type_defs:
            # Convert Python types to TypeScript
            ts_types = types.replace("str", "string").replace("int", "number").replace("float", "number")
            ts_types = ts_types.replace("bool", "boolean").replace("Any", "any")
            ts_types = re.sub(r'List\\[([^\\]]+)\\]', r'$1[]', ts_types)
            ts_types = re.sub(r'Dict\\[([^,]+),\\s*([^\\]]+)\\]', r'Record<$1, $2>', ts_types)
            ts_types = re.sub(r'Optional\\[([^\\]]+)\\]', r'$1 | null', ts_types)
            ts_types = re.sub(r'Tuple\\[([^\\]]+)\\]', r'[$1]', ts_types)
            
            interfaces += f'export type {name} = {ts_types};\\n'
        
        # Look for class properties with type hints
        class_props = re.findall(r'self\\.(\\w+)(?:\\s*:\\s*([^=\\n]+))?(?:\\s*=\\s*([^#\\n]+))?', content)
        interface_name = ""
        
        # Try to extract class name for better interface naming
        class_match = re.search(r'class\\s+(\\w+)', content)
        if class_match:
            interface_name = f"{class_match.group(1)}Props"
        else:
            interface_name = "Props"
        
        if class_props:
            interfaces += f'\\nexport interface {interface_name} {\\n'
            
            for prop_match in class_props:
                prop_name = prop_match[0]
                prop_type = prop_match[1] if len(prop_match) > 1 and prop_match[1] else "any"
                
                # Convert Python type to TypeScript
                ts_type = prop_type.strip()
                ts_type = re.sub(r'str', 'string', ts_type)
                ts_type = re.sub(r'int|float', 'number', ts_type)
                ts_type = re.sub(r'bool', 'boolean', ts_type)
                ts_type = re.sub(r'List\\[([\\w]+)\\]', r'$1[]', ts_type)
                ts_type = re.sub(r'Dict\\[([\\w]+),\\s*([\\w]+)\\]', r'Record<$1, $2>', ts_type)
                ts_type = re.sub(r'Optional\\[([\\w]+)\\]', r'$1 | null', ts_type)
                ts_type = re.sub(r'Any', 'any', ts_type)
                
                interfaces += f'  {prop_name}: {ts_type};\\n'
            interfaces += '}\\n\\n'
        
        # Look for method parameters with type hints
        methods = re.finditer(r'def\\s+(\\w+)\\s*\\(self(?:,\\s*([^)]+))?\\)(?:\\s*->\\s*([^:]+))?:', content)
        for method in methods:
            method_name = method.group(1)
            params = method.group(2)
            return_type = method.group(3)
            
            if params:
                param_interface = f'\\ninterface {method_name.capitalize()}Params {\\n'
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
                        ts_type = re.sub(r'List\\[([\\w]+)\\]', r'$1[]', ts_type)
                        ts_type = re.sub(r'Dict\\[([\\w]+),\\s*([\\w]+)\\]', r'Record<$1, $2>', ts_type)
                        ts_type = re.sub(r'Optional\\[([\\w]+)\\]', r'$1 | null', ts_type)
                        ts_type = re.sub(r'Any', 'any', ts_type)
                        
                        # Handle default values
                        if '=' in param_name:
                            param_parts = param_name.split('=')
                            param_name = param_parts[0].strip()
                            param_interface += f'  {param_name}?: {ts_type};\\n'
                        else:
                            param_interface += f'  {param_name}: {ts_type};\\n'
                
                if has_params:
                    param_interface += '}\\n'
                    interfaces += param_interface
        
        return interfaces'''
    
    # Replace the original method
    updated_content = source_content.replace(
        method_match.group(0),
        improved_method
    )
    
    return updated_content

def enhance_map_file_to_destination(source_content: str) -> str:
    """Enhance the map_file_to_destination method for better import path handling"""
    # Find map_file_to_destination method
    method_match = re.search(r'def\s+map_file_to_destination.*?return\s+os\.path\.join\(Config\.TARGET_DIR,\s*"src/utils",\s*os\.path\.splitext\(basename\)\[0\]\s*\+\s*output_ext\)', source_content, re.DOTALL)
    
    if not method_match:
        logger.error("Could not find map_file_to_destination method")
        return source_content
    
    # Create improved method ending (keep the beginning part unchanged)
    method_end = '''        # Special file types
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
            return os.path.join(Config.TARGET_DIR, "src/utils", os.path.splitext(basename)[0] + output_ext)'''
    
    # Extract the beginning of the method (keep it unchanged)
    method_start_match = re.search(r'def\s+map_file_to_destination.*?# Special file types', source_content, re.DOTALL)
    if not method_start_match:
        logger.error("Could not find the beginning of map_file_to_destination method")
        return source_content
    
    method_start = method_start_match.group(0)
    
    # Replace the method
    updated_content = source_content.replace(
        method_match.group(0),
        method_start + method_end
    )
    
    return updated_content

def improve_add_closing_braces(source_content: str) -> str:
    """Improve the _add_closing_braces method to handle nested blocks better"""
    # Find _add_closing_braces method
    method_match = re.search(r'def\s+_add_closing_braces.*?return\s+content', source_content, re.DOTALL)
    
    if not method_match:
        logger.error("Could not find _add_closing_braces method")
        return source_content
    
    # Create improved method
    improved_method = '''    @staticmethod
    def _add_closing_braces(content: str) -> str:
        """Add closing braces to match the opening ones with improved handling of nested blocks"""
        lines = content.split('\\n')
        result_lines = []
        stack = []
        indent_stack = []
        
        for i, line in enumerate(lines):
            # Track current indentation level
            current_indent = len(re.match(r'^\\s*', line).group(0))
            
            # Look for opening braces at the end of a line
            opening_braces = len(re.findall(r'{\\s*$', line))
            if opening_braces > 0:
                for _ in range(opening_braces):
                    stack.append(i)
                    indent_stack.append(current_indent)
                result_lines.append(line)
            # Check if next line already has a closing brace that matches
            elif i < len(lines) - 1 and re.match(r'\\s*}', lines[i+1]):
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
        content = '\\n'.join(result_lines)
        
        # Fix common brace issues
        
        # Fix duplicate closing braces
        content = re.sub(r'}\\s*}([^}])', r'}\\1', content)
        
        # Fix duplicate opening braces
        content = re.sub(r'([^{]){(\\s*){', r'\\1{\\2', content)
        
        # Fix spacing around braces
        content = re.sub(r'{\\s*{', r'{', content)
        content = re.sub(r'}\\s*}', r'}', content)
        
        # Fix semicolons after braces
        content = re.sub(r'}\\s*;', r'}', content)
        
        # Ensure semicolons at the end of statements
        content = re.sub(r'(\\w+\\s*=\\s*[^;{\\n]+)\\n', r'\\1;\\n', content)
        
        # Fix missing semicolons after return statements
        content = re.sub(r'return\\s+([^;{\\n]+)\\n', r'return \\1;\\n', content)
        
        # Fix dangling else (else without preceding if)
        content = re.sub(r'\\n(\\s*)} else', r'\\n\\1}\\n\\1else', content)
        
        # Fix missing braces for single-line if statements
        content = re.sub(r'if\\s*\\(([^)]+)\\)\\s*([^{\\n][^;\\n]+);', r'if (\\1) {\\n  \\2;\\n}', content)
        
        # Fix array destructuring syntax that TypeScript doesn't support well
        content = re.sub(r'const\\s*\\[([^=]+)\\]\\s*=\\s*([^;]+);', r'const _tmp = \\2;\\nconst \\1 = _tmp;', content)
        
        return content'''
    
    # Replace the original method
    updated_content = source_content.replace(
        method_match.group(0),
        improved_method
    )
    
    return updated_content

def test_conversion(python_file: str, converter_path: str) -> str:
    """Test the conversion process on a sample Python file"""
    logger.info(f"Testing conversion of {python_file} with {converter_path}")
    
    try:
        # Load the Python file
        with open(python_file, 'r', encoding='utf-8') as f:
            py_content = f.read()
        
        # Load the converter module
        import importlib.util
        spec = importlib.util.spec_from_file_location("py_to_ts_converter", converter_path)
        converter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter_module)
        
        # Get the converter class
        PyToTsConverter = converter_module.PyToTsConverter
        
        # Convert the Python file
        ts_content = PyToTsConverter.convert_py_to_ts(py_content, python_file)
        
        # Save the result to a temporary file
        ts_file = os.path.splitext(python_file)[0] + "_converted.ts"
        with open(ts_file, 'w', encoding='utf-8') as f:
            f.write(ts_content)
        
        logger.info(f"Conversion test completed. Result saved to {ts_file}")
        return ts_file
    except Exception as e:
        logger.error(f"Error testing conversion: {e}")
        return None

def main():
    """Main function"""
    setup_args()
    
    # Read the source converter file
    try:
        with open(Config.SOURCE_CONVERTER_PATH, 'r', encoding='utf-8') as f:
            source_content = f.read()
    except Exception as e:
        logger.error(f"Error reading source file: {e}")
        sys.exit(1)
    
    # Apply improvements
    logger.info("Applying improvements to the converter...")
    
    # Update PATTERN_MAP
    logger.info("Updating pattern mapping...")
    updated_content = update_pattern_map(source_content)
    
    # Update CLASS_CONVERSIONS
    logger.info("Updating class templates...")
    updated_content = update_class_conversions(updated_content)
    
    # Enhance _generate_class_from_template method
    logger.info("Enhancing class generation method...")
    updated_content = enhance_generate_class_method(updated_content)
    
    # Enhance _extract_interfaces method
    logger.info("Enhancing interface extraction method...")
    updated_content = enhance_extract_interfaces(updated_content)
    
    # Enhance map_file_to_destination method
    logger.info("Enhancing file mapping method...")
    updated_content = enhance_map_file_to_destination(updated_content)
    
    # Improve _add_closing_braces method
    logger.info("Improving brace handling method...")
    updated_content = improve_add_closing_braces(updated_content)
    
    # Save the improved converter
    if Config.APPLY_CHANGES:
        output_path = Config.SOURCE_CONVERTER_PATH
        logger.info(f"Applying changes directly to {output_path}")
    else:
        output_path = Config.OUTPUT_CONVERTER_PATH
        logger.info(f"Saving improved converter to {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        logger.info("Improvements applied successfully!")
    except Exception as e:
        logger.error(f"Error saving improved converter: {e}")
        sys.exit(1)
    
    # Test the conversion if a test file is provided
    if Config.TEST_MODE and Config.TEST_FILE:
        logger.info(f"Testing conversion with file: {Config.TEST_FILE}")
        test_conversion(Config.TEST_FILE, output_path)
    
    logger.info("Done!")

# Direct conversion function for testing
def convert_file(input_path, output_path):
    """Convert a Python file to TypeScript directly"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            python_code = f.read()
        
        # Create interfaces
        interfaces = ConverterImprovements.create_improved_interfaces(python_code)
        
        # Apply all pattern conversions
        typescript_code = python_code
        for pattern, replacement in ConverterImprovements.IMPROVED_PATTERN_MAP:
            typescript_code = re.sub(pattern, replacement, typescript_code)
        
        # Add interfaces at the beginning
        typescript_code = interfaces + typescript_code
        
        # Fix import paths
        typescript_code = ConverterImprovements.fix_import_paths(
            python_code,
            typescript_code,
            input_path,
            os.path.dirname(output_path)
        )
        
        # Add TypeScript header
        header = '/**\n'
        header += f' * Converted from Python: {os.path.basename(input_path)}\n'
        header += f' * Conversion date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        header += ' * Generated with improved Python-to-TypeScript converter\n'
        header += ' */\n\n'
        
        # Write the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + typescript_code)
        
        return True
    except Exception as e:
        logger.error(f"Error converting file: {e}")
        return False

if __name__ == "__main__":
    main()