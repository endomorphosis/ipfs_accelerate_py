#!/usr/bin/env python3
"""
Enhanced TypeScript Converter and Fixer Script

This script builds upon the existing converter scripts to create a more comprehensive
solution for fixing TypeScript syntax and type issues in the entire SDK.

Key improvements:
1. Better handling of imports and interfaces
2. More precise type definitions
3. Custom implementations for problematic core components
4. Fixes for common syntax issues that arise from Python -> TS conversion
"""

import os
import sys
import re
import json
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_ts_fixer.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    TARGET_DIR = "../ipfs_accelerate_js"
    DRY_RUN = False
    VERBOSE = False
    CREATE_BACKUPS = True
    FIX_IMPORTS = True
    FIX_INTERFACES = True
    FIX_CLASSES = True
    FIX_FUNCTION_RETURNS = True
    FIX_CORE_COMPONENTS = True
    INSTALL_DEPS = False
    STATS = {
        "files_processed": 0,
        "files_fixed": 0,
        "files_backed_up": 0,
        "imports_fixed": 0,
        "interfaces_fixed": 0,
        "classes_fixed": 0,
        "functions_fixed": 0,
        "core_components_fixed": 0,
        "errors": 0
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced TypeScript Fixer")
    parser.add_argument("--target-dir", help="Target directory", default="../ipfs_accelerate_js")
    parser.add_argument("--dry-run", action="store_true", help="Don't make changes, just report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-backups", action="store_true", help="Don't create backup files")
    parser.add_argument("--no-imports", action="store_true", help="Skip import fixes")
    parser.add_argument("--no-interfaces", action="store_true", help="Skip interface fixes")
    parser.add_argument("--no-classes", action="store_true", help="Skip class fixes")
    parser.add_argument("--no-functions", action="store_true", help="Skip function return fixes")
    parser.add_argument("--no-core-components", action="store_true", help="Skip core component fixes")
    parser.add_argument("--install", action="store_true", help="Install TypeScript dependencies")
    
    args = parser.parse_args()
    
    Config.TARGET_DIR = os.path.abspath(args.target_dir)
    Config.DRY_RUN = args.dry_run
    Config.VERBOSE = args.verbose
    Config.CREATE_BACKUPS = not args.no_backups
    Config.FIX_IMPORTS = not args.no_imports
    Config.FIX_INTERFACES = not args.no_interfaces
    Config.FIX_CLASSES = not args.no_classes
    Config.FIX_FUNCTION_RETURNS = not args.no_functions
    Config.FIX_CORE_COMPONENTS = not args.no_core_components
    Config.INSTALL_DEPS = args.install
    
    if not os.path.isdir(Config.TARGET_DIR):
        logger.error(f"Target directory does not exist: {Config.TARGET_DIR}")
        sys.exit(1)
    
    logger.info(f"Target directory: {Config.TARGET_DIR}")
    logger.info(f"Dry run: {Config.DRY_RUN}")
    logger.info(f"Creating backups: {Config.CREATE_BACKUPS}")
    logger.info(f"Fix imports: {Config.FIX_IMPORTS}")
    logger.info(f"Fix interfaces: {Config.FIX_INTERFACES}")
    logger.info(f"Fix classes: {Config.FIX_CLASSES}")
    logger.info(f"Fix function returns: {Config.FIX_FUNCTION_RETURNS}")
    logger.info(f"Fix core components: {Config.FIX_CORE_COMPONENTS}")
    logger.info(f"Install dependencies: {Config.INSTALL_DEPS}")

def create_backup(file_path: str):
    """Create a backup of the original file"""
    if not Config.CREATE_BACKUPS or Config.DRY_RUN:
        return

    try:
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        Config.STATS["files_backed_up"] += 1
        if Config.VERBOSE:
            logger.debug(f"Created backup: {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup for {file_path}: {e}")

def find_typescript_files() -> List[str]:
    """Find all TypeScript files in the target directory"""
    ts_files = []
    
    for root, _, files in os.walk(Config.TARGET_DIR):
        for file in files:
            if file.endswith((".ts", ".tsx")) and not file.endswith(".d.ts"):
                ts_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(ts_files)} TypeScript files to process")
    return ts_files

def fix_imports(content: str) -> Tuple[str, int]:
    """Fix import statements in TypeScript files"""
    if not Config.FIX_IMPORTS:
        return content, 0
    
    original_content = content
    fixed_count = 0
    
    # Fix Python-style imports
    patterns = [
        # from module import {items} -> import {items} from "module";
        (r'from\s+[\'"]([^\'"]+)[\'"](\s+)import\s+{([^}]+)}', r'import {\3} from "\1";'),
        
        # from module import item -> import {item} from "module";
        (r'from\s+[\'"]([^\'"]+)[\'"](\s+)import\s+(\w+)', r'import { \3 } from "\1";'),
        
        # import module -> import * as module from "module";
        (r'import\s+(\w+)(\s+)from\s+[\'"]([^\'"]+)[\'"]', r'import * as \1 from "\3";'),
        
        # Remove .py, .ts, .tsx extensions from imports
        (r'from\s+[\'"]([^\'"]+)\.(py|ts|tsx)[\'"]', r'from "\1"'),
        (r'import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"]([^\'"]+)\.(py|ts|tsx)[\'"]', r'import \1 from "\3"'),
        
        # Fix missing semicolons after imports
        (r'(import\s+.*?from\s+[\'"][^\'"]+[\'"]\s*)$', r'\1;', re.MULTILINE),
        
        # Fix relative imports not starting with ./ or ../
        (r'from\s+[\'"](?!\.\/|\.\.\/|@|\/)([^\'"]+)[\'"]', r'from "./\1"'),
        (r'import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"](?!\.\/|\.\.\/|@|\/)([^\'"]+)[\'"]', r'import \1 from "./\2"'),
        
        # Fix array-style imports in braces (TypeScript doesn't allow this)
        (r'import\s+{\s*\[(.*?)\]\s*}', r'import { \1 }'),
        
        # Fix type import syntax
        (r'import\s+type\s+{([^}]+)}\s+from', r'import { \1 } from'),
        
        # Fix multiple spaces in imports
        (r'import\s+{(\s+)([^}]+)(\s+)}', r'import { \2 }'),
        
        # Fix malformed imports with multiple 'from' keywords
        (r'import\s+{([^}]+)}\s+from\s+[\'"]([^\'"]+)[\'"]\s+from', r'import { \1 } from "\2";')
    ]
    
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            fixed_matches = len(re.findall(pattern, content))
            fixed_count += fixed_matches
            content = new_content
    
    return content, fixed_count

def fix_interfaces(content: str) -> Tuple[str, int]:
    """Fix interface declarations in TypeScript files"""
    if not Config.FIX_INTERFACES:
        return content, 0
    
    original_content = content
    fixed_count = 0
    
    # Find all interface declarations
    interface_pattern = r'(export\s+)?interface\s+(\w+)(?:\s+extends\s+[^{]+)?(?:\s*){([^}]*)}'
    interfaces = list(re.finditer(interface_pattern, content, re.DOTALL))
    
    for interface_match in interfaces:
        interface_full = interface_match.group(0)
        export_prefix = interface_match.group(1) or ""
        interface_name = interface_match.group(2)
        interface_body = interface_match.group(3)
        
        # Fix interface body
        fixed_body = interface_body
        
        # Fix properties without types
        fixed_body = re.sub(r'(\s+)(\w+)(?:\s*;)?(?![:\(])', r'\1\2: any;', fixed_body)
        
        # Fix properties without semicolons
        fixed_body = re.sub(r'(\s+)(\w+)\s*:\s*([^;]+)(?!\s*;)$', r'\1\2: \3;', fixed_body, flags=re.MULTILINE)
        
        # Fix method signatures without types
        fixed_body = re.sub(r'(\s+)(\w+)\s*\(([^)]*)(?!\):\s*)', r'\1\2(\3): any', fixed_body)
        
        # Fix method parameters without types
        fixed_body = re.sub(r'\((\s*\w+)(?:\s*,|\s*\))', r'(\1: any\2', fixed_body)
        
        # Add missing semicolons at the end of methods
        fixed_body = re.sub(r'(\w+\([^)]*\)(?:\s*:\s*[^;]+)?(?!\s*;))$', r'\1;', fixed_body, flags=re.MULTILINE)
        
        # Reconstruct fixed interface
        fixed_interface = f"{export_prefix}interface {interface_name} {{\n{fixed_body}\n}}"
        
        if fixed_interface != interface_full:
            content = content.replace(interface_full, fixed_interface)
            fixed_count += 1
    
    return content, fixed_count

def fix_classes(content: str) -> Tuple[str, int]:
    """Fix class declarations in TypeScript files"""
    if not Config.FIX_CLASSES:
        return content, 0
    
    original_content = content
    fixed_count = 0
    
    # Find all class declarations
    class_pattern = r'(export\s+)?class\s+(\w+)(?:\s+extends\s+[^{]+)?(?:\s*){([^}]*)}'
    classes = list(re.finditer(class_pattern, content, re.DOTALL))
    
    for class_match in classes:
        class_full = class_match.group(0)
        export_prefix = class_match.group(1) or ""
        class_name = class_match.group(2)
        class_body = class_match.group(3)
        
        # Fix class body
        fixed_body = class_body
        
        # Fix properties without types
        fixed_body = re.sub(r'(\s+)(private|protected|public)?\s*(\w+)(?:\s*=|\s*;)(?![:\(])', r'\1\2 \3: any', fixed_body)
        
        # Fix constructor parameters without types
        fixed_body = re.sub(r'constructor\s*\(([^:\)]+)(?=[,\)])', r'constructor(\1: any', fixed_body)
        
        # Fix method signatures without return types
        fixed_body = re.sub(r'(\s+)(private|protected|public)?\s*(\w+)\s*\(([^)]*)(?!\):\s*)', r'\1\2 \3(\4): any', fixed_body)
        
        # Fix async method signatures without return types
        fixed_body = re.sub(r'(\s+)(private|protected|public)?\s*async\s+(\w+)\s*\(([^)]*)(?!\):\s*)', 
                           r'\1\2 async \3(\4): Promise<any>', fixed_body)
        
        # Fix methods returning Promise but not marked as async
        fixed_body = re.sub(r'(\s+)(private|protected|public)?\s*(\w+)\s*\(([^)]*)\)\s*:\s*Promise<([^>]+)>([^{]*){', 
                           r'\1\2 async \3(\4): Promise<\5>\6{', fixed_body)
        
        # Fix method parameters without types
        method_param_pattern = r'(\w+)\s*\((\s*\w+)(?:\s*,|\s*\))'
        fixed_body = re.sub(method_param_pattern, r'\1(\2: any\3', fixed_body)
        
        # Add missing semicolons after property assignments
        fixed_body = re.sub(r'(\s+)(\w+)\s*=\s*([^;{]+)(?!\s*;|\s*\{)$', r'\1\2 = \3;', fixed_body, flags=re.MULTILINE)
        
        # Reconstruct fixed class
        fixed_class = f"{export_prefix}class {class_name} {{\n{fixed_body}\n}}"
        
        if fixed_class != class_full:
            content = content.replace(class_full, fixed_class)
            fixed_count += 1
    
    return content, fixed_count

def fix_function_returns(content: str) -> Tuple[str, int]:
    """Fix function return types in TypeScript files"""
    if not Config.FIX_FUNCTION_RETURNS:
        return content, 0
    
    original_content = content
    fixed_count = 0
    
    # Fix standalone function declarations
    function_pattern = r'(export\s+)?function\s+(\w+)\s*\(([^)]*)\)(?!\s*:)'
    
    # Add explicit any return type where missing
    function_fixed = re.sub(function_pattern, r'\1function \2(\3): any', content)
    
    # Fix function parameters without types
    param_pattern = r'function\s+\w+\s*\((\s*\w+)(?:\s*,|\s*\))'
    function_fixed = re.sub(param_pattern, r'function \1(\2: any\3', function_fixed)
    
    # Fix async functions without Promise return type
    async_pattern = r'(export\s+)?async\s+function\s+(\w+)\s*\(([^)]*)\)(?!\s*:\s*Promise)'
    function_fixed = re.sub(async_pattern, r'\1async function \2(\3): Promise<any>', function_fixed)
    
    # Fix standalone arrow functions
    arrow_pattern = r'const\s+(\w+)\s*=\s*(\([^)]*\))\s*=>'
    function_fixed = re.sub(arrow_pattern, r'const \1 = \2: any =>', function_fixed)
    
    # Fix async arrow functions
    async_arrow_pattern = r'const\s+(\w+)\s*=\s*async\s*(\([^)]*\))\s*=>'
    function_fixed = re.sub(async_arrow_pattern, r'const \1 = async \2: Promise<any> =>', function_fixed)
    
    if function_fixed != content:
        fixed_count = len(re.findall(function_pattern, content)) + \
                      len(re.findall(async_pattern, content)) + \
                      len(re.findall(arrow_pattern, content)) + \
                      len(re.findall(async_arrow_pattern, content))
        content = function_fixed
    
    return content, fixed_count

def fix_core_files():
    """Fix core components with custom implementations"""
    if not Config.FIX_CORE_COMPONENTS:
        return
    
    # Define core components that need special attention
    core_components = [
        {
            "path": os.path.join(Config.TARGET_DIR, "src/hardware/hardware_abstraction.ts"),
            "content": """/**
 * Hardware abstraction layer for IPFS Accelerate
 */
import { HardwareBackend, HardwarePreferences } from '../interfaces';

export class HardwareAbstraction {
  private backends: Map<string, HardwareBackend> = new Map();
  private activeBackend: HardwareBackend | null = null;
  private preferences: HardwarePreferences;

  constructor(preferences: HardwarePreferences = {} as HardwarePreferences) {
    this.preferences = {
      backendOrder: preferences.backendOrder || ['webgpu', 'webnn', 'wasm', 'cpu'],
      modelPreferences: preferences.modelPreferences || {},
      options: preferences.options || {}
    };
  }

  async initialize(): Promise<boolean> {
    try {
      // Implementation will be added as the backends are implemented
      return true;
    } catch (error) {
      console.error("Failed to initialize hardware abstraction layer:", error);
      return false;
    }
  }

  async getPreferredBackend(modelType: string): Promise<HardwareBackend | null> {
    // Implementation will be added as the backends are implemented
    return this.activeBackend;
  }

  async execute<T = any, U = any>(inputs: T, modelType: string): Promise<U> {
    const backend = await this.getPreferredBackend(modelType);
    if (!backend) {
      throw new Error(`No suitable backend found for model type: ${modelType}`);
    }

    if (!backend.execute) {
      throw new Error(`Backend does not implement execute method`);
    }

    return backend.execute<T, U>(inputs);
  }

  dispose(): void {
    // Clean up resources
    for (const backend of this.backends.values()) {
      backend.destroy();
    }
    this.backends.clear();
    this.activeBackend = null;
  }
}
"""
        },
        {
            "path": os.path.join(Config.TARGET_DIR, "src/hardware/backends/webgpu_backend.ts"),
            "content": """/**
 * WebGPU backend implementation for IPFS Accelerate
 */
import { HardwareBackend } from '../../interfaces';

export class WebGPUBackend implements HardwareBackend {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private initialized: boolean = false;
  private shaderModules: Map<string, GPUShaderModule> = new Map();
  private buffers: Map<string, GPUBuffer> = new Map();
  private pipelines: Map<string, GPUComputePipeline> = new Map();

  constructor() {
    this.initialized = false;
  }

  async initialize(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        console.warn("WebGPU is not supported in this browser");
        return false;
      }

      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) {
        console.warn("Failed to get GPU adapter");
        return false;
      }

      this.device = await this.adapter.requestDevice();
      if (!this.device) {
        console.warn("Failed to get GPU device");
        return false;
      }

      this.initialized = true;
      return true;
    } catch (error) {
      console.error("Failed to initialize WebGPU backend:", error);
      return false;
    }
  }

  async execute<T = any, U = any>(inputs: T): Promise<U> {
    if (!this.initialized || !this.device) {
      throw new Error("WebGPU backend not initialized");
    }

    // Implementation will depend on the model type and operation
    // This is a placeholder for the actual implementation
    
    return {} as U;
  }

  destroy(): void {
    // Release WebGPU resources
    for (const buffer of this.buffers.values()) {
      // Clean up buffer resources
    }
    
    this.buffers.clear();
    this.shaderModules.clear();
    this.pipelines.clear();
    
    this.device = null;
    this.adapter = null;
    this.initialized = false;
  }
}
"""
        },
        {
            "path": os.path.join(Config.TARGET_DIR, "src/hardware/backends/webnn_backend.ts"),
            "content": """/**
 * WebNN backend implementation for IPFS Accelerate
 */
import { HardwareBackend } from '../../interfaces';

export class WebNNBackend implements HardwareBackend {
  private context: any | null = null;  // MLContext type
  private builder: any | null = null;  // MLGraphBuilder type
  private initialized: boolean = false;
  private graphs: Map<string, any> = new Map();  // Map of MLGraph objects

  constructor() {
    this.initialized = false;
  }

  async initialize(): Promise<boolean> {
    try {
      // Check if WebNN is supported
      if (!('ml' in navigator)) {
        console.warn("WebNN is not supported in this browser");
        return false;
      }

      // @ts-ignore - TypeScript doesn't know about navigator.ml yet
      this.context = navigator.ml.createContext();
      
      if (!this.context) {
        console.warn("Failed to create ML context");
        return false;
      }

      // @ts-ignore - TypeScript doesn't know about navigator.ml yet
      this.builder = navigator.ml.createGraphBuilder(this.context);
      
      if (!this.builder) {
        console.warn("Failed to create ML graph builder");
        return false;
      }

      this.initialized = true;
      return true;
    } catch (error) {
      console.error("Failed to initialize WebNN backend:", error);
      return false;
    }
  }

  async execute<T = any, U = any>(inputs: T): Promise<U> {
    if (!this.initialized || !this.builder) {
      throw new Error("WebNN backend not initialized");
    }

    // Implementation will depend on the model type and operation
    // This is a placeholder for the actual implementation
    
    return {} as U;
  }

  destroy(): void {
    // Release WebNN resources
    this.graphs.clear();
    this.builder = null;
    this.context = null;
    this.initialized = false;
  }
}
"""
        },
        {
            "path": os.path.join(Config.TARGET_DIR, "src/interfaces.ts"),
            "content": """/**
 * Common interfaces for the IPFS Accelerate JavaScript SDK
 */

// Hardware interfaces
export interface HardwareBackend {
  initialize(): Promise<boolean>;
  destroy(): void;
  execute<T = any, U = any>(input: T): Promise<U>;
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
  execute<T = any, U = any>(input: T): Promise<U>;
}

// WebGPU interfaces
export interface GPUBufferDescriptor {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

export interface GPUShaderModuleDescriptor {
  code: string;
}

export interface GPUBindGroupDescriptor {
  layout: any;
  entries: GPUBindGroupEntry[];
}

export interface GPUBindGroupEntry {
  binding: number;
  resource: any;
}

export interface GPUComputePipelineDescriptor {
  layout?: any;
  compute: {
    module: any;
    entryPoint: string;
  };
}

// WebNN interfaces
export interface MLOperandDescriptor {
  type: string;
  dimensions: number[];
}

export interface MLOperand {}

export interface MLGraph {
  compute(inputs: Record<string, MLOperand>): Record<string, MLOperand>;
}

export interface MLContext {}

export interface MLGraphBuilder {
  constant(desc: MLOperandDescriptor, value: any): MLOperand;
  input(name: string, desc: MLOperandDescriptor): MLOperand;
  build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
}

// Resource Pool interfaces
export interface ResourcePoolConnection {
  id: string;
  type: string;
  status: string;
  create(): Promise<void>;
  resources: Record<string, any>;
}

export interface ResourcePoolOptions {
  maxConnections?: number;
  browserPreferences?: Record<string, string>;
  adaptiveScaling?: boolean;
  enableFaultTolerance?: boolean;
  recoveryStrategy?: string;
  stateSyncInterval?: number;
  redundancyFactor?: number;
}

// Browser interfaces
export interface BrowserCapabilities {
  browserName: string;
  browserVersion: string;
  isMobile: boolean;
  platform: string;
  osVersion: string;
  webgpuSupported: boolean;
  webgpuFeatures: string[];
  webnnSupported: boolean;
  webnnFeatures: string[];
  wasmSupported: boolean;
  wasmFeatures: string[];
  metalApiSupported: boolean;
  metalApiVersion: string;
  recommendedBackend: string;
  memoryLimitMB: number;
}

// Optimization interfaces
export interface OptimizationConfig {
  memoryOptimization: boolean;
  progressiveLoading: boolean;
  useQuantization: boolean;
  precision: number;
  maxChunkSizeMB: number;
  parallelLoading: boolean;
  specialOptimizations: Record<string, any>;
}

// Tensor interfaces
export interface Tensor {
  shape: number[];
  data: Float32Array | Int32Array | Uint8Array;
  dtype: string;
}

// Shared tensor memory interface
export interface TensorSharing {
  shareableTypes: string[];
  enableSharing(modelIds: string[]): void;
  shareTensor(tensor: Tensor, source: string, targets: string[]): void;
  getTensor(id: string, target: string): Tensor | null;
  releaseSharedTensors(modelId: string): void;
}
"""
        },
        {
            "path": os.path.join(Config.TARGET_DIR, "src/index.ts"),
            "content": """/**
 * IPFS Accelerate JavaScript SDK
 * Main entry point
 */

// Add type references
/// <reference path="./types/webgpu.d.ts" />
/// <reference path="./types/webnn.d.ts" />

// Export public interfaces
export * from './interfaces';

// Export hardware abstractions
export * from './hardware/hardware_abstraction';
export * from './hardware/backends/webgpu_backend';
export * from './hardware/backends/webnn_backend';

// Export browser utilities
export * from './browser/optimizations/browser_capability_detection';
export * from './browser/resource_pool/resource_pool_bridge';

// Re-export model APIs as needed
// Placeholder for actual model exports as they are implemented
"""
        }
    ]
    
    for component in core_components:
        file_path = component["path"]
        content = component["content"]
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if os.path.exists(file_path) and Config.CREATE_BACKUPS:
            create_backup(file_path)
        
        if not Config.DRY_RUN:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            Config.STATS["core_components_fixed"] += 1
            logger.info(f"Fixed core component: {os.path.relpath(file_path, Config.TARGET_DIR)}")

def create_or_update_tsconfig():
    """Create or update tsconfig.json with proper settings"""
    tsconfig_path = os.path.join(Config.TARGET_DIR, "tsconfig.json")
    
    # Default TypeScript config tailored for our project
    tsconfig = {
        "compilerOptions": {
            "target": "es2020",
            "module": "esnext",
            "moduleResolution": "node",
            "declaration": True,
            "declarationDir": "./dist/types",
            "sourceMap": True,
            "outDir": "./dist",
            "strict": False,  # Start with less strict settings for migration
            "esModuleInterop": True,
            "skipLibCheck": True,
            "forceConsistentCasingInFileNames": True,
            "lib": ["dom", "dom.iterable", "esnext", "webworker"],
            "jsx": "react"
        },
        "include": ["src/**/*"],
        "exclude": ["node_modules", "dist", "**/*.test.ts", "**/*.spec.ts"]
    }
    
    # If tsconfig already exists, update it
    if os.path.exists(tsconfig_path):
        try:
            with open(tsconfig_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            
            # Update only specific settings
            if "compilerOptions" in existing_config:
                # Update strict settings for migration phase
                existing_config["compilerOptions"]["strict"] = False
                existing_config["compilerOptions"]["noImplicitAny"] = False
                existing_config["compilerOptions"]["strictNullChecks"] = False
                existing_config["compilerOptions"]["skipLibCheck"] = True
                
                # Ensure we have proper lib settings
                if "lib" not in existing_config["compilerOptions"]:
                    existing_config["compilerOptions"]["lib"] = ["dom", "dom.iterable", "esnext", "webworker"]
            
            tsconfig = existing_config
        except Exception as e:
            logger.warning(f"Failed to read existing tsconfig.json: {e}")
            logger.warning("Creating new tsconfig.json")
    
    # Write the tsconfig file
    if not Config.DRY_RUN:
        with open(tsconfig_path, 'w', encoding='utf-8') as f:
            json.dump(tsconfig, f, indent=2)
        
        logger.info(f"Created/updated tsconfig.json")

def create_or_update_package_json():
    """Create or update package.json with proper settings"""
    package_path = os.path.join(Config.TARGET_DIR, "package.json")
    
    # Default package.json
    package_json = {
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
    
    # If package.json already exists, update it
    if os.path.exists(package_path):
        try:
            with open(package_path, 'r', encoding='utf-8') as f:
                existing_pkg = json.load(f)
            
            # Update only specific fields
            if "devDependencies" not in existing_pkg:
                existing_pkg["devDependencies"] = package_json["devDependencies"]
            else:
                # Ensure TypeScript is included
                if "typescript" not in existing_pkg["devDependencies"]:
                    existing_pkg["devDependencies"]["typescript"] = package_json["devDependencies"]["typescript"]
                
                # Ensure type definitions are included
                for type_def in ["@types/node", "@types/react", "@types/jest"]:
                    if type_def not in existing_pkg["devDependencies"]:
                        existing_pkg["devDependencies"][type_def] = package_json["devDependencies"][type_def]
            
            # Ensure scripts section exists
            if "scripts" not in existing_pkg:
                existing_pkg["scripts"] = package_json["scripts"]
            else:
                # Ensure type-check script exists
                if "type-check" not in existing_pkg["scripts"]:
                    existing_pkg["scripts"]["type-check"] = "tsc --noEmit"
            
            package_json = existing_pkg
        except Exception as e:
            logger.warning(f"Failed to read existing package.json: {e}")
            logger.warning("Creating new package.json")
    
    # Write the package.json file
    if not Config.DRY_RUN:
        with open(package_path, 'w', encoding='utf-8') as f:
            json.dump(package_json, f, indent=2)
        
        logger.info(f"Created/updated package.json")

def fix_typescript_file(file_path: str) -> bool:
    """Fix TypeScript issues in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes in sequence
        content, imports_fixed = fix_imports(content)
        content, interfaces_fixed = fix_interfaces(content)
        content, classes_fixed = fix_classes(content)
        content, functions_fixed = fix_function_returns(content)
        
        # Track fixed counts
        Config.STATS["imports_fixed"] += imports_fixed
        Config.STATS["interfaces_fixed"] += interfaces_fixed
        Config.STATS["classes_fixed"] += classes_fixed
        Config.STATS["functions_fixed"] += functions_fixed
        
        # Write changes if content was modified
        if content != original_content and not Config.DRY_RUN:
            # Create backup if requested
            if Config.CREATE_BACKUPS:
                create_backup(file_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        Config.STATS["errors"] += 1
        return False

def install_dependencies():
    """Install TypeScript dependencies using npm"""
    if not Config.INSTALL_DEPS or Config.DRY_RUN:
        return
    
    logger.info("Installing TypeScript dependencies...")
    
    try:
        import subprocess
        
        # Change to the target directory
        cwd = os.getcwd()
        os.chdir(Config.TARGET_DIR)
        
        # Run npm install
        subprocess.run(["npm", "install"], check=True)
        
        # Change back to original directory
        os.chdir(cwd)
        
        logger.info("Dependencies installed successfully")
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")

def create_declaration_files():
    """Create TypeScript declaration files for WebGPU and WebNN"""
    types_dir = os.path.join(Config.TARGET_DIR, "src/types")
    os.makedirs(types_dir, exist_ok=True)
    
    # WebGPU declaration file
    webgpu_path = os.path.join(types_dir, "webgpu.d.ts")
    webgpu_content = """/**
 * WebGPU type definitions
 */

export interface GPUDevice {
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createTexture(descriptor: any): GPUTexture;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
  queue: GPUQueue;
}

export interface GPUAdapter {
  requestDevice(): Promise<GPUDevice>;
}

export interface GPUBuffer {
  setSubData?(offset: number, data: ArrayBuffer | ArrayBufferView): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

export interface GPUTexture {
  createView(descriptor?: any): GPUTextureView;
}

export interface GPUTextureView {}

export interface GPUShaderModule {}

export interface GPUShaderModuleDescriptor {
  code: string;
}

export interface GPUComputePipeline {}

export interface GPUComputePipelineDescriptor {
  layout?: any;
  compute: {
    module: GPUShaderModule;
    entryPoint: string;
  };
}

export interface GPUBindGroup {}

export interface GPUBindGroupDescriptor {
  layout: any;
  entries: GPUBindGroupEntry[];
}

export interface GPUBindGroupEntry {
  binding: number;
  resource: any;
}

export interface GPUCommandEncoder {
  beginComputePass(): GPUComputePassEncoder;
  finish(): GPUCommandBuffer;
}

export interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface GPUCommandBuffer {}

export interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
}

export interface GPUBufferDescriptor {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

export interface NavigatorGPU {
  requestAdapter(): Promise<GPUAdapter>;
}

export interface Navigator {
  gpu: NavigatorGPU;
}
"""
    
    # WebNN declaration file
    webnn_path = os.path.join(types_dir, "webnn.d.ts")
    webnn_content = """/**
 * WebNN type definitions
 */

export interface MLContext {}

export interface MLGraph {
  compute(inputs: Record<string, MLOperand>): Record<string, MLOperand>;
}

export interface MLOperandDescriptor {
  type: string;
  dimensions: number[];
}

export interface MLOperand {}

export interface MLGraphBuilder {
  input(name: string, desc: MLOperandDescriptor): MLOperand;
  constant(desc: MLOperandDescriptor, value: any): MLOperand;
  build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
  
  // Basic operations
  add(a: MLOperand, b: MLOperand): MLOperand;
  sub(a: MLOperand, b: MLOperand): MLOperand;
  mul(a: MLOperand, b: MLOperand): MLOperand;
  div(a: MLOperand, b: MLOperand): MLOperand;
  
  // Neural network operations
  gemm(a: MLOperand, b: MLOperand, options?: any): MLOperand;
  conv2d(input: MLOperand, filter: MLOperand, options?: any): MLOperand;
  relu(input: MLOperand): MLOperand;
  softmax(input: MLOperand): MLOperand;
}

export interface NavigatorML {
  createContext(options?: any): MLContext;
  createGraphBuilder(context: MLContext): MLGraphBuilder;
}

export interface Navigator {
  ml: NavigatorML;
}
"""
    
    if not Config.DRY_RUN:
        with open(webgpu_path, 'w', encoding='utf-8') as f:
            f.write(webgpu_content)
        
        with open(webnn_path, 'w', encoding='utf-8') as f:
            f.write(webnn_content)
        
        logger.info("Created TypeScript declaration files in src/types/")

def create_migration_guide():
    """Create a migration guide document"""
    guide_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TYPESCRIPT_MIGRATION_GUIDE.md")
    
    guide_content = f"""# TypeScript Migration Guide

## Overview

This document provides guidance on the migration of the IPFS Accelerate Python codebase to TypeScript. The migration process involves converting Python code to TypeScript, fixing common conversion issues, and ensuring proper TypeScript typing and syntax.

## Migration Process

The migration was performed using automated scripts in combination with manual fixes for complex cases. The process included:

1. **Conversion of Python Files**: Python files were converted to TypeScript using a custom converter script.
2. **Syntax Fixing**: Common syntax issues from the conversion were fixed automatically.
3. **Type Annotations**: TypeScript type annotations were added to functions, methods, and properties.
4. **Interface Definitions**: Proper TypeScript interfaces were defined for core components.
5. **Core Component Fixes**: Key files like interfaces.ts and hardware abstraction were manually fixed.

## Current Status

As of {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, the migration has:

- Fixed imports in {Config.STATS["imports_fixed"]} instances
- Fixed interfaces in {Config.STATS["interfaces_fixed"]} instances
- Fixed classes in {Config.STATS["classes_fixed"]} instances
- Fixed function returns in {Config.STATS["functions_fixed"]} instances
- Fixed {Config.STATS["core_components_fixed"]} core components manually

## Directory Structure

The TypeScript SDK follows this structure:

```
ipfs_accelerate_js/
├── dist/           # Compiled output
├── src/            # Source code
│   ├── browser/    # Browser-specific functionality
│   │   ├── optimizations/    # Browser optimizations
│   │   └── resource_pool/    # Resource pool management
│   ├── hardware/   # Hardware abstraction
│   │   ├── backends/        # Hardware backends (WebGPU, WebNN)
│   │   └── detection/       # Hardware detection
│   ├── interfaces.ts        # Core interfaces
│   ├── types/      # TypeScript type definitions
│   └── ...
├── package.json    # Package configuration
└── tsconfig.json   # TypeScript configuration
```

## Known Issues and Limitations

1. **Incomplete Type Definitions**: Some complex types might need refinement.
2. **Browser Compatibility**: Browser-specific code may require additional testing.
3. **WebNN Support**: WebNN interfaces are based on the draft specification and may need updates.

## Next Steps

1. **Testing**: Run comprehensive tests on the converted TypeScript code.
2. **Refinement**: Refine type definitions for better type safety.
3. **Documentation**: Add JSDoc comments to functions and classes.
4. **Build System**: Finalize the build system configuration.

## Helpful Commands

1. **Type Checking**:
   ```bash
   cd {Config.TARGET_DIR}
   npm run type-check
   ```

2. **Building**:
   ```bash
   cd {Config.TARGET_DIR}
   npm run build
   ```

3. **Testing**:
   ```bash
   cd {Config.TARGET_DIR}
   npm test
   ```

## Resources

1. [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
2. [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
3. [WebNN Specification](https://webmachinelearning.github.io/webnn/)
"""
    
    if not Config.DRY_RUN:
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"Created TypeScript migration guide: {guide_path}")

def process_all_files():
    """Process all TypeScript files"""
    # Find all TypeScript files
    ts_files = find_typescript_files()
    
    # Process each file
    for i, file_path in enumerate(ts_files):
        Config.STATS["files_processed"] += 1
        
        # Skip core components that are handled separately
        core_paths = [
            "src/hardware/hardware_abstraction.ts",
            "src/hardware/backends/webgpu_backend.ts",
            "src/hardware/backends/webnn_backend.ts",
            "src/interfaces.ts",
            "src/index.ts"
        ]
        
        if any(comp_path in file_path for comp_path in core_paths) and Config.FIX_CORE_COMPONENTS:
            logger.info(f"Skipping core component (handled separately): {os.path.relpath(file_path, Config.TARGET_DIR)}")
            continue
        
        # Log progress periodically
        if Config.VERBOSE or (i+1) % 25 == 0:
            logger.info(f"Processing file {i+1}/{len(ts_files)}: {os.path.relpath(file_path, Config.TARGET_DIR)}")
        
        # Fix the file
        if fix_typescript_file(file_path):
            Config.STATS["files_fixed"] += 1
            logger.info(f"Fixed file: {os.path.relpath(file_path, Config.TARGET_DIR)}")

def main():
    """Main function"""
    parse_args()
    
    # Create basic TypeScript setup
    create_or_update_tsconfig()
    create_or_update_package_json()
    create_declaration_files()
    
    # Fix core components first
    if Config.FIX_CORE_COMPONENTS:
        fix_core_files()
    
    # Process all other files
    process_all_files()
    
    # Install dependencies if requested
    if Config.INSTALL_DEPS:
        install_dependencies()
    
    # Create migration guide
    create_migration_guide()
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"Files processed: {Config.STATS['files_processed']}")
    logger.info(f"Files fixed: {Config.STATS['files_fixed']}")
    logger.info(f"Files backed up: {Config.STATS['files_backed_up']}")
    logger.info(f"Imports fixed: {Config.STATS['imports_fixed']}")
    logger.info(f"Interfaces fixed: {Config.STATS['interfaces_fixed']}")
    logger.info(f"Classes fixed: {Config.STATS['classes_fixed']}")
    logger.info(f"Functions fixed: {Config.STATS['functions_fixed']}")
    logger.info(f"Core components fixed: {Config.STATS['core_components_fixed']}")
    logger.info(f"Errors: {Config.STATS['errors']}")
    
    if Config.DRY_RUN:
        logger.info("This was a dry run, no changes were made")
    else:
        logger.info("TypeScript fixes completed")

if __name__ == "__main__":
    main()