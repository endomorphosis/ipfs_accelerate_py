#!/usr/bin/env python3
# test_improved_converter.py
# Comprehensive test script for the improved Python to TypeScript converter

import os
import sys
import re
import subprocess
import logging
import argparse
from typing import Dict, List, Tuple
from improve_py_to_ts_converter import ConverterImprovements, convert_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('improved_converter_test.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    SAMPLE_FILE = None
    COMPARISON = False
    VERIFY_TYPESCRIPT = False
    TEST_DIRECTORY = None
    OUTPUT_DIRECTORY = None
    VERBOSE = False

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test improved Python to TypeScript converter")
    parser.add_argument("--sample", help="Path to sample Python file", default="sample_webgpu_backend.py")
    parser.add_argument("--compare", action="store_true", help="Compare with original converter")
    parser.add_argument("--verify", action="store_true", help="Verify TypeScript output")
    parser.add_argument("--test-dir", help="Directory with Python files to test")
    parser.add_argument("--output-dir", help="Output directory for TypeScript files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    Config.SAMPLE_FILE = os.path.abspath(args.sample) if args.sample else None
    Config.COMPARISON = args.compare
    Config.VERIFY_TYPESCRIPT = args.verify
    Config.TEST_DIRECTORY = os.path.abspath(args.test_dir) if args.test_dir else None
    Config.OUTPUT_DIRECTORY = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    Config.VERBOSE = args.verbose
    
    # Create output directory if it doesn't exist
    if Config.OUTPUT_DIRECTORY:
        os.makedirs(Config.OUTPUT_DIRECTORY, exist_ok=True)
    
    logger.info(f"Testing converter with:")
    logger.info(f"  Sample file: {Config.SAMPLE_FILE}")
    logger.info(f"  Comparison: {Config.COMPARISON}")
    logger.info(f"  Verify TypeScript: {Config.VERIFY_TYPESCRIPT}")
    logger.info(f"  Test directory: {Config.TEST_DIRECTORY}")
    logger.info(f"  Output directory: {Config.OUTPUT_DIRECTORY}")

def run_original_converter(input_file: str, output_file: str) -> bool:
    """Run the original Python to TypeScript converter"""
    try:
        # Use the setup_ipfs_accelerate_js_py_converter.py script
        converter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup_ipfs_accelerate_js_py_converter.py")
        
        if not os.path.exists(converter_path):
            logger.error(f"Original converter not found: {converter_path}")
            return False
        
        # Import the converter module
        import importlib.util
        spec = importlib.util.spec_from_file_location("original_converter", converter_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert using the original converter
        try:
            ts_content = module.PyToTsConverter.convert_py_to_ts(content, input_file)
            
            # Write the output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(ts_content)
                
            return True
        except Exception as e:
            logger.error(f"Error using original converter: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to run original converter: {e}")
        return False

def compare_converters(input_file: str) -> Dict[str, any]:
    """Compare the original and improved converters"""
    # Generate output file paths
    original_output = os.path.join(Config.OUTPUT_DIRECTORY, os.path.basename(input_file).replace(".py", "_original.ts"))
    improved_output = os.path.join(Config.OUTPUT_DIRECTORY, os.path.basename(input_file).replace(".py", "_improved.ts"))
    
    # Run both converters
    original_success = run_original_converter(input_file, original_output)
    improved_success = convert_file(input_file, improved_output)
    
    if not original_success or not improved_success:
        logger.error("Conversion failed for one or both converters")
        return {
            "original_success": original_success,
            "improved_success": improved_success,
            "comparison": None
        }
    
    # Read both outputs
    with open(original_output, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    with open(improved_output, 'r', encoding='utf-8') as f:
        improved_content = f.read()
    
    # Compare metrics
    metrics = {
        "original": {
            "lines": len(original_content.split('\n')),
            "interfaces": len(re.findall(r'interface\s+\w+', original_content)),
            "typed_methods": len(re.findall(r'[a-zA-Z0-9_]+\([^)]*\)\s*:\s*[a-zA-Z0-9_<>|]+', original_content)),
            "typed_props": len(re.findall(r'[a-zA-Z0-9_]+\s*:\s*[a-zA-Z0-9_<>|]+\s*[;=]', original_content)),
            "any_types": len(re.findall(r':\s*any', original_content)),
            "string_types": len(re.findall(r':\s*string', original_content)),
            "number_types": len(re.findall(r':\s*number', original_content)),
            "boolean_types": len(re.findall(r':\s*boolean', original_content)),
            "array_types": len(re.findall(r':\s*\w+\[\]', original_content)),
            "typescript_errors": 0  # Will be set if verification is enabled
        },
        "improved": {
            "lines": len(improved_content.split('\n')),
            "interfaces": len(re.findall(r'interface\s+\w+', improved_content)),
            "typed_methods": len(re.findall(r'[a-zA-Z0-9_]+\([^)]*\)\s*:\s*[a-zA-Z0-9_<>|]+', improved_content)),
            "typed_props": len(re.findall(r'[a-zA-Z0-9_]+\s*:\s*[a-zA-Z0-9_<>|]+\s*[;=]', improved_content)),
            "any_types": len(re.findall(r':\s*any', improved_content)),
            "string_types": len(re.findall(r':\s*string', improved_content)),
            "number_types": len(re.findall(r':\s*number', improved_content)),
            "boolean_types": len(re.findall(r':\s*boolean', improved_content)),
            "array_types": len(re.findall(r':\s*\w+\[\]', improved_content)),
            "typescript_errors": 0  # Will be set if verification is enabled
        }
    }
    
    # Verify TypeScript if requested
    if Config.VERIFY_TYPESCRIPT:
        # Create temporary tsconfig.json
        tsconfig_path = os.path.join(Config.OUTPUT_DIRECTORY, "tsconfig.json")
        with open(tsconfig_path, 'w', encoding='utf-8') as f:
            f.write("""
{
  "compilerOptions": {
    "target": "es2020",
    "module": "esnext",
    "moduleResolution": "node",
    "strict": false,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "noEmit": true,
    "lib": ["dom", "dom.iterable", "esnext", "webworker"]
  },
  "include": ["*.ts"]
}
""")
        
        # Create type definition files
        webgpu_types = os.path.join(Config.OUTPUT_DIRECTORY, "webgpu.d.ts")
        with open(webgpu_types, 'w', encoding='utf-8') as f:
            f.write("""
interface GPUDevice {
  createBuffer(descriptor: any): GPUBuffer;
  createShaderModule(descriptor: any): GPUShaderModule;
  createComputePipeline(descriptor: any): GPUComputePipeline;
  createBindGroup(descriptor: any): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
  queue: GPUQueue;
}

interface GPUAdapter {
  requestDevice(): Promise<GPUDevice>;
  features: Set<string>;
  limits: any;
  get_preferred_format(): string;
}

interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  write_buffer(buffer: GPUBuffer, offset: number, data: any): void;
  on_submitted_work_done(): Promise<void>;
}

interface GPUBuffer {
  map_async(mode: number): Promise<void>;
  get_mapped_range(): ArrayBuffer;
  unmap(): void;
}

interface GPUShaderModule {}

interface GPUComputePipeline {}

interface GPUBindGroup {}

interface GPUCommandEncoder {
  begin_compute_pass(): GPUComputePassEncoder;
  finish(): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  set_pipeline(pipeline: GPUComputePipeline): void;
  set_bind_group(index: number, bindGroup: GPUBindGroup): void;
  dispatch_workgroups(...args: number[]): void;
  end(): void;
}

interface GPUCommandBuffer {}

interface NavigatorGPU {
  request_adapter(): Promise<GPUAdapter>;
  requestAdapter(): Promise<GPUAdapter>;
}

interface Navigator {
  gpu: NavigatorGPU;
}

declare var navigator: Navigator;
""")
        
        # Run TypeScript compiler on both files
        try:
            # Original file
            result = subprocess.run(
                ["npx", "tsc", "--noEmit", os.path.basename(original_output)],
                cwd=Config.OUTPUT_DIRECTORY,
                check=False,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Count errors
                error_count = result.stdout.count("error TS")
                metrics["original"]["typescript_errors"] = error_count
                if Config.VERBOSE:
                    logger.info(f"Original converter TypeScript errors ({error_count}):")
                    logger.info(result.stdout)
            
            # Improved file
            result = subprocess.run(
                ["npx", "tsc", "--noEmit", os.path.basename(improved_output)],
                cwd=Config.OUTPUT_DIRECTORY,
                check=False,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Count errors
                error_count = result.stdout.count("error TS")
                metrics["improved"]["typescript_errors"] = error_count
                if Config.VERBOSE:
                    logger.info(f"Improved converter TypeScript errors ({error_count}):")
                    logger.info(result.stdout)
        except Exception as e:
            logger.error(f"Failed to verify TypeScript: {e}")
    
    return {
        "original_success": original_success,
        "improved_success": improved_success,
        "comparison": metrics
    }

def test_sample_file():
    """Test the converter on the sample file"""
    if not Config.SAMPLE_FILE or not os.path.exists(Config.SAMPLE_FILE):
        logger.error(f"Sample file not found: {Config.SAMPLE_FILE}")
        return
    
    logger.info(f"Testing converter on sample file: {Config.SAMPLE_FILE}")
    
    if Config.COMPARISON:
        result = compare_converters(Config.SAMPLE_FILE)
        
        if result["comparison"]:
            metrics = result["comparison"]
            
            # Print comparison table
            print("\nComparison Results:")
            print("-------------------")
            print(f"{'Metric':<20} | {'Original':<10} | {'Improved':<10} | {'Diff':<10}")
            print(f"{'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
            
            for key in metrics["original"]:
                original = metrics["original"][key]
                improved = metrics["improved"][key]
                diff = improved - original
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                print(f"{key:<20} | {original:<10} | {improved:<10} | {diff_str:<10}")
            
            # Print summary
            print("\nSummary:")
            print(f"- Improved converter generated {metrics['improved']['interfaces'] - metrics['original']['interfaces']} more interfaces")
            print(f"- Improved converter has {metrics['improved']['typescript_errors'] - metrics['original']['typescript_errors']} more TypeScript errors")
            print(f"- Improved converter has {metrics['improved']['typed_methods'] - metrics['original']['typed_methods']} more typed methods")
            print(f"- Improved converter has {metrics['improved']['typed_props'] - metrics['original']['typed_props']} more typed properties")
            
            # Recommendation
            if metrics['improved']['typescript_errors'] < metrics['original']['typescript_errors']:
                print("\nRecommendation: The improved converter is better")
            elif metrics['improved']['typescript_errors'] > metrics['original']['typescript_errors']:
                print("\nRecommendation: The original converter has fewer TypeScript errors")
            else:
                better_typing = (metrics['improved']['typed_methods'] + metrics['improved']['typed_props']) > (metrics['original']['typed_methods'] + metrics['original']['typed_props'])
                if better_typing:
                    print("\nRecommendation: The improved converter has better type coverage")
                else:
                    print("\nRecommendation: Both converters perform similarly")
    else:
        # Just run the improved converter
        output_file = os.path.join(Config.OUTPUT_DIRECTORY, os.path.basename(Config.SAMPLE_FILE).replace(".py", "_improved.ts"))
        success = convert_file(Config.SAMPLE_FILE, output_file)
        
        if success:
            logger.info(f"Conversion successful! Output written to: {output_file}")
            
            # Print the first 20 lines of the output
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            print("\nFirst 20 lines of the generated TypeScript file:")
            print("----------------------------------------------")
            print("\n".join(lines[:20]))
            print("...")
            
            # Check for TypeScript errors if requested
            if Config.VERIFY_TYPESCRIPT:
                # Create temporary tsconfig.json
                tsconfig_path = os.path.join(Config.OUTPUT_DIRECTORY, "tsconfig.json")
                with open(tsconfig_path, 'w', encoding='utf-8') as f:
                    f.write("""
{
  "compilerOptions": {
    "target": "es2020",
    "module": "esnext",
    "moduleResolution": "node",
    "strict": false,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "noEmit": true,
    "lib": ["dom", "dom.iterable", "esnext", "webworker"]
  },
  "include": ["*.ts"]
}
""")
                
                # Create type definition files
                webgpu_types = os.path.join(Config.OUTPUT_DIRECTORY, "webgpu.d.ts")
                with open(webgpu_types, 'w', encoding='utf-8') as f:
                    f.write("""
interface GPUDevice {
  createBuffer(descriptor: any): GPUBuffer;
  createShaderModule(descriptor: any): GPUShaderModule;
  createComputePipeline(descriptor: any): GPUComputePipeline;
  createBindGroup(descriptor: any): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
  queue: GPUQueue;
}

interface GPUAdapter {
  requestDevice(): Promise<GPUDevice>;
  features: Set<string>;
  limits: any;
  get_preferred_format(): string;
}

interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  write_buffer(buffer: GPUBuffer, offset: number, data: any): void;
  on_submitted_work_done(): Promise<void>;
}

interface GPUBuffer {
  map_async(mode: number): Promise<void>;
  get_mapped_range(): ArrayBuffer;
  unmap(): void;
}

interface GPUShaderModule {}

interface GPUComputePipeline {}

interface GPUBindGroup {}

interface GPUCommandEncoder {
  begin_compute_pass(): GPUComputePassEncoder;
  finish(): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  set_pipeline(pipeline: GPUComputePipeline): void;
  set_bind_group(index: number, bindGroup: GPUBindGroup): void;
  dispatch_workgroups(...args: number[]): void;
  end(): void;
}

interface GPUCommandBuffer {}

interface NavigatorGPU {
  request_adapter(): Promise<GPUAdapter>;
  requestAdapter(): Promise<GPUAdapter>;
}

interface Navigator {
  gpu: NavigatorGPU;
}

declare var navigator: Navigator;
""")
                
                # Run TypeScript compiler
                try:
                    result = subprocess.run(
                        ["npx", "tsc", "--noEmit", os.path.basename(output_file)],
                        cwd=Config.OUTPUT_DIRECTORY,
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        logger.info("TypeScript verification passed!")
                    else:
                        error_count = result.stdout.count("error TS")
                        logger.warning(f"TypeScript verification failed with {error_count} errors")
                        if Config.VERBOSE:
                            logger.warning(result.stdout)
                except Exception as e:
                    logger.error(f"Failed to verify TypeScript: {e}")
        else:
            logger.error("Conversion failed")

def test_directory():
    """Test the converter on all Python files in a directory"""
    if not Config.TEST_DIRECTORY or not os.path.isdir(Config.TEST_DIRECTORY):
        logger.error(f"Test directory not found: {Config.TEST_DIRECTORY}")
        return
    
    logger.info(f"Testing converter on directory: {Config.TEST_DIRECTORY}")
    
    # Find all Python files
    python_files = []
    for root, _, files in os.walk(Config.TEST_DIRECTORY):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(python_files)} Python files to test")
    
    # Test each file
    results = []
    for i, file_path in enumerate(python_files):
        logger.info(f"Testing file {i+1}/{len(python_files)}: {file_path}")
        
        if Config.COMPARISON:
            result = compare_converters(file_path)
            results.append((file_path, result))
        else:
            # Just run the improved converter
            rel_path = os.path.relpath(file_path, Config.TEST_DIRECTORY)
            output_file = os.path.join(Config.OUTPUT_DIRECTORY, rel_path.replace(".py", ".ts"))
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            success = convert_file(file_path, output_file)
            results.append((file_path, {"improved_success": success}))
    
    # Print summary
    print("\nTest Results Summary:")
    print("---------------------")
    
    if Config.COMPARISON:
        # Count success and failures for both converters
        original_success = sum(1 for _, res in results if res["original_success"])
        original_failure = len(results) - original_success
        improved_success = sum(1 for _, res in results if res["improved_success"])
        improved_failure = len(results) - improved_success
        
        print(f"Original Converter: {original_success} successes, {original_failure} failures")
        print(f"Improved Converter: {improved_success} successes, {improved_failure} failures")
        
        # Compute average metrics
        all_metrics = [res["comparison"] for _, res in results if res["comparison"]]
        if all_metrics:
            avg_metrics = {
                "original": {},
                "improved": {}
            }
            
            for key in all_metrics[0]["original"]:
                avg_metrics["original"][key] = sum(m["original"][key] for m in all_metrics) / len(all_metrics)
                avg_metrics["improved"][key] = sum(m["improved"][key] for m in all_metrics) / len(all_metrics)
            
            # Print comparison table
            print("\nAverage Metrics:")
            print(f"{'Metric':<20} | {'Original':<10} | {'Improved':<10} | {'Diff':<10}")
            print(f"{'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
            
            for key in avg_metrics["original"]:
                original = avg_metrics["original"][key]
                improved = avg_metrics["improved"][key]
                diff = improved - original
                diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
                print(f"{key:<20} | {original:<10.1f} | {improved:<10.1f} | {diff_str:<10}")
            
            # Print summary
            print("\nSummary:")
            print(f"- Improved converter generates {avg_metrics['improved']['interfaces'] - avg_metrics['original']['interfaces']:.1f} more interfaces on average")
            print(f"- Improved converter has {avg_metrics['improved']['typescript_errors'] - avg_metrics['original']['typescript_errors']:.1f} more TypeScript errors on average")
            print(f"- Improved converter has {avg_metrics['improved']['typed_methods'] - avg_metrics['original']['typed_methods']:.1f} more typed methods on average")
            print(f"- Improved converter has {avg_metrics['improved']['typed_props'] - avg_metrics['original']['typed_props']:.1f} more typed properties on average")
            
            # Overall recommendation
            if avg_metrics['improved']['typescript_errors'] < avg_metrics['original']['typescript_errors']:
                print("\nRecommendation: The improved converter is better overall")
            elif avg_metrics['improved']['typescript_errors'] > avg_metrics['original']['typescript_errors']:
                print("\nRecommendation: The original converter has fewer TypeScript errors overall")
            else:
                better_typing = (avg_metrics['improved']['typed_methods'] + avg_metrics['improved']['typed_props']) > (avg_metrics['original']['typed_methods'] + avg_metrics['original']['typed_props'])
                if better_typing:
                    print("\nRecommendation: The improved converter has better type coverage overall")
                else:
                    print("\nRecommendation: Both converters perform similarly overall")
    else:
        success_count = sum(1 for _, res in results if res["improved_success"])
        failure_count = len(results) - success_count
        print(f"Improved Converter: {success_count} successes, {failure_count} failures ({success_count/len(results)*100:.1f}% success rate)")

def main():
    """Main function"""
    setup_args()
    
    if Config.TEST_DIRECTORY:
        test_directory()
    else:
        test_sample_file()

if __name__ == "__main__":
    main()