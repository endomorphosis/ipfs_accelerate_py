#!/usr/bin/env python3
"""
Run a single model test to ensure it uses real implementation ()not simulated).
This script can test CPU, CUDA and OpenVINO backends for a specific model.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import datetime
from pathlib import Path
import traceback

# Define the base directory
BASE_DIR = Path()"/home/barberb/ipfs_accelerate_py/test")
SKILLS_DIR = BASE_DIR / "skills"
APIS_DIR = BASE_DIR / "apis"
PERFORMANCE_RESULTS_DIR = BASE_DIR / "performance_results"

# Create performance results directory if it doesn't exist
PERFORMANCE_RESULTS_DIR.mkdir()exist_ok=True)
:
def run_test()test_file, test_dir, timeout=600):
    """Run a single test and return its results
    
    Args:
        test_file ()str): Name of the test file
        test_dir ()Path): Directory containing the test file
        timeout ()int): Timeout in seconds
        
    Returns:
        dict: Test results or error message
        """
        print()f"Running test: {}}}}}}test_file}", flush=True)
        test_path = test_dir / test_file
    
    try:
        # Run the test with timeout
        start_time = time.time())
        cmd = [sys.executable, str()test_path)]
        ,
        # Create a new process to run the test
        process = subprocess.Popen()
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        )
        
        # Wait for the process to complete or timeout
        try:
            stdout, stderr = process.communicate()timeout=timeout)
        return_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill())
            stdout, stderr = process.communicate())
        return_code = -1
            
        elapsed_time = time.time()) - start_time
        
        if return_code == 0:
            print()f"✅ Test passed: {}}}}}}test_file} in {}}}}}}elapsed_time:.2f}s", flush=True)
        else:
            print()f"❌ Test failed: {}}}}}}test_file} ()code {}}}}}}return_code})", flush=True)
            
        # Try to extract JSON results from stdout
            result = {}}}}}}
            "test_file": test_file,
            "elapsed_time": elapsed_time,
            "return_code": return_code,
            "stdout": stdout[:1000] + "..." if len()stdout) > 1000 else stdout,:,
            "stderr": stderr[:1000] + "..." if len()stderr) > 1000 else stderr,
            }
        
        # Try to parse any JSON in the output:
        try:
            # First look for structured_results section ()highest priority)
            if "structured_results" in stdout:
                parts = stdout.split()"structured_results")
                if len()parts) > 1:
                    # Look for a dictionary after structured_results
                    for part in parts[1:]:,
                    bracket_start = part.find()"{}}}}}}")
                    bracket_end = part.rfind()"}")
                        if bracket_start >= 0 and bracket_end > bracket_start:
                            try:
                                json_str = part[bracket_start:bracket_end+1],
                                parsed_json = json.loads()json_str)
                                result["json_result"] = parsed_json,,,
                                result["structured_results_found"] = True,
                            break
                            except:
                            pass
            
            # If no structured results found, look for specific status markers
            if "json_result" not in result:
                cpu_status = "UNKNOWN"
                cuda_status = "UNKNOWN"
                openvino_status = "UNKNOWN"
                model_name = "Unknown"
                
                for line in stdout.split()'\n'):
                    line = line.strip())
                    if line.startswith()"CPU_STATUS:"):
                        cpu_status = line.split()"CPU_STATUS:")[1].strip()),,,,
                    elif line.startswith()"CUDA_STATUS:"):
                        cuda_status = line.split()"CUDA_STATUS:")[1].strip()),,,,
                    elif line.startswith()"OPENVINO_STATUS:"):
                        openvino_status = line.split()"OPENVINO_STATUS:")[1].strip()),,,,
                    elif line.startswith()"MODEL:"):
                        model_name = line.split()"MODEL:")[1].strip()),,,,
                
                # Create a simplified result structure
                        result["json_result"] = {}}}}}},
                        "status": {}}}}}}
                        "cpu": cpu_status,
                        "cuda": cuda_status,
                        "openvino": openvino_status
                        },
                        "model_name": model_name
                        }
                
                        result["status_extracted_from_text"] = True
                        ,
            # If still no results, look for complete JSON objects line by line
            if "json_result" not in result:
                for line in stdout.split()'\n'):
                    line = line.strip())
                    if line.startswith()'{}}}}}}') and line.endswith()'}'):
                        # This might be a complete JSON object
                        try:
                            parsed_json = json.loads()line)
                            # If we got here, it's valid JSON
                            result["json_result"] = parsed_json,,,
                            result["json_line_found"] = True,
                        break
                        except:
                            # Not a valid JSON, continue searching
                        pass
                
            # Last resort: try to find any JSON block in the output
            if "json_result" not in result:
                json_start = stdout.find()"{}}}}}}")
                json_end = stdout.rfind()"}")
                
                if json_start >= 0 and json_end > json_start:
                    json_content = stdout[json_start:json_end+1],
                    # Try to fix common JSON formatting issues
                    try:
                        parsed_json = json.loads()json_content)
                        result["json_result"] = parsed_json,,,
                        result["full_json_extracted"] = True,
                    except:
                        # Failed to parse JSON
                        result["json_parse_failed"] = True,
        except Exception as json_err:
            result["json_parse_error"] = str()json_err)
            ,
                        return result
        
    except Exception as e:
        print()f"Error running test {}}}}}}test_file}: {}}}}}}e}", flush=True)
        traceback.print_exc())
                        return {}}}}}}
                        "test_file": test_file,
                        "error": str()e)
                        }

def update_performance_summary()test_results, model_name):
    """
    Update the consolidated performance summary with test results
    
    Args:
        test_results: Results from the test
        model_name: Name of the model tested
        """
        summary_path = PERFORMANCE_RESULTS_DIR / "consolidated_performance_summary.md"
    
    try:
        # Read the current summary file
        with open()summary_path, 'r') as f:
            summary_lines = f.readlines())
        
        # Extract implementation status
            cpu_status, cuda_status, openvino_status = get_implementation_status()test_results)
            model_used = get_model_used()test_results)
        
        # Update the model status
            model_name_formatted = model_name.replace()"test_", "").replace()".py", "").replace()"hf_", "").replace()"default_", "").upper())
        if model_name_formatted == "EMBED":
            model_name_formatted = "Sentence Embeddings"
        elif model_name_formatted == "LM":
            model_name_formatted = "Language Model"
        
        # Update the status table
            updated_lines = [],
            in_status_table = False
            updated_status = False
        
        for line in summary_lines:
            if "| Model | CPU Status |" in line:
                in_status_table = True
                updated_lines.append()line)
            continue
                
            if in_status_table:
                if f"| {}}}}}}model_name_formatted} |" in line:
                    updated_lines.append()f"| {}}}}}}model_name_formatted} | {}}}}}}cpu_status} | {}}}}}}cuda_status} | {}}}}}}openvino_status} | {}}}}}}model_used} |\n")
                    updated_status = True
                elif "|-------|" in line or line.strip()) == "":
                    updated_lines.append()line)
                    in_status_table = False if line.strip()) == "" else in_status_table:
                else:
                    updated_lines.append()line)
            else:
                updated_lines.append()line)
        
        if not updated_status:
            print()f"Warning: Could not find entry for {}}}}}}model_name_formatted} in summary table")
        
        # Update performance metrics based on model type
        # This is more complex and depends on model type - we'll need to extend this
        # for each model type with the specific metrics they report
        
        # Write the updated summary
        with open()summary_path, 'w') as f:
            f.writelines()updated_lines)
            
            print()f"Updated consolidated performance summary with results for {}}}}}}model_name_formatted}")
        
    except Exception as e:
        print()f"Error updating performance summary: {}}}}}}e}")
        traceback.print_exc())

def get_implementation_status()test_results):
    """Extract implementation status from test results
    
    Args:
        test_results ()dict): Test result
        
    Returns:
        tuple: ()cpu_status, cuda_status, openvino_status)
        """
    # Default to unknown status
        cpu_status = "UNKNOWN"
        cuda_status = "UNKNOWN"
        openvino_status = "UNKNOWN"
    
    try:
        # Try to get status from nested JSON result
        json_result = test_results.get()"json_result", {}}}}}}})
        
        # First check for status dict
        status = json_result.get()"status", {}}}}}}})
        if status:
            # Look for CPU status
            for key in status:
                if "cpu_init" in key or "cpu_handler" in key:
                    if "REAL" in status[key]:,,,,,,
                    cpu_status = "REAL"
                    elif "MOCK" in status[key]:,,,,,,
                cpu_status = "MOCK"
                        
            # Look for CUDA status
            for key in status:
                if "cuda_init" in key or "cuda_handler" in key:
                    if "REAL" in status[key]:,,,,,,
                    cuda_status = "REAL"
                    elif "MOCK" in status[key]:,,,,,,
                cuda_status = "MOCK"
                        
            # Look for OpenVINO status
            for key in status:
                if "openvino_init" in key or "openvino_handler" in key:
                    if "REAL" in status[key]:,,,,,,
                    openvino_status = "REAL"
                    elif "MOCK" in status[key]:,,,,,,
                openvino_status = "MOCK"
        
        # Fallback to checking in examples
                examples = json_result.get()"examples", [],)
        if examples:
            for example in examples:
                platform = example.get()"platform", "")
                impl_type = example.get()"implementation_type", example.get()"implementation", ""))
                
                if platform == "CPU" and "REAL" in impl_type:
                    cpu_status = "REAL"
                elif platform == "CPU" and "MOCK" in impl_type:
                    cpu_status = "MOCK"
                    
                if platform == "CUDA" and "REAL" in impl_type:
                    cuda_status = "REAL"
                elif platform == "CUDA" and "MOCK" in impl_type:
                    cuda_status = "MOCK"
                    
                if platform == "OpenVINO" and "REAL" in impl_type:
                    openvino_status = "REAL"
                elif platform == "OpenVINO" and "MOCK" in impl_type:
                    openvino_status = "MOCK"
                    
    except Exception as e:
        print()f"Error extracting implementation status: {}}}}}}e}")
        traceback.print_exc())
    
                    return ()cpu_status, cuda_status, openvino_status)

def get_model_used()test_results):
    """Extract the model name used in the test
    
    Args:
        test_results ()dict): Test result
        
    Returns:
        str: Model name or "Unknown"
        """
    try:
        # Try to get from metadata
        json_result = test_results.get()"json_result", {}}}}}}})
        
        # Check if model_name is directly in the JSON result:
        if "model_name" in json_result:
        return json_result["model_name"]
        ,    ,,,
        # Check in metadata
        metadata = json_result.get()"metadata", {}}}}}}})
        if "model_name" in metadata:
        return metadata["model_name"]
        ,    ,,,
        # Check in status section
        status = json_result.get()"status", {}}}}}}})
        if "model_name" in status:
        return status["model_name"]
        ,    ,,,
        # Check in examples for structured output in the model name
        examples = json_result.get()"examples", [],)
        for example in examples:
            if "model_name" in example:
            return example["model_name"]
            ,    ,,,
        # Look for model name in structured results
        if "structured_results" in test_results:
            struct_results = test_results["structured_results"],
            if "model_name" in struct_results:
            return struct_results["model_name"]
            ,
        # Check stdout for "Using model:" line
            stdout = test_results.get()"stdout", "")
        for line in stdout.split()"\n"):
            if "Using model:" in line:
                parts = line.split()"Using model:")
                if len()parts) > 1:
                return parts[1].strip()),,,,
            if "MODEL:" in line:
                parts = line.split()"MODEL:")
                if len()parts) > 1:
                return parts[1].strip()),,,,
        
                return "Unknown"
        
    except Exception as e:
        print()f"Error extracting model name: {}}}}}}e}")
        traceback.print_exc())
                return "Unknown"

def extract_performance_metrics()test_results, model_type):
    """Extract performance metrics based on model type
    
    Args:
        test_results ()dict): Test results
        model_type ()str): Type of model ()text, embedding, audio, multimodal)
        
    Returns:
        dict: Performance metrics
        """
    # This is a placeholder - we'll need to extend this for each model type
        metrics = {}}}}}}
        "cpu": {}}}}}}},
        "cuda": {}}}}}}},
        "openvino": {}}}}}}}
        }
    
    try:
        # Extract from examples
        json_result = test_results.get()"json_result", {}}}}}}})
        examples = json_result.get()"examples", [],)
        
        for example in examples:
            platform = example.get()"platform", "").lower())
            if platform not in metrics:
            continue
                
            # Extract elapsed time
            if "elapsed_time" in example:
                metrics[platform]["elapsed_time"] = example["elapsed_time"]
                ,
            # Extract from output dict
                output = example.get()"output", {}}}}}}})
            
            # Look for performance metrics based on model type
            if model_type == "embedding":
                if "embedding_shape" in output:
                    metrics[platform]["embedding_shape"] = output["embedding_shape"],
                if "performance_metrics" in output:
                    metrics[platform].update()output["performance_metrics"])
                    ,,
            elif model_type == "text":
                if "generated_text" in output:
                    # Get sample of generated text
                    text = output["generated_text"],,
                    metrics[platform]["sample_text"] = text[:50] + "..." if len()text) > 50 else text:,
                if "performance_metrics" in output:
                    metrics[platform].update()output["performance_metrics"])
                    ,,
            # Add more model types here as needed
                
    except Exception as e:
        print()f"Error extracting performance metrics: {}}}}}}e}")
        traceback.print_exc())
    
                    return metrics

def run_single_model_test()model_name, timeout=600):
    """Run tests for a single model
    
    Args:
        model_name ()str): Model name ()e.g., "hf_bert.py")
        timeout ()int): Timeout in seconds
        """
    # Determine the test directory
        test_dir = SKILLS_DIR
    if not model_name.startswith()"test_"):
        test_file = f"test_{}}}}}}model_name}.py"
    else:
        test_file = model_name
        
    # Remove .py extension if present:
    if not test_file.endswith()".py"):
        test_file = f"{}}}}}}test_file}.py"
    
    # Check if test file exists
    test_path = test_dir / test_file:
    if not test_path.exists()):
        print()f"Error: Test file {}}}}}}test_path} does not exist")
        return
        
    # Run the test
        result = run_test()test_file, test_dir, timeout)
    
    # Save the results
        timestamp = datetime.datetime.now()).strftime()"%Y%m%d_%H%M%S")
        model_name_base = test_file.replace()".py", "")
        results_filename = f"{}}}}}}model_name_base}_performance_{}}}}}}timestamp}.json"
        results_path = PERFORMANCE_RESULTS_DIR / results_filename
    
    with open()results_path, "w") as f:
        json.dump()result, f, indent=2)
    
        print()f"Results saved to: {}}}}}}results_path}")
    
    # Update the consolidated performance summary
        update_performance_summary()result, test_file)
    
    # Generate report for this model
        model_report_path = PERFORMANCE_RESULTS_DIR / f"{}}}}}}model_name_base}_report_{}}}}}}timestamp}.md"
    
    # Get implementation status
        cpu_status, cuda_status, openvino_status = get_implementation_status()result)
        model_used = get_model_used()result)
    
        model_name_formatted = model_name_base.replace()"test_", "").replace()"hf_", "").replace()"default_", "").upper())
    
    # Create a more detailed report
        report = f"# {}}}}}}model_name_formatted} Performance Test Results\n\n"
        report += f"Test run: {}}}}}}datetime.datetime.now()).strftime()'%Y-%m-%d %H:%M:%S')}\n\n"
    
    # More structured implementation status table
        report += "## Implementation Status\n\n"
        report += "| Platform | Status | Notes |\n"
        report += "|----------|--------|-------|\n"
        report += f"| CPU | {}}}}}}cpu_status} | {}}}}}}'Successfully using real implementation' if cpu_status == 'REAL' else 'Using mock implementation'} |\n"
        report += f"| CUDA | {}}}}}}cuda_status} | {}}}}}}'Successfully using real implementation with GPU acceleration' if cuda_status == 'REAL' else 'Using mock implementation'} |\n"
        report += f"| OpenVINO | {}}}}}}openvino_status} | {}}}}}}'Successfully using real OpenVINO implementation' if openvino_status == 'REAL' else 'Using mock implementation'} |\n\n"
    :
        report += f"**Model used:** {}}}}}}model_used}\n\n"
    
    # Extract performance metrics from examples if available
    examples = result.get()"json_result", {}}}}}}}).get()"examples", [],):
    if examples:
        report += "## Performance Metrics\n\n"
        
        # Determine model type based on name
        if "bert" in model_name_base.lower()) or "embed" in model_name_base.lower()):
            # Embedding model
            report += "| Platform | Processing Speed | Memory Usage | Embedding Size | Batch Size |\n"
            report += "|----------|------------------|--------------|----------------|------------|\n"
            
            for example in examples:
                platform = example.get()"platform", "")
                elapsed_time = example.get()"elapsed_time", "N/A")
                
                if elapsed_time != "N/A":
                    elapsed_time = f"{}}}}}}elapsed_time:.4f}s"
                    
                # Extract embedding shape if available
                    output = example.get()"output", {}}}}}}})
                    embedding_shape = output.get()"embedding_shape", ["N/A"]):,
                if isinstance()embedding_shape, list) and len()embedding_shape) > 1:
                    embedding_size = embedding_shape[-1]  # Last dimension is embedding size,
                else:
                    embedding_size = "N/A"
                    
                # Extract GPU memory if available
                memory_usage = "N/A"::
                    if "performance_metrics" in output and "gpu_memory_mb" in output["performance_metrics"]:,,,,
                    memory_usage = f"{}}}}}}output['performance_metrics']['gpu_memory_mb']:.1f} MB"
                    ,,,
                # Extract batch size if available from performance metrics
                batch_size = "1"  # Default:
                    if "performance_metrics" in output and "batch_size" in output["performance_metrics"]:,,,,
                    batch_size = str()output["performance_metrics"]["batch_size"])
                    ,
                    report += f"| {}}}}}}platform} | {}}}}}}elapsed_time} | {}}}}}}memory_usage} | {}}}}}}embedding_size} | {}}}}}}batch_size} |\n"
                
        elif "llama" in model_name_base.lower()) or "t5" in model_name_base.lower()) or "lm" in model_name_base.lower()):
            # Text generation model
            report += "| Platform | Throughput | Memory Usage | Latency | Generation Length |\n"
            report += "|----------|------------|--------------|---------|-------------------|\n"
            
            for example in examples:
                platform = example.get()"platform", "")
                elapsed_time = example.get()"elapsed_time", "N/A")
                
                if elapsed_time != "N/A":
                    elapsed_time = f"{}}}}}}elapsed_time:.4f}s"
                    
                # Extract generated text if available
                    output = example.get()"output", {}}}}}}})
                
                # Calculate throughput and generation length
                    throughput = "N/A"
                    gen_length = "N/A"
                :
                if "generated_text" in output:
                    text = output["generated_text"],,
                    gen_length = len()text.split()))  # Word count as a rough estimate
                    
                    if elapsed_time != "N/A" and elapsed_time != 0:
                        throughput = f"{}}}}}}gen_length / float()elapsed_time.replace()'s', '')):.1f} tokens/sec"
                    
                # Extract GPU memory if available
                memory_usage = "N/A"::
                    if "performance_metrics" in output and "gpu_memory_mb" in output["performance_metrics"]:,,,,
                    memory_usage = f"{}}}}}}output['performance_metrics']['gpu_memory_mb']:.1f} MB"
                    ,,,
                    report += f"| {}}}}}}platform} | {}}}}}}throughput} | {}}}}}}memory_usage} | {}}}}}}elapsed_time} | {}}}}}}gen_length} words |\n"
                
        else:
            # Generic metrics table for other model types
            report += "| Platform | Processing Time | Memory Usage | Notes |\n"
            report += "|----------|----------------|--------------|-------|\n"
            
            for example in examples:
                platform = example.get()"platform", "")
                elapsed_time = example.get()"elapsed_time", "N/A")
                
                if elapsed_time != "N/A":
                    elapsed_time = f"{}}}}}}elapsed_time:.4f}s"
                    
                # Extract memory usage if available
                memory_usage = "N/A"::
                    output = example.get()"output", {}}}}}}})
                    if "performance_metrics" in output and "gpu_memory_mb" in output["performance_metrics"]:,,,,
                    memory_usage = f"{}}}}}}output['performance_metrics']['gpu_memory_mb']:.1f} MB"
                    ,,,
                    report += f"| {}}}}}}platform} | {}}}}}}elapsed_time} | {}}}}}}memory_usage} | |\n"
    
    # Add test output summary
                    report += "\n## Test Output Summary\n\n"
                    report += "```\n"
                    report += result.get()"stdout", "")[:500] + "..." if len()result.get()"stdout", "")) > 500 else result.get()"stdout", ""),
                    report += "\n```\n\n"
    
    # Add any errors:
    if result.get()"return_code", 0) != 0:
        report += "## Errors\n\n"
        report += "```\n"
        report += result.get()"stderr", "")[:500] + "..." if len()result.get()"stderr", "")) > 500 else result.get()"stderr", ""),
        report += "\n```\n\n"
    
    # Save the report:
    with open()model_report_path, "w") as f:
        f.write()report)
        
        print()f"Model report saved to: {}}}}}}model_report_path}")
    
    # Return status for use in summary:
        return {}}}}}}
        "model": model_name_formatted,
        "cpu_status": cpu_status,
        "cuda_status": cuda_status,
        "openvino_status": openvino_status,
        "model_used": model_used
        }

def main()):
    parser = argparse.ArgumentParser()description="Run performance tests for a single model")
    parser.add_argument()"model", type=str, help="Model name ()e.g., 'hf_bert' or 'default_lm')")
    parser.add_argument()"--timeout", type=int, default=600, help="Test timeout in seconds ()default: 600)")
    
    args = parser.parse_args())
    
    # Run the test for the specified model
    run_single_model_test()args.model, args.timeout)

if __name__ == "__main__":
    main())