#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware Compatibility Matrix Generator for HuggingFace Models

This script tests HuggingFace models across different hardware platforms and
generates a compatibility matrix with performance metrics.

Features:
- Detects available hardware on the system (CPU, CUDA, MPS, OpenVINO, WebNN, WebGPU)
- Runs tests for representative models from each architecture family
- Collects performance metrics (load time, inference time, memory usage)
- Stores results in a structured format (JSON and DuckDB)
- Generates compatibility reports for each model-hardware combination
"""

import os
import sys
import json
import time
import argparse
import importlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Define hardware platforms
HARDWARE_PLATFORMS = [
    "cpu",
    "cuda",
    "mps",
    "openvino",
    "webnn",
    "webgpu",
]

# Representative models for each architecture
REPRESENTATIVE_MODELS = {
    "encoder-only": ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
    "decoder-only": ["gpt2", "facebook/opt-125m", "bigscience/bloom-560m"],
    "encoder-decoder": ["t5-small", "facebook/bart-base", "google/pegasus-xsum"],
    "vision": ["google/vit-base-patch16-224", "microsoft/swin-tiny-patch4-window7-224", "facebook/deit-base-patch16-224"],
    "multimodal": ["openai/clip-vit-base-patch32", "Salesforce/blip-image-captioning-base", "llava-hf/llava-1.5-7b-hf"],
    "audio": ["openai/whisper-tiny", "facebook/wav2vec2-base", "facebook/hubert-base-ls960"],
}

# Hardware detection functions
def check_cpu():
    """Always available but check for vectorization support"""
    return {
        "available": True,
        "name": platform.processor() or "Unknown CPU",
        "features": {
            "avx": check_cpu_feature("avx"),
            "avx2": check_cpu_feature("avx2"),
            "avx512": check_cpu_feature("avx512"),
            "cores": os.cpu_count(),
        }
    }

def check_cpu_feature(feature):
    """Check if CPU supports specific feature"""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            return feature in cpuinfo.lower()
        except:
            return False
    elif platform.system() == "Darwin":
        try:
            output = subprocess.check_output(["sysctl", "-a"]).decode()
            if feature == "avx":
                return "hw.optional.avx1_0" in output
            elif feature == "avx2":
                return "hw.optional.avx2_0" in output
            elif feature == "avx512":
                return "hw.optional.avx512f" in output
            return False
        except:
            return False
    return False

def check_cuda():
    """Check for CUDA availability"""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown CUDA Device"
            return {
                "available": True,
                "name": device_name,
                "features": {
                    "device_count": device_count,
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, "version") else "Unknown"
                }
            }
        return {"available": False}
    except:
        return {"available": False}

def check_mps():
    """Check for MPS (Metal Performance Shaders) availability on macOS"""
    try:
        import torch
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if available:
            return {
                "available": True,
                "name": "Apple Silicon MPS",
                "features": {
                    "device": "Apple Silicon",
                    "platform": platform.machine(),
                }
            }
        return {"available": False}
    except:
        return {"available": False}

def check_openvino():
    """Check for OpenVINO availability"""
    try:
        import openvino
        return {
            "available": True,
            "name": "Intel OpenVINO",
            "features": {
                "version": openvino.__version__,
            }
        }
    except:
        return {"available": False}

def check_webnn():
    """Check for WebNN availability (would be NA for native Python)"""
    # In a real implementation, this would be browser-dependent
    return {"available": False}

def check_webgpu():
    """Check for WebGPU availability (would be NA for native Python)"""
    # In a real implementation, this would be browser-dependent
    return {"available": False}

def detect_hardware():
    """Detect all available hardware on the system"""
    return {
        "cpu": check_cpu(),
        "cuda": check_cuda(),
        "mps": check_mps(),
        "openvino": check_openvino(),
        "webnn": check_webnn(),
        "webgpu": check_webgpu(),
        "system_info": {
            "platform": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "timestamp": datetime.now().isoformat(),
        }
    }

# Testing functions
def test_model_on_hardware(model_id, model_type, hardware, timeout=300):
    """Test a model on specified hardware and return performance metrics"""
    result = {
        "model_id": model_id,
        "model_type": model_type,
        "hardware": hardware,
        "success": False,
        "load_time": None,
        "inference_time": None,
        "memory_usage": None,
        "error": None,
    }
    
    try:
        # Import hardware-specific testing module dynamically
        # This would be a module like test_on_cpu, test_on_cuda, etc.
        module_name = f"test_on_{hardware}"
        try:
            testing_module = importlib.import_module(module_name)
        except ImportError:
            # Use built-in functionality
            from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoFeatureExtractor
            import torch
            
            # Start timer for model loading
            start_load = time.time()
            
            # Load appropriate model and processor based on model type
            if model_type in ["encoder-only", "decoder-only", "encoder-decoder"]:
                model = AutoModel.from_pretrained(model_id)
                processor = AutoTokenizer.from_pretrained(model_id)
                
                if hardware == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda")
                elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    model = model.to("mps")
                
                # Default input for text models
                inputs = processor("Testing the model performance across different hardware platforms.", return_tensors="pt")
                
            elif model_type == "vision":
                model = AutoModel.from_pretrained(model_id)
                processor = AutoFeatureExtractor.from_pretrained(model_id)
                
                if hardware == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda")
                elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    model = model.to("mps")
                
                # Default input for vision models (would be a dummy tensor)
                inputs = processor(torch.randn(3, 224, 224), return_tensors="pt")
                
            else:  # multimodal, audio
                # More specialized handling would be implemented here
                model = AutoModel.from_pretrained(model_id)
                processor = AutoProcessor.from_pretrained(model_id)
                
                if hardware == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda")
                elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    model = model.to("mps")
                
                # Default input would depend on model specifics
                inputs = processor("Testing the model performance across different hardware platforms.", return_tensors="pt")
            
            end_load = time.time()
            load_time = end_load - start_load
            
            # Measure inference time
            if hardware == "cuda" and torch.cuda.is_available():
                # Move inputs to CUDA
                inputs = {k: v.to("cuda") for k, v in inputs.items() if hasattr(v, "to")}
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = model(**inputs)
                torch.cuda.synchronize()
                start_inference = time.time()
                with torch.no_grad():
                    output = model(**inputs)
                torch.cuda.synchronize()
                end_inference = time.time()
            elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Move inputs to MPS
                inputs = {k: v.to("mps") for k, v in inputs.items() if hasattr(v, "to")}
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = model(**inputs)
                start_inference = time.time()
                with torch.no_grad():
                    output = model(**inputs)
                end_inference = time.time()
            else:
                # CPU inference
                start_inference = time.time()
                with torch.no_grad():
                    output = model(**inputs)
                end_inference = time.time()
            
            inference_time = end_inference - start_inference
            
            # Estimate memory usage (very rough approximation)
            if hardware == "cuda" and torch.cuda.is_available():
                memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            else:
                # Rough process memory estimate for CPU
                import psutil
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / (1024 ** 2)  # MB
            
            result.update({
                "success": True,
                "load_time": load_time,
                "inference_time": inference_time,
                "memory_usage": memory_usage,
                "output_shape": str(tuple(output[0].shape)) if hasattr(output, "shape") else str(type(output)),
            })
            
        return result
        
    except Exception as e:
        result["error"] = str(e)
        return result

def create_db_and_tables():
    """Create DuckDB database and tables for storing results"""
    if not DUCKDB_AVAILABLE:
        return None
    
    db_path = "hardware_compatibility_matrix.duckdb"
    conn = duckdb.connect(db_path)
    
    # Create results table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_results (
            id INTEGER PRIMARY KEY,
            model_id VARCHAR,
            model_type VARCHAR,
            hardware VARCHAR,
            success BOOLEAN,
            load_time DOUBLE,
            inference_time DOUBLE,
            memory_usage DOUBLE,
            error VARCHAR,
            timestamp TIMESTAMP,
            output_shape VARCHAR
        )
    """)
    
    # Create hardware detection table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_detection (
            id INTEGER PRIMARY KEY,
            hardware_type VARCHAR,
            available BOOLEAN,
            name VARCHAR,
            features VARCHAR,
            timestamp TIMESTAMP
        )
    """)
    
    return conn

def store_results_in_db(conn, results, hardware_info):
    """Store results in DuckDB database"""
    if not conn:
        return
    
    timestamp = datetime.now()
    
    # Store hardware detection results
    for hw_type, hw_data in hardware_info.items():
        if hw_type != "system_info":
            conn.execute("""
                INSERT INTO hardware_detection (hardware_type, available, name, features, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                hw_type,
                hw_data.get("available", False),
                hw_data.get("name", "Unknown"),
                json.dumps(hw_data.get("features", {})),
                timestamp
            ))
    
    # Store test results
    for result in results:
        conn.execute("""
            INSERT INTO hardware_results (
                model_id, model_type, hardware, success, load_time, 
                inference_time, memory_usage, error, timestamp, output_shape
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["model_id"],
            result["model_type"],
            result["hardware"],
            result["success"],
            result["load_time"],
            result["inference_time"],
            result["memory_usage"],
            result["error"],
            timestamp,
            result.get("output_shape", None)
        ))

def generate_compatibility_report(results, hardware_info, output_dir="compatibility_reports"):
    """Generate compatibility reports for models and hardware"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    with open(os.path.join(output_dir, "hardware_compatibility_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save hardware info
    with open(os.path.join(output_dir, "hardware_detection.json"), "w") as f:
        json.dump(hardware_info, f, indent=2)
    
    # Generate markdown summary
    with open(os.path.join(output_dir, "hardware_compatibility_summary.md"), "w") as f:
        f.write("# Hardware Compatibility Matrix\n\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System info
        f.write("## System Information\n\n")
        sys_info = hardware_info["system_info"]
        f.write(f"- **Platform**: {sys_info['platform']} {sys_info['release']} {sys_info['version']}\n")
        f.write(f"- **Machine**: {sys_info['machine']}\n")
        f.write(f"- **Python Version**: {sys_info['python_version']}\n\n")
        
        # Detected hardware
        f.write("## Detected Hardware\n\n")
        for hw_type, hw_data in hardware_info.items():
            if hw_type != "system_info":
                status = "✅ Available" if hw_data.get("available", False) else "❌ Not Available"
                f.write(f"### {hw_type.upper()}: {status}\n\n")
                if hw_data.get("available", False):
                    f.write(f"- **Name**: {hw_data.get('name', 'Unknown')}\n")
                    for feature, value in hw_data.get("features", {}).items():
                        f.write(f"- **{feature}**: {value}\n")
                f.write("\n")
        
        # Results by architecture
        f.write("## Results by Architecture\n\n")
        
        # Group results by architecture
        arch_results = {}
        for result in results:
            arch_type = result["model_type"]
            if arch_type not in arch_results:
                arch_results[arch_type] = []
            arch_results[arch_type].append(result)
        
        # Table for each architecture
        for arch_type, arch_data in arch_results.items():
            f.write(f"### {arch_type.title()} Architecture\n\n")
            
            # Create table
            f.write("| Model | Hardware | Status | Load Time (s) | Inference Time (s) | Memory (MB) |\n")
            f.write("|-------|----------|--------|--------------|-------------------|------------|\n")
            
            for result in arch_data:
                status = "✅ Success" if result["success"] else f"❌ Failed: {result['error']}"
                load_time = f"{result['load_time']:.3f}" if result["load_time"] else "N/A"
                inference_time = f"{result['inference_time']:.3f}" if result["inference_time"] else "N/A"
                memory = f"{result['memory_usage']:.1f}" if result["memory_usage"] else "N/A"
                
                f.write(f"| {result['model_id']} | {result['hardware']} | {status} | {load_time} | {inference_time} | {memory} |\n")
            
            f.write("\n")
    
    # Generate a separate performance analysis report
    with open(os.path.join(output_dir, "performance_analysis.md"), "w") as f:
        f.write("# Performance Analysis\n\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Get successful results
        successful_results = [r for r in results if r["success"] and r["inference_time"]]
        
        if not successful_results:
            f.write("No successful benchmarks to analyze.\n")
            return
        
        # Analyze by hardware
        f.write("## Performance by Hardware Platform\n\n")
        hw_performance = {}
        for result in successful_results:
            hw = result["hardware"]
            if hw not in hw_performance:
                hw_performance[hw] = []
            hw_performance[hw].append(result["inference_time"])
        
        f.write("| Hardware | Average Inference Time (s) | Min Time (s) | Max Time (s) |\n")
        f.write("|----------|----------------------------|--------------|-------------|\n")
        
        for hw, times in hw_performance.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            f.write(f"| {hw} | {avg_time:.4f} | {min_time:.4f} | {max_time:.4f} |\n")
        
        f.write("\n")
        
        # Analyze by model type
        f.write("## Performance by Model Architecture\n\n")
        model_performance = {}
        for result in successful_results:
            model_type = result["model_type"]
            if model_type not in model_performance:
                model_performance[model_type] = []
            model_performance[model_type].append(result["inference_time"])
        
        f.write("| Architecture | Average Inference Time (s) | Min Time (s) | Max Time (s) |\n")
        f.write("|--------------|----------------------------|--------------|-------------|\n")
        
        for model_type, times in model_performance.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            f.write(f"| {model_type} | {avg_time:.4f} | {min_time:.4f} | {max_time:.4f} |\n")
        
        f.write("\n")
        
        # Find fastest hardware for each model type
        f.write("## Optimal Hardware by Model Architecture\n\n")
        f.write("| Architecture | Recommended Hardware | Average Speedup Factor |\n")
        f.write("|--------------|----------------------|------------------------|\n")
        
        for model_type in set(result["model_type"] for result in successful_results):
            # Get results for this model type
            model_results = [r for r in successful_results if r["model_type"] == model_type]
            
            # Group by hardware
            hw_results = {}
            for r in model_results:
                hw = r["hardware"]
                if hw not in hw_results:
                    hw_results[hw] = []
                hw_results[hw].append(r["inference_time"])
            
            # Find hardware with lowest average time
            hw_avg_times = {hw: sum(times)/len(times) for hw, times in hw_results.items()}
            if not hw_avg_times:
                continue
                
            baseline_hw = "cpu"  # Compare against CPU as baseline
            baseline_time = hw_avg_times.get(baseline_hw)
            
            if not baseline_time:
                continue
                
            fastest_hw = min(hw_avg_times.items(), key=lambda x: x[1])
            speedup = baseline_time / fastest_hw[1] if fastest_hw[1] > 0 else 1.0
            
            f.write(f"| {model_type} | {fastest_hw[0]} | {speedup:.2f}x |\n")
        
        f.write("\n")
        
        # Memory usage analysis
        f.write("## Memory Usage Analysis\n\n")
        f.write("| Model | Hardware | Memory Usage (MB) |\n")
        f.write("|-------|----------|------------------|\n")
        
        # Sort by memory usage (highest first)
        memory_results = sorted(
            [r for r in successful_results if r["memory_usage"]],
            key=lambda x: x["memory_usage"],
            reverse=True
        )
        
        for result in memory_results[:20]:  # Show top 20
            f.write(f"| {result['model_id']} | {result['hardware']} | {result['memory_usage']:.1f} |\n")
        
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Hardware Compatibility Matrix Generator for HuggingFace Models")
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to test")
    parser.add_argument("--architectures", type=str, nargs="+", choices=REPRESENTATIVE_MODELS.keys(),
                       help="Model architectures to test")
    parser.add_argument("--hardware", type=str, nargs="+", choices=HARDWARE_PLATFORMS,
                       help="Hardware platforms to test")
    parser.add_argument("--output-dir", type=str, default="compatibility_reports",
                       help="Directory to store results and reports")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads for parallel testing")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds for each model test")
    parser.add_argument("--detect-only", action="store_true",
                       help="Only detect hardware without running tests")
    args = parser.parse_args()
    
    # Detect available hardware
    print("Detecting available hardware...")
    hardware_info = detect_hardware()
    
    # Print available hardware
    for hw_type, hw_data in hardware_info.items():
        if hw_type != "system_info":
            status = "Available" if hw_data.get("available", False) else "Not Available"
            print(f"{hw_type.upper()}: {status}")
            if hw_data.get("available", False):
                print(f"  - Name: {hw_data.get('name', 'Unknown')}")
                for feature, value in hw_data.get("features", {}).items():
                    print(f"  - {feature}: {value}")
    
    if args.detect_only:
        print("\nHardware detection complete. Exiting without running tests.")
        return
    
    # Determine which hardware platforms to test
    test_hardware = args.hardware if args.hardware else [
        hw for hw, data in hardware_info.items() 
        if hw != "system_info" and data.get("available", False)
    ]
    print(f"\nTesting on hardware platforms: {', '.join(test_hardware)}")
    
    # Determine which models to test
    if args.models:
        # Test specific models provided by user
        test_models = [(model_id, "unknown") for model_id in args.models]
    else:
        # Test representative models for selected architectures
        arch_to_test = args.architectures if args.architectures else REPRESENTATIVE_MODELS.keys()
        test_models = []
        for arch in arch_to_test:
            for model_id in REPRESENTATIVE_MODELS[arch]:
                test_models.append((model_id, arch))
    
    print(f"Testing {len(test_models)} models on {len(test_hardware)} hardware platforms...\n")
    
    # Test models on all available hardware
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for model_id, model_type in test_models:
            for hw in test_hardware:
                print(f"Submitting test: {model_id} on {hw}")
                future = executor.submit(test_model_on_hardware, model_id, model_type, hw, args.timeout)
                futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "Success" if result["success"] else f"Failed: {result['error']}"
            print(f"Completed: {result['model_id']} on {result['hardware']} - {status}")
    
    # Store results in DuckDB if available
    if DUCKDB_AVAILABLE:
        print("\nStoring results in DuckDB...")
        conn = create_db_and_tables()
        if conn:
            store_results_in_db(conn, results, hardware_info)
            conn.close()
            print("Results stored in hardware_compatibility_matrix.duckdb")
        else:
            print("DuckDB not available, skipping database storage.")
    
    # Generate compatibility report
    print(f"\nGenerating compatibility reports in {args.output_dir}...")
    generate_compatibility_report(results, hardware_info, args.output_dir)
    print("Reports generated successfully.")

if __name__ == "__main__":
    main()