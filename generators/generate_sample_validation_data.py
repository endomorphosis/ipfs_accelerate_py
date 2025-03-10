#!/usr/bin/env python3
"""
Generate sample validation data for the model hardware validation tracker.

This script creates a set of sample validation results to demonstrate and test
the model hardware validation tracker system.

Usage:
    python generate_sample_validation_data.py --output-dir ./validation_samples
    python generate_sample_validation_data.py --realistic
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List

# Import model and hardware definitions
try:
    from test_comprehensive_hardware_coverage import KEY_MODELS, HARDWARE_PLATFORMS
except ImportError:
    print("Error: Could not import from test_comprehensive_hardware_coverage.py")
    print("Make sure it exists in the same directory")
    exit(1)

# Mock issues that could be encountered
MOCK_ISSUES = [
    {
        "description": "Out of memory error when using batch size > 2",
        "workaround": "Reduce batch size or use model quantization"
    },
    {
        "description": "Slow inference performance compared to CPU baseline",
        "workaround": "Ensure latest device drivers are installed"
    },
    {
        "description": "Model crashes with certain input dimensions",
        "workaround": "Ensure inputs are padded to multiples of 8"
    },
    {
        "description": "Precision issues on 16-bit floating point operations",
        "workaround": "Use mixed precision training"
    },
    {
        "description": "Incompatible tensor operations in model definition",
        "workaround": "Use compatibility layer or alternative implementation"
    },
    {
        "description": "Model weights exceed available memory",
        "workaround": "Use model sharding or quantization"
    },
    {
        "description": "Device driver compatibility issues",
        "workaround": "Update to latest driver version"
    }
]

def generate_mock_performance(model_category: str, hardware_key: str, batch_size: int = 1) -> Dict:
    """
    Generate mock performance metrics based on model category and hardware.
    
    Args:
        model_category: Category of the model
        hardware_key: Hardware platform key
        batch_size: Batch size for inference
        
    Returns:
        Dict: Mock performance metrics
    """
    # Base throughput values by hardware
    base_throughput = {
        "cpu": 20,
        "cuda": 100,
        "rocm": 85,
        "mps": 65,
        "openvino": 40,
        "webnn": 15,
        "webgpu": 25
    }
    
    # Category multipliers
    category_multiplier = {
        "embedding": 5.0,
        "text_generation": 0.2,
        "vision": 0.8,
        "vision_text": 0.6,
        "audio": 0.3,
        "audio_text": 0.4,
        "multimodal": 0.1,
        "video": 0.05
    }
    
    # Memory usage by category (in MB)
    base_memory = {
        "embedding": 50,
        "text_generation": 500,
        "vision": 200,
        "vision_text": 300,
        "audio": 150,
        "audio_text": 250,
        "multimodal": 800,
        "video": 400
    }
    
    # Calculate performance values with some randomness
    throughput_base = base_throughput.get(hardware_key, 10)
    category_mult = category_multiplier.get(model_category, 1.0)
    
    # Add randomness (±20%)
    random_factor = random.uniform(0.8, 1.2)
    
    throughput = throughput_base * category_mult * batch_size * random_factor
    latency = 1000 / throughput  # ms
    memory = base_memory.get(model_category, 100) * (1 + 0.2 * (batch_size - 1)) * random.uniform(0.9, 1.1)
    
    return {
        "throughput": round(throughput, 2),
        "latency_ms": round(latency, 2),
        "memory_usage_mb": round(memory, 1),
        "batch_size": batch_size
    }

def generate_mock_requirements(model_category: str, hardware_key: str) -> Dict:
    """
    Generate mock hardware requirements.
    
    Args:
        model_category: Category of the model
        hardware_key: Hardware platform key
        
    Returns:
        Dict: Mock hardware requirements
    """
    # Memory requirements by category (in MB)
    memory_requirements = {
        "embedding": random.randint(100, 500),
        "text_generation": random.randint(1000, 8000),
        "vision": random.randint(300, 1000),
        "vision_text": random.randint(500, 2000),
        "audio": random.randint(200, 800),
        "audio_text": random.randint(400, 1200),
        "multimodal": random.randint(2000, 16000),
        "video": random.randint(800, 3000)
    }
    
    # CPU requirements
    cpu_requirements = {
        "embedding": "1+ cores",
        "text_generation": "4+ cores",
        "vision": "2+ cores",
        "vision_text": "4+ cores",
        "audio": "2+ cores",
        "audio_text": "4+ cores",
        "multimodal": "8+ cores",
        "video": "4+ cores"
    }
    
    memory = memory_requirements.get(model_category, 500)
    cpu = cpu_requirements.get(model_category, "2+ cores")
    
    # Hardware-specific requirements
    if hardware_key == "cuda":
        compute = random.choice(["CUDA 10.0+", "CUDA 11.0+", "CUDA 11.7+", "CUDA 12.0+"])
        vram = f"{int(memory * 1.2)} MB VRAM"
        return {"memory": memory, "cpu": cpu, "compute_capability": compute, "vram": vram}
    
    elif hardware_key == "rocm":
        compute = random.choice(["ROCm 4.0+", "ROCm 5.0+", "ROCm 5.4+"])
        vram = f"{int(memory * 1.3)} MB VRAM"
        return {"memory": memory, "cpu": cpu, "rocm_version": compute, "vram": vram}
    
    elif hardware_key == "mps":
        compute = random.choice(["macOS 12.0+", "macOS 13.0+"])
        return {"memory": memory, "cpu": cpu, "os_version": compute}
    
    elif hardware_key == "openvino":
        compute = random.choice(["OpenVINO 2021.4+", "OpenVINO 2022.1+", "OpenVINO 2023.0+"])
        return {"memory": memory, "cpu": cpu, "openvino_version": compute}
    
    elif hardware_key in ["webnn", "webgpu"]:
        browser = random.choice(["Chrome 113+", "Edge 113+", "Chrome 115+", "Edge 115+"])
        return {"memory": memory, "browser": browser}
    
    else:  # CPU
        return {"memory": memory, "cpu": cpu}

def generate_sample_result(
    model_key: str, 
    hardware_key: str, 
    status: str,
    implementation_type: str,
    date_offset: int = 0
) -> Dict:
    """
    Generate a sample validation result.
    
    Args:
        model_key: Model key
        hardware_key: Hardware platform key
        status: Validation status
        implementation_type: Implementation type
        date_offset: Days to offset the date by
        
    Returns:
        Dict: Sample validation result
    """
    model_info = KEY_MODELS.get(model_key, {})
    model_name = model_info.get("name", model_key)
    model_category = model_info.get("category", "unknown")
    
    hw_info = HARDWARE_PLATFORMS.get(hardware_key, {})
    hw_name = hw_info.get("name", hardware_key)
    
    # Generate date (recent, with offset)
    test_date = datetime.now() - timedelta(days=date_offset)
    
    result = {
        "model_key": model_key,
        "model_name": model_name,
        "hardware_key": hardware_key,
        "hardware_name": hw_name,
        "status": status,
        "implementation_type": implementation_type,
        "date": test_date.isoformat()
    }
    
    # Add performance metrics if passed
    if status == "pass":
        batch_sizes = [1, 2, 4, 8]
        result["performance"] = {
            str(bs): generate_mock_performance(model_category, hardware_key, bs)
            for bs in batch_sizes
        }
    
    # Add requirements
    result["requirements"] = generate_mock_requirements(model_category, hardware_key)
    
    # Add notes based on status
    if status == "pass":
        notes = random.choice([
            "All tests passed successfully",
            "Verified with latest model weights",
            "Performance within expected range",
            "All test cases executed"
        ])
    elif status == "fail":
        notes = random.choice([
            "Test failed due to compatibility issues",
            "Memory allocation error during testing",
            "Unexpected results compared to reference implementation",
            "Test timeout exceeded"
        ])
    else:
        notes = ""
    
    result["notes"] = notes
    
    # Add known issues for some failing tests
    if status == "fail" and random.random() < 0.7:
        issue = random.choice(MOCK_ISSUES)
        issue["date"] = test_date.isoformat()
        result["known_issues"] = [issue]
    
    return result

def generate_realistic_dataset() -> List[Dict]:
    """
    Generate a realistic set of validation results based on known
    compatibility and current implementation status.
    
    Returns:
        List[Dict]: List of validation results
    """
    results = []
    
    # Define implementation status based on our knowledge
    implementation_status = {
        # Real implementations
        ("bert", "cpu"): "real",
        ("bert", "cuda"): "real",
        ("bert", "rocm"): "real",
        ("bert", "mps"): "real",
        ("bert", "openvino"): "real",
        ("bert", "webnn"): "real",
        ("bert", "webgpu"): "real",
        
        ("t5", "cpu"): "real",
        ("t5", "cuda"): "real",
        ("t5", "rocm"): "real",
        ("t5", "mps"): "real",
        ("t5", "openvino"): "mock",
        ("t5", "webnn"): "real",
        ("t5", "webgpu"): "real",
        
        ("llama", "cpu"): "real",
        ("llama", "cuda"): "real",
        ("llama", "rocm"): "real",
        ("llama", "mps"): "real",
        ("llama", "openvino"): "real",
        
        ("clip", "cpu"): "real",
        ("clip", "cuda"): "real",
        ("clip", "rocm"): "real",
        ("clip", "mps"): "real",
        ("clip", "openvino"): "real",
        ("clip", "webnn"): "real",
        ("clip", "webgpu"): "real",
        
        ("vit", "cpu"): "real",
        ("vit", "cuda"): "real",
        ("vit", "rocm"): "real",
        ("vit", "mps"): "real",
        ("vit", "openvino"): "real",
        ("vit", "webnn"): "real",
        ("vit", "webgpu"): "real",
        
        ("clap", "cpu"): "real",
        ("clap", "cuda"): "real",
        ("clap", "rocm"): "real",
        ("clap", "mps"): "real",
        ("clap", "openvino"): "mock",
        
        ("whisper", "cpu"): "real",
        ("whisper", "cuda"): "real",
        ("whisper", "rocm"): "real",
        ("whisper", "mps"): "real",
        ("whisper", "openvino"): "real",
        ("whisper", "webnn"): "mock",
        ("whisper", "webgpu"): "mock",
        
        ("wav2vec2", "cpu"): "real",
        ("wav2vec2", "cuda"): "real",
        ("wav2vec2", "rocm"): "real",
        ("wav2vec2", "mps"): "real",
        ("wav2vec2", "openvino"): "mock",
        
        ("llava", "cpu"): "real",
        ("llava", "cuda"): "real",
        ("llava", "openvino"): "mock",
        
        ("llava_next", "cpu"): "real",
        ("llava_next", "cuda"): "real",
        
        ("xclip", "cpu"): "real",
        ("xclip", "cuda"): "real",
        ("xclip", "rocm"): "real",
        ("xclip", "mps"): "real",
        ("xclip", "openvino"): "real",
        
        ("qwen", "cpu"): "real",
        ("qwen", "cuda"): "real",
        ("qwen", "rocm"): "mock",
        ("qwen", "mps"): "mock",
        ("qwen", "openvino"): "mock",
        
        ("detr", "cpu"): "real",
        ("detr", "cuda"): "real",
        ("detr", "rocm"): "real",
        ("detr", "mps"): "real",
        ("detr", "openvino"): "real",
    }
    
    # Generate results for each model-hardware combination
    for model_key in KEY_MODELS:
        for hw_key in HARDWARE_PLATFORMS:
            # Check if the combination is compatible
            is_compatible = model_key in HARDWARE_PLATFORMS[hw_key]["compatibility"]
            
            if not is_compatible:
                status = "incompatible"
                impl_type = "none"
            else:
                # Get implementation status (default to untested)
                impl_type = implementation_status.get((model_key, hw_key), "none")
                
                if impl_type == "none":
                    status = "untested"
                elif impl_type == "mock":
                    # Mocks have a mix of pass/fail
                    status = random.choices(["pass", "fail"], weights=[0.7, 0.3])[0]
                else:  # real
                    # Real implementations mostly pass
                    status = random.choices(["pass", "fail"], weights=[0.9, 0.1])[0]
            
            # Generate result with random date offset (0-30 days)
            date_offset = random.randint(0, 30)
            
            result = generate_sample_result(
                model_key,
                hw_key,
                status,
                impl_type,
                date_offset
            )
            
            results.append(result)
    
    return results

def generate_random_dataset() -> List[Dict]:
    """
    Generate a completely random set of validation results.
    
    Returns:
        List[Dict]: List of validation results
    """
    results = []
    
    for model_key in KEY_MODELS:
        for hw_key in HARDWARE_PLATFORMS:
            # Randomly decide if compatible
            is_compatible = random.random() < 0.8
            
            if not is_compatible:
                status = "incompatible"
                impl_type = "none"
            else:
                # Randomly choose status and implementation type
                status = random.choice(["pass", "fail", "untested"])
                
                if status == "untested":
                    impl_type = "none"
                else:
                    impl_type = random.choice(["real", "mock"])
            
            # Generate result with random date offset (0-60 days)
            date_offset = random.randint(0, 60)
            
            result = generate_sample_result(
                model_key,
                hw_key,
                status,
                impl_type,
                date_offset
            )
            
            results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate sample validation data")
    parser.add_argument("--output-dir", default="./validation_samples", help="Output directory for samples")
    parser.add_argument("--realistic", action="store_true", help="Generate realistic dataset based on known implementation status")
    parser.add_argument("--count", type=int, default=None, help="Number of samples to generate (default: all combinations)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset
    if args.realistic:
        results = generate_realistic_dataset()
        dataset_type = "realistic"
    else:
        results = generate_random_dataset()
        dataset_type = "random"
    
    # Limit count if specified
    if args.count is not None:
        results = random.sample(results, min(args.count, len(results)))
    
    # Save individual result files
    for i, result in enumerate(results):
        filename = f"{dataset_type}_validation_{i:03d}_{result['model_key']}_{result['hardware_key']}.json"
        filepath = os.path.join(args.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save complete dataset
    dataset_file = os.path.join(args.output_dir, f"{dataset_type}_validation_dataset.json")
    with open(dataset_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated {len(results)} sample validation results in {args.output_dir}")
    print(f"Complete dataset saved to {dataset_file}")

if __name__ == "__main__":
    main()