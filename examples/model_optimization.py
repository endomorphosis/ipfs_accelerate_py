#!/usr/bin/env python3
"""
Model Optimization Example

This example demonstrates how to use the model-hardware compatibility
system to optimize model selection and deployment strategies.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

def demonstrate_model_optimization():
    """Demonstrate model optimization features."""
    
    print("🤖 IPFS Accelerate Python - Model Optimization Example")
    print("=" * 65)
    
    try:
        from hardware_detection import HardwareDetector
        from utils.model_compatibility import (
            get_optimal_hardware, 
            check_model_compatibility,
            get_supported_models,
            compatibility_manager
        )
        
        # Initialize hardware detector
        detector = HardwareDetector()
        available_hardware = [hw for hw, avail in detector.get_available_hardware().items() if avail]
        
        print(f"🖥️  Available hardware: {', '.join(available_hardware)}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required modules are available")
        return False
    
    # Part 1: Model Family Overview
    print("\n📚 Part 1: Supported Model Families")
    print("-" * 40)
    
    supported_models = get_supported_models()
    print(f"Supported model families: {len(supported_models)}")
    
    for i, model_family in enumerate(supported_models, 1):
        profile = compatibility_manager.get_model_profile(model_family)
        if profile:
            size_info = f"({profile.model_size.value}, ~{profile.memory_base_mb}MB)"
            web_compat = "🌐" if profile.web_compatible else ""
            mobile_compat = "📱" if profile.mobile_compatible else ""
            print(f"  {i:2d}. {model_family:15} {size_info:20} {web_compat}{mobile_compat}")
    
    # Part 2: Optimization Scenarios
    print("\n🎯 Part 2: Optimization Scenarios")
    print("-" * 40)
    
    scenarios = [
        {
            "name": "High Performance Setup",
            "models": ["llama", "gpt2", "clip"],
            "constraints": {"memory_limit_gb": None, "prefer_web_compatible": False},
            "description": "Optimize for maximum performance on available hardware"
        },
        {
            "name": "Web Application", 
            "models": ["bert", "whisper", "vit"],
            "constraints": {"memory_limit_gb": 2, "prefer_web_compatible": True},
            "description": "Optimize for web browser deployment"
        },
        {
            "name": "Mobile/Edge Deployment",
            "models": ["bert", "whisper", "vit"],
            "constraints": {"memory_limit_gb": 1, "prefer_mobile": True},
            "description": "Optimize for mobile and edge devices"
        },
        {
            "name": "Memory-Constrained Environment",
            "models": ["bert", "gpt2", "t5"],
            "constraints": {"memory_limit_gb": 4, "prefer_web_compatible": False},
            "description": "Optimize for systems with limited memory"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🚀 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Testing models: {', '.join(scenario['models'])}")
        
        for model in scenario['models']:
            recommendation = get_optimal_hardware(
                model, 
                available_hardware, 
                **scenario['constraints']
            )
            
            hw = recommendation.get('recommended_hardware', 'unknown')
            confidence = recommendation.get('confidence', 'unknown')
            performance = recommendation.get('performance_multiplier', 1.0)
            
            # Get memory info
            memory_req = recommendation.get('memory_requirements', {})
            min_mem = memory_req.get('minimum_gb', '?')
            rec_mem = memory_req.get('recommended_gb', '?')
            
            print(f"     📝 {model:8} → {hw:8} (conf: {confidence:6}, perf: {performance:4.1f}x, mem: {min_mem}-{rec_mem}GB)")
    
    # Part 3: Detailed Compatibility Analysis
    print("\n🔍 Part 3: Detailed Compatibility Analysis")
    print("-" * 45)
    
    # Pick a representative model for detailed analysis
    test_model = "bert-base-uncased"
    print(f"🧪 Analyzing: {test_model}")
    
    for hardware in available_hardware:
        compat = check_model_compatibility(test_model, hardware)
        
        status = "✅" if compat['compatible'] else "❌"
        performance = compat.get('performance_rating', 1.0)
        
        print(f"  {status} {hardware:10} (perf: {performance:4.1f}x)", end="")
        
        # Show memory requirements
        memory_req = compat.get('memory_requirements')
        if memory_req:
            min_mem = memory_req.get('minimum_gb', '?')
            rec_mem = memory_req.get('recommended_gb', '?')
            print(f" mem: {min_mem}-{rec_mem}GB", end="")
        
        # Show supported precisions
        precisions = compat.get('supported_precisions', [])
        if precisions:
            print(f" precisions: {'/'.join(precisions)}", end="")
        
        print()  # New line
        
        # Show issues if any
        issues = compat.get('issues', [])
        for issue in issues:
            print(f"      ⚠️  {issue}")
        
        # Show notes
        notes = compat.get('notes')
        if notes:
            print(f"      💡 {notes}")
    
    # Part 4: Performance Estimation
    print("\n⚡ Part 4: Performance Estimation")
    print("-" * 35)
    
    test_scenarios = [
        ("bert", "cpu", 1, 128),
        ("bert", "cuda", 1, 128) if "cuda" in available_hardware else ("bert", "mps", 1, 128),
        ("gpt2", "cpu", 1, 256),
        ("whisper", "cpu", 1, None),
    ]
    
    print("Model estimation (rough approximations):")
    for model, hardware, batch_size, seq_len in test_scenarios:
        if hardware not in available_hardware:
            continue
            
        timing = compatibility_manager.estimate_inference_time(
            model, hardware, batch_size, seq_len or 512
        )
        
        time_ms = timing.get('estimated_time_ms')
        confidence = timing.get('confidence', 'unknown')
        
        if time_ms:
            seq_info = f" (seq_len: {seq_len})" if seq_len else ""
            print(f"  🕒 {model:8} on {hardware:8}: ~{time_ms:6.1f}ms{seq_info} (confidence: {confidence})")
        else:
            print(f"  ❓ {model:8} on {hardware:8}: No timing data available")
    
    # Part 5: Optimization Recommendations
    print("\n💡 Part 5: System-Specific Recommendations")
    print("-" * 45)
    
    print("Based on your available hardware, here are optimization tips:")
    
    if "cuda" in available_hardware:
        print("\n🚀 CUDA GPU Detected:")
        print("  ✨ Best for: Large models (LLaMA, GPT), heavy workloads")
        print("  💡 Tips:")
        print("    - Use FP16 precision for 2x memory efficiency")
        print("    - Consider batch processing for throughput")
        print("    - Monitor GPU memory usage with nvidia-smi")
        
        # Get CUDA details
        try:
            cuda_details = detector.get_hardware_details("cuda")
            if cuda_details and cuda_details.get('memory_total_gb'):
                memory_gb = cuda_details['memory_total_gb']
                print(f"    - Your GPU has {memory_gb:.1f}GB memory")
                
                if memory_gb >= 16:
                    print("    - Can handle large models (7B parameters)")
                elif memory_gb >= 8:
                    print("    - Good for medium models (3B parameters)")
                else:
                    print("    - Best for smaller models, use quantization")
        except:
            pass
    
    elif "mps" in available_hardware:
        print("\n🍎 Apple Silicon MPS Detected:")
        print("  ✨ Best for: Efficient inference with unified memory")
        print("  💡 Tips:")
        print("    - Excellent memory efficiency due to unified architecture")
        print("    - Good performance on most transformer models")
        print("    - Consider FP16 for better performance")
    
    elif "webnn" in available_hardware or "webgpu" in available_hardware:
        web_types = [hw for hw in ["webnn", "webgpu"] if hw in available_hardware]
        print(f"\n🌐 Web Acceleration Detected: {', '.join(web_types)}")
        print("  ✨ Best for: Browser-based applications")
        print("  💡 Tips:")
        print("    - Use smaller models (BERT, DistilBERT)")
        print("    - Consider quantized versions")
        print("    - Test across different browsers")
    
    else:
        print("\n🖥️  CPU-Only Setup:")
        print("  ✨ Best for: Development, testing, smaller models")
        print("  💡 Tips:")
        print("    - Use quantized models (INT8) for better performance")
        print("    - Consider smaller model variants (DistilBERT vs BERT)")
        print("    - Optimize batch sizes for your use case")
        print("    - Consider ONNX runtime for CPU optimization")
    
    # Part 6: Model Selection Guide
    print("\n📋 Part 6: Model Selection Guide")
    print("-" * 35)
    
    use_cases = {
        "Text Classification": ["bert", "distilbert", "roberta"],
        "Text Generation": ["gpt2", "t5"],
        "Question Answering": ["bert", "distilbert"],
        "Image Classification": ["vit", "resnet"],
        "Image-Text Tasks": ["clip", "blip"],
        "Audio Processing": ["whisper", "wav2vec2"],
    }
    
    for use_case, recommended_models in use_cases.items():
        print(f"\n🎯 {use_case}:")
        
        for model in recommended_models:
            if model in supported_models:
                # Get best hardware for this model
                rec = get_optimal_hardware(model, available_hardware)
                best_hw = rec.get('recommended_hardware', 'cpu')
                perf = rec.get('performance_multiplier', 1.0)
                
                print(f"  📝 {model:12} → best on {best_hw:8} ({perf:3.1f}x)")
    
    print("\n✅ Model optimization analysis completed!")
    print("\n🚀 Ready to start optimizing your ML workflows!")
    
    return True

if __name__ == "__main__":
    success = demonstrate_model_optimization()
    sys.exit(0 if success else 1)