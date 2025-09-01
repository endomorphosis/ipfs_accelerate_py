#!/usr/bin/env python3
"""
Comprehensive Model Deployment Example

This example demonstrates real-world model deployment scenarios with
hardware optimization, performance monitoring, and production readiness assessment.
"""

import sys
import os
import time
import json
from typing import Dict, List, Any, Optional
import argparse

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Deployment Example")
    parser.add_argument("--model", default="auto", help="Model to test (auto for best available)")
    parser.add_argument("--hardware", default="auto", help="Hardware to use (auto for best available)")  
    parser.add_argument("--mode", choices=["quick", "comprehensive", "production"], default="quick", 
                       help="Testing mode")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarking")
    parser.add_argument("--validate", action="store_true", help="Run production validation")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 IPFS ACCELERATE PYTHON - COMPREHENSIVE MODEL DEPLOYMENT EXAMPLE")
    print("=" * 80)
    print(f"Mode: {args.mode} | Model: {args.model} | Hardware: {args.hardware}")
    print()
    
    # Phase 1: Hardware Detection and Optimization
    print("📡 PHASE 1: Hardware Detection and Optimization")
    print("-" * 60)
    
    try:
        from hardware_detection import HardwareDetector
        detector = HardwareDetector()
        
        available_hardware = detector.get_available_hardware()
        best_hardware = detector.get_best_available_hardware()
        
        print(f"✅ Available Hardware: {', '.join(available_hardware)}")
        print(f"🎯 Recommended Hardware: {best_hardware}")
        
        # Get detailed hardware info
        for hw in available_hardware[:3]:  # Show first 3 for brevity
            try:
                info = detector.get_hardware_info(hw)
                print(f"   📊 {hw}: {info.get('status', 'unknown')} - {info.get('description', 'No details')}")
            except:
                print(f"   📊 {hw}: Available")
        
        target_hardware = args.hardware if args.hardware != "auto" else best_hardware
        
    except ImportError as e:
        print(f"⚠️  Hardware detection not available: {e}")
        print("   Using CPU as fallback")
        target_hardware = "cpu"
        available_hardware = ["cpu"]
    
    print()
    
    # Phase 2: Model Compatibility Assessment
    print("🧪 PHASE 2: Model Compatibility Assessment")
    print("-" * 60)
    
    try:
        from real_world_model_testing import RealWorldModelTester, get_test_models_catalog
        
        tester = RealWorldModelTester()
        models_catalog = get_test_models_catalog()
        
        print(f"📋 Available Test Models: {len(models_catalog)}")
        
        # Show model catalog
        for i, (model_name, info) in enumerate(list(models_catalog.items())[:3]):
            print(f"   {i+1}. {model_name}")
            print(f"      Size: {info['size_mb']}MB | Family: {info['family']} | {info['description']}")
        
        # Get compatibility matrix
        print(f"\n🔍 Generating compatibility matrix...")
        compatibility = tester.get_model_compatibility_matrix()
        print(f"✅ Matrix generated for {compatibility['models_count']} models and {compatibility['hardware_count']} hardware types")
        
        # Select model for testing
        if args.model == "auto":
            # Choose smallest model for quick demo
            test_model = min(models_catalog.items(), key=lambda x: x[1]['size_mb'])[0]
        else:
            test_model = args.model if args.model in models_catalog else list(models_catalog.keys())[0]
        
        print(f"🎯 Selected Model: {test_model} ({models_catalog[test_model]['size_mb']}MB)")
        
    except Exception as e:
        print(f"⚠️  Model testing not available: {e}")
        test_model = "bert-base-uncased"
        print(f"   Using simulated model: {test_model}")
    
    print()
    
    # Phase 3: Performance Benchmarking (Optional)
    if args.benchmark:
        print("📊 PHASE 3: Performance Benchmarking")
        print("-" * 60)
        
        try:
            from advanced_benchmarking import run_quick_benchmark, AdvancedBenchmarker
            
            print("🚀 Running comprehensive benchmark suite...")
            benchmark_results = run_quick_benchmark()
            
            if benchmark_results:
                print(f"✅ Benchmark completed successfully")
                if hasattr(benchmark_results, 'results') and benchmark_results.results:
                    avg_latency = sum(r.latency_ms for r in benchmark_results.results) / len(benchmark_results.results)
                    print(f"   📈 Average Latency: {avg_latency:.1f}ms")
                    print(f"   🎯 Success Rate: {benchmark_results.success_rate:.1f}%")
                else:
                    print("   📊 Benchmark completed with simulated results")
            else:
                print("   ⚠️  Benchmark completed with limitations")
        
        except Exception as e:
            print(f"⚠️  Benchmarking not available: {e}")
            print("   Simulating benchmark: 15.2ms avg latency, 98% success rate")
        
        print()
    
    # Phase 4: Production Validation (Optional)
    if args.validate:
        print("🔍 PHASE 4: Production Validation")
        print("-" * 60)
        
        try:
            from production_validation import run_production_validation
            
            validation_level = "production" if args.mode == "production" else "basic"
            print(f"🔍 Running {validation_level} validation...")
            
            validation_results = run_production_validation(validation_level)
            
            if validation_results:
                score = getattr(validation_results, 'overall_score', 0)
                print(f"📊 Production Readiness Score: {score:.1f}/100")
                
                if score >= 80:
                    print("   ✅ EXCELLENT - Ready for production deployment")
                elif score >= 60:
                    print("   ✅ GOOD - Ready with minor improvements")
                elif score >= 40:
                    print("   ⚠️  FAIR - Needs improvements before production")
                else:
                    print("   ❌ POOR - Major improvements needed")
            else:
                print("   ⚠️  Validation completed with limitations")
        
        except Exception as e:
            print(f"⚠️  Production validation not available: {e}")
            print("   Simulating validation: 82.5/100 - Good, minor improvements recommended")
        
        print()
    
    # Phase 5: Real Model Testing
    print("🧬 PHASE 5: Real Model Testing")
    print("-" * 60)
    
    try:
        print(f"🔬 Testing {test_model} on {target_hardware}...")
        
        if 'tester' in locals():
            test_result = tester.test_single_model(test_model, target_hardware)
            
            if test_result.success:
                print(f"✅ Model test successful!")
                print(f"   ⚡ Latency: {test_result.latency_ms:.1f}ms")
                print(f"   💾 Memory: {test_result.memory_mb:.0f}MB")
                if test_result.tokens_per_second:
                    print(f"   🚀 Throughput: {test_result.tokens_per_second:.1f} tokens/second")
                
                if test_result.inference_details:
                    details = test_result.inference_details
                    print(f"   📋 Task: {details.get('task_type', 'unknown')}")
                    if details.get('simulation'):
                        print("   🔮 Results: Simulated (transformers not available)")
                    else:
                        print("   🎯 Results: Real model execution")
            else:
                print(f"❌ Model test failed: {test_result.error_message}")
        else:
            # Simulate model testing
            print(f"🔮 Simulating model test (real testing not available)...")
            simulated_latency = 15.5
            simulated_memory = 440
            print(f"✅ Simulated results:")
            print(f"   ⚡ Latency: {simulated_latency}ms")  
            print(f"   💾 Memory: {simulated_memory}MB")
            print(f"   🚀 Throughput: ~65 tokens/second")
    
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        print("   This is normal if transformers library is not installed")
    
    print()
    
    # Phase 6: Optimization Recommendations
    print("💡 PHASE 6: Optimization Recommendations")
    print("-" * 60)
    
    try:
        # Generate optimization recommendations based on available hardware
        recommendations = []
        
        if "cuda" in available_hardware:
            recommendations.append("🚀 CUDA available: Consider GPU acceleration for large models")
        elif "mps" in available_hardware:
            recommendations.append("🍎 Apple Silicon detected: Use MPS for optimal performance")
        elif "openvino" in available_hardware:
            recommendations.append("🔧 Intel optimization: Consider OpenVINO for CPU acceleration")
        
        if "webnn" in available_hardware or "webgpu" in available_hardware:
            recommendations.append("🌐 Web acceleration available: Deploy in browser environments")
        
        # Model-specific recommendations
        if 'test_model' in locals() and 'models_catalog' in locals():
            model_info = models_catalog.get(test_model, {})
            model_size = model_info.get('size_mb', 0)
            
            if model_size < 100:
                recommendations.append(f"📱 Small model ({model_size}MB): Great for mobile/edge deployment")
            elif model_size > 500:
                recommendations.append(f"🏭 Large model ({model_size}MB): Consider quantization for deployment")
        
        # Performance recommendations
        if args.benchmark and 'benchmark_results' in locals():
            recommendations.append("📊 Run advanced benchmarking for production deployment")
        
        if not recommendations:
            recommendations = [
                "🔧 Install optional dependencies for enhanced functionality",
                "📚 Check documentation for platform-specific optimizations",
                "🧪 Run comprehensive tests before production deployment"
            ]
        
        print("🎯 Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    except Exception as e:
        print(f"⚠️  Error generating recommendations: {e}")
        print("   1. 🔧 Install required dependencies for full functionality")
        print("   2. 📊 Run production validation before deployment")
        print("   3. 🧪 Test with real models for accurate performance metrics")
    
    print()
    
    # Phase 7: Deployment Summary
    print("📋 PHASE 7: Deployment Summary")
    print("-" * 60)
    
    try:
        summary = {
            "hardware_recommendation": target_hardware,
            "model_tested": test_model if 'test_model' in locals() else "simulated",
            "deployment_mode": args.mode,
            "features_tested": {
                "hardware_detection": "available_hardware" in locals(),
                "model_testing": "tester" in locals(),
                "benchmarking": args.benchmark,
                "validation": args.validate
            }
        }
        
        print("✅ Deployment Analysis Complete:")
        print(f"   🎯 Target Hardware: {summary['hardware_recommendation']}")
        print(f"   🧬 Model Tested: {summary['model_tested']}")
        print(f"   ⚙️  Mode: {summary['deployment_mode']}")
        
        # Next steps
        print(f"\n🚀 Next Steps:")
        print(f"   1. Install dependencies: pip install ipfs-accelerate-py[full]")
        print(f"   2. Run production validation: python utils/production_validation.py")
        print(f"   3. Start performance monitoring: python utils/performance_dashboard.py")
        print(f"   4. Deploy to production: python utils/production_deployment.py")
        
    except Exception as e:
        print(f"Summary generation error: {e}")
        print("✅ Basic deployment analysis completed")
        print("   Check individual components for detailed results")
    
    print()
    print("=" * 80)
    print("🎉 COMPREHENSIVE MODEL DEPLOYMENT EXAMPLE COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()