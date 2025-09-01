#!/usr/bin/env python3
"""
Advanced Performance Analysis Example

Demonstrates the enhanced performance modeling capabilities of IPFS Accelerate Python,
including realistic hardware benchmarking, optimization recommendations, and detailed
performance analysis across different model types and hardware configurations.
"""

import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.performance_modeling import (
    performance_simulator, 
    simulate_model_performance,
    get_hardware_recommendations,
    HardwareType,
    PrecisionMode
)
from utils.model_compatibility import (
    get_detailed_performance_analysis,
    benchmark_model_performance,
    get_optimal_hardware
)
from hardware_detection import HardwareDetector

def demonstrate_basic_performance_simulation():
    """Demonstrate basic performance simulation for different models."""
    print("🚀 Basic Performance Simulation")
    print("=" * 50)
    
    # Test models with different characteristics
    test_models = [
        "bert-base-uncased",
        "gpt2", 
        "clip-vit-base-patch32",
        "whisper-base",
        "llama-7b"
    ]
    
    hardware_options = ["cpu", "cuda", "mps", "webnn", "webgpu"]
    
    for model in test_models:
        print(f"\n📊 Model: {model}")
        print("-" * 30)
        
        for hardware in hardware_options[:3]:  # Test top 3 for brevity
            try:
                result = simulate_model_performance(model, hardware)
                print(f"  {hardware.upper():>8}: {result.inference_time_ms:>6.1f}ms | "
                      f"{result.memory_usage_mb:>6.0f}MB | "
                      f"{result.efficiency_score:>5.3f} efficiency | "
                      f"bottleneck: {result.bottleneck}")
            except Exception as e:
                print(f"  {hardware.upper():>8}: Error - {e}")

def demonstrate_hardware_recommendations():
    """Show intelligent hardware recommendations."""
    print("\n🎯 Hardware Recommendations")
    print("=" * 50)
    
    scenarios = [
        {
            "model": "bert-base-uncased",
            "hardware": ["cpu", "cuda", "mps", "webnn"],
            "description": "BERT for text classification"
        },
        {
            "model": "gpt2",
            "hardware": ["cpu", "cuda", "mps"],
            "description": "GPT-2 for text generation"
        },
        {
            "model": "whisper-base",
            "hardware": ["cpu", "cuda", "qualcomm"],
            "description": "Whisper for speech recognition"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔍 {scenario['description']}")
        print(f"   Model: {scenario['model']}")
        
        recommendations = get_hardware_recommendations(
            scenario["model"], scenario["hardware"]
        )
        
        if "error" not in recommendations:
            best = recommendations["recommended_hardware"]
            perf = recommendations["performance"]
            
            print(f"   ✅ Best: {best.upper()} "
                  f"({perf.inference_time_ms:.1f}ms, "
                  f"{perf.memory_usage_mb:.0f}MB)")
            print(f"   💡 Bottleneck: {perf.bottleneck}")
            
            if perf.recommendations:
                print(f"   📋 Top recommendation: {perf.recommendations[0]}")
                
            # Show comparison with alternatives
            all_options = recommendations["all_options"]
            if len(all_options) > 1:
                print("   📊 Alternatives:")
                for hw, result in all_options.items():
                    if hw != best:
                        speedup = result.inference_time_ms / perf.inference_time_ms
                        print(f"      {hw.upper()}: {speedup:.1f}x slower, "
                              f"{result.memory_usage_mb:.0f}MB memory")
        else:
            print(f"   ❌ Error: {recommendations['error']}")

def demonstrate_detailed_analysis():
    """Show detailed performance analysis capabilities."""
    print("\n🔬 Detailed Performance Analysis")
    print("=" * 50)
    
    model = "bert-base-uncased"
    hardware_options = ["cpu", "cuda", "mps", "webnn"]
    
    print(f"📝 Analyzing {model} across {len(hardware_options)} hardware options...")
    
    try:
        analysis = get_detailed_performance_analysis(model, hardware_options)
        
        print(f"\n✨ Recommended Configuration:")
        print(f"   Hardware: {analysis['recommended_hardware'].upper()}")
        
        perf = analysis['performance_details']
        print(f"   Performance: {perf['inference_time_ms']}ms inference")
        print(f"   Memory: {perf['memory_usage_mb']:.0f}MB required")
        print(f"   Throughput: {perf['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"   Efficiency: {perf['efficiency_score']:.3f}/1.000")
        print(f"   Power: {perf['power_consumption_watts']:.2f}W")
        
        print(f"\n🔧 Optimization Recommendations:")
        for i, rec in enumerate(analysis['optimization_recommendations'][:3], 1):
            print(f"   {i}. {rec}")
            
        print(f"\n⚡ Hardware Comparison:")
        for hw, comp in analysis['hardware_comparison'].items():
            print(f"   {hw.upper():>8}: {comp['inference_time_ms']:>6.1f}ms | "
                  f"{comp['efficiency_score']:>5.3f} efficiency | "
                  f"{comp['relative_performance']:>6} vs best")
                  
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def demonstrate_benchmarking():
    """Show comprehensive benchmarking across configurations."""
    print("\n🏁 Performance Benchmarking")
    print("=" * 50)
    
    model = "gpt2"
    hardware_options = ["cpu", "cuda", "mps"]
    batch_sizes = [1, 4, 16]
    
    print(f"🧪 Benchmarking {model} across:")
    print(f"   Hardware: {', '.join(hw.upper() for hw in hardware_options)}")
    print(f"   Batch sizes: {', '.join(map(str, batch_sizes))}")
    
    try:
        results = benchmark_model_performance(
            model, hardware_options, batch_sizes=batch_sizes, optimize_for="speed"
        )
        
        if "error" not in results:
            print(f"\n📈 Results by Hardware:")
            for hardware, hw_results in results["hardware_results"].items():
                print(f"\n   {hardware.upper()}:")
                for batch_key, metrics in hw_results.items():
                    batch_num = batch_key.split('_')[1]
                    print(f"     Batch {batch_num:>2}: {metrics['inference_time_ms']:>6.1f}ms | "
                          f"{metrics['throughput']:>6.1f} samples/sec | "
                          f"{metrics['memory_usage_mb']:>6.0f}MB")
                          
            print(f"\n🎯 Optimal Configurations (optimized for speed):")
            for hardware, config in results["optimal_configurations"].items():
                batch_size = config["best_batch_size"].split('_')[1]
                perf = config["performance"]
                print(f"   {hardware.upper():>8}: Batch size {batch_size} "
                      f"-> {perf['inference_time_ms']:.1f}ms "
                      f"({perf['throughput']:.1f} samples/sec)")
                      
        else:
            print(f"❌ Benchmarking failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Benchmarking error: {e}")

def demonstrate_precision_optimization():
    """Show precision optimization recommendations."""
    print("\n🎛️ Precision Optimization")
    print("=" * 50)
    
    model = "bert-base-uncased"
    hardware = "cuda"
    precisions = ["fp32", "fp16", "int8"]
    
    print(f"🔢 Testing {model} on {hardware.upper()} with different precisions:")
    
    results = {}
    for precision in precisions:
        try:
            result = performance_simulator.simulate_inference_performance(
                model, hardware, precision=precision
            )
            results[precision] = result
            
            print(f"   {precision.upper():>4}: {result.inference_time_ms:>6.1f}ms | "
                  f"{result.memory_usage_mb:>6.0f}MB | "
                  f"{result.efficiency_score:>5.3f} efficiency")
        except Exception as e:
            print(f"   {precision.upper():>4}: Error - {e}")
    
    # Show recommendations
    if results:
        print(f"\n💡 Precision Recommendations:")
        
        # Find best for each metric
        fastest = min(results.keys(), key=lambda p: results[p].inference_time_ms)
        most_efficient = max(results.keys(), key=lambda p: results[p].efficiency_score)
        lowest_memory = min(results.keys(), key=lambda p: results[p].memory_usage_mb)
        
        print(f"   🏃 Fastest: {fastest.upper()} "
              f"({results[fastest].inference_time_ms:.1f}ms)")
        print(f"   ⚡ Most efficient: {most_efficient.upper()} "
              f"({results[most_efficient].efficiency_score:.3f})")
        print(f"   💾 Lowest memory: {lowest_memory.upper()} "
              f"({results[lowest_memory].memory_usage_mb:.0f}MB)")

def demonstrate_hardware_detection_integration():
    """Show integration with actual hardware detection."""
    print("\n🔌 Hardware Detection Integration")
    print("=" * 50)
    
    # Detect available hardware
    detector = HardwareDetector()
    
    try:
        # This will work with mocked hardware in CI environments
        available_hw = detector.get_available_hardware_types()
        print(f"🖥️  Detected hardware: {', '.join(hw.upper() for hw in available_hw)}")
        
        if available_hw:
            # Get recommendation for detected hardware
            model = "bert-base-uncased"
            recommendation = get_optimal_hardware(model, available_hw)
            
            print(f"\n🎯 For {model} on your system:")
            print(f"   Best option: {recommendation['recommended_hardware'].upper()}")
            print(f"   Confidence: {recommendation['confidence']}")
            
            mem_req = recommendation.get('memory_requirements', {})
            if mem_req:
                print(f"   Memory needed: {mem_req.get('recommended_gb', 'unknown')}GB")
                
            # Show performance estimate
            best_hw = recommendation['recommended_hardware']
            if best_hw in available_hw:
                try:
                    perf = simulate_model_performance(model, best_hw)
                    print(f"   Expected: {perf.inference_time_ms:.1f}ms inference, "
                          f"{perf.memory_usage_mb:.0f}MB memory")
                except Exception as e:
                    print(f"   Performance estimation failed: {e}")
        else:
            print("❌ No hardware detected")
            
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")

def run_comprehensive_demo():
    """Run all demonstration functions."""
    print("🌟 IPFS Accelerate Python - Advanced Performance Analysis Demo")
    print("=" * 80)
    print("This demo showcases enhanced performance modeling and optimization")
    print("capabilities that work without requiring actual GPU hardware.")
    print()
    
    demonstrations = [
        ("Basic Performance Simulation", demonstrate_basic_performance_simulation),
        ("Hardware Recommendations", demonstrate_hardware_recommendations), 
        ("Detailed Analysis", demonstrate_detailed_analysis),
        ("Performance Benchmarking", demonstrate_benchmarking),
        ("Precision Optimization", demonstrate_precision_optimization),
        ("Hardware Detection Integration", demonstrate_hardware_detection_integration)
    ]
    
    for name, demo_func in demonstrations:
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
        
        print("\n" + "─" * 80)
    
    print("\n✨ Demo completed! The repository now includes:")
    print("   • Realistic performance modeling for 7 model families")
    print("   • Support for 8 different hardware platforms")
    print("   • Intelligent optimization recommendations")
    print("   • Comprehensive benchmarking capabilities")
    print("   • All features work without GPU dependencies!")

def main():
    """Main entry point with command line options."""
    parser = argparse.ArgumentParser(
        description="Advanced Performance Analysis Demo for IPFS Accelerate Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performance_analysis.py                    # Run full demo
  python performance_analysis.py --quick           # Quick overview
  python performance_analysis.py --model bert-base # Analyze specific model
  python performance_analysis.py --hardware cuda   # Test specific hardware
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                       help="Run quick performance overview only")
    parser.add_argument("--model", type=str,
                       help="Analyze specific model performance")
    parser.add_argument("--hardware", type=str, nargs='+',
                       help="Test specific hardware options")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"],
                       help="Test specific precision mode")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run comprehensive benchmarking")
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Quick Performance Overview")
        print("=" * 40)
        demonstrate_basic_performance_simulation()
        
    elif args.model:
        hardware_list = args.hardware or ["cpu", "cuda", "mps", "webnn"]
        print(f"🔍 Analyzing {args.model}")
        print("=" * 40)
        
        if args.precision:
            try:
                result = simulate_model_performance(args.model, hardware_list[0], 
                                                  precision=args.precision)
                print(f"Model: {args.model}")
                print(f"Hardware: {hardware_list[0].upper()}")
                print(f"Precision: {args.precision.upper()}")
                print(f"Inference time: {result.inference_time_ms:.1f}ms")
                print(f"Memory usage: {result.memory_usage_mb:.0f}MB")
                print(f"Efficiency: {result.efficiency_score:.3f}")
                print(f"Bottleneck: {result.bottleneck}")
                for rec in result.recommendations[:3]:
                    print(f"💡 {rec}")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            try:
                analysis = get_detailed_performance_analysis(args.model, hardware_list)
                print(json.dumps(analysis, indent=2))
            except Exception as e:
                print(f"❌ Error: {e}")
                
    elif args.benchmark:
        print("🏁 Running Comprehensive Benchmarks")
        print("=" * 40)
        demonstrate_benchmarking()
        
    else:
        # Run full comprehensive demo
        run_comprehensive_demo()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())