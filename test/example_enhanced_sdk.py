#!/usr/bin/env python
"""
Example usage of the enhanced IPFS Accelerate Python SDK.

This script demonstrates various capabilities of the enhanced SDK,
including hardware detection, model acceleration, and benchmarking.
"""

import os
import time
from ipfs_accelerate_py import Worker, ModelManager, ModelAccelerator, Benchmark
from ipfs_accelerate_py.hardware import HardwareProfile, HardwareDetector
from ipfs_accelerate_py.benchmark import BenchmarkConfig
from ipfs_accelerate_py.quantization import QuantizationEngine, QuantizationConfig, CalibrationDataset

def main()))))))))):
    """Run example SDK usage."""
    print()))))))))"IPFS Accelerate Python SDK - Enhanced Example\n")
    
    # Step 1: Detect available hardware
    print()))))))))"Step 1: Detecting available hardware...")
    hardware_detector = HardwareDetector())))))))))
    available_hardware = hardware_detector.detect_all())))))))))
    
    print()))))))))f"Available hardware platforms:")
    for hw_type, hw_details in available_hardware.items()))))))))):
        is_real = not hw_details.get()))))))))"simulation_enabled", False)
        status = "✓ Real" if is_real else "⚠ Simulation":
            print()))))))))f"  - {hw_type}: {hw_details.get()))))))))'name', 'Unknown')} []]],,,{status}]"),
            print())))))))))
    
    # Step 2: Initialize worker with hardware
            print()))))))))"Step 2: Initializing worker...")
            worker = Worker())))))))))
            worker_status = worker.init_hardware())))))))))
            print()))))))))f"\1{list()))))))))worker_status[]]],,,'hwtest'].keys()))))))))))}\3"),
            print())))))))))
    
    # Step 3: Select optimal hardware for a model
            model_name = "bert-base-uncased"
            print()))))))))f"Step 3: Finding optimal hardware for {model_name}...")
            optimal_hardware = worker.get_optimal_hardware()))))))))model_name, task_type="text", batch_size=1)
            print()))))))))f"\1{optimal_hardware}\3")
            print())))))))))
    
    # Step 4: Create hardware profile and load model
            print()))))))))"Step 4: Creating hardware profile and loading model...")
            hardware_profile = HardwareProfile()))))))))
            backend=optimal_hardware,
            precision="fp16",
            optimization_level="performance"
            )
            print()))))))))f"\1{hardware_profile}\3")
    
    # Initialize worker with model
            worker.init_worker()))))))))[]]],,,model_name]),
            print()))))))))f"Model {model_name} initialized with worker")
            print())))))))))
    
    # Step 5: Run inference using worker
            print()))))))))"Step 5: Running inference using worker...")
            content = "This is a test sentence for inference."
            result = worker.accelerate()))))))))
            model_name=model_name,
            content=content,
            hardware_profile=hardware_profile
            )
            print()))))))))f"Inference result:")
            print()))))))))f"  - Latency: {result.get()))))))))'latency_ms', 0):.2f} ms")
            print()))))))))f"  - Throughput: {result.get()))))))))'throughput_items_per_second', 0):.2f} items/sec")
            print()))))))))f"  - Memory usage: {result.get()))))))))'memory_usage_mb', 0):.2f} MB")
            print()))))))))f"\1{result.get()))))))))'hardware_used', 'unknown')}\3")
            print())))))))))
    
    # Step 6: Using the higher-level ModelManager
            print()))))))))"Step 6: Using the higher-level ModelManager...")
            model_manager = ModelManager()))))))))worker)
    
    # Load model with automatic hardware selection
            model = model_manager.load_model()))))))))model_name)
            print()))))))))f"Model loaded with {model.get_current_hardware())))))))))} hardware")
    
    # Run inference with the loaded model
            start_time = time.time())))))))))
            embedding_result = model.get_embeddings()))))))))content)
            inference_time = time.time()))))))))) - start_time
            print()))))))))f"Inference completed in {inference_time*1000:.2f} ms")
            print())))))))))
    
    # Switch to a different hardware backend if available
    alternative_hardware = "cpu"  # CPU should always be available:
    if alternative_hardware != model.get_current_hardware()))))))))):
        print()))))))))f"Switching model to {alternative_hardware} hardware...")
        model.switch_hardware()))))))))alternative_hardware)
        print()))))))))f"Model now using {model.get_current_hardware())))))))))} hardware")
        
        # Run inference again on new hardware
        start_time = time.time())))))))))
        embedding_result = model.get_embeddings()))))))))content)
        inference_time = time.time()))))))))) - start_time
        print()))))))))f"Inference on {alternative_hardware} completed in {inference_time*1000:.2f} ms")
        print())))))))))
    
    # Step 7: Using ModelAccelerator for batch processing
        print()))))))))"Step 7: Using ModelAccelerator for batch processing...")
        model_accelerator = ModelAccelerator()))))))))worker)
    
    # Create a batch of inputs
        batch_content = []]],,,
        "This is the first sentence in the batch.",
        "This is the second sentence in the batch.",
        "This is the third sentence in the batch."
        ]
    
    # Run batch inference
        batch_results = model_accelerator.batch_accelerate()))))))))
        model_name=model_name,
        content_list=batch_content,
        hardware_profile=hardware_profile
        )
        print()))))))))f"Batch inference completed for {len()))))))))batch_results)} inputs")
        print())))))))))
    
    # Step 8: Run a benchmark
        print()))))))))"Step 8: Running a benchmark...")
        benchmark_config = BenchmarkConfig()))))))))
        model_names=[]]],,,model_name],
        hardware_profiles=[]]],,,
        HardwareProfile()))))))))backend=optimal_hardware),
        HardwareProfile()))))))))backend="cpu")
        ],
        metrics=[]]],,,"latency", "throughput", "memory"],
        iterations=3,  # Use a small number for the example
        warmup_iterations=1
        )
    
        benchmark = Benchmark()))))))))
        model_ids=[]]],,,model_name],
        hardware_profiles=benchmark_config.hardware_profiles,
        metrics=benchmark_config.metrics,
        worker=worker,
        config=benchmark_config
        )
    
    # Run benchmark
        benchmark_id, benchmark_results = benchmark.run())))))))))
        print()))))))))f"\1{benchmark_id}\3")
    
    # Generate a report
        report = benchmark.generate_report()))))))))results=benchmark_results, format="markdown")
        print()))))))))"\nBenchmark Report:")
        print()))))))))report)
        print())))))))))
    
    # Step 9: Quantize a model
        print()))))))))"Step 9: Quantizing a model...")
        quantization_engine = QuantizationEngine()))))))))worker)
    
    # Create a calibration dataset
        calibration_dataset = CalibrationDataset.from_examples()))))))))
        model_name=model_name,
        examples=[]]],,,
        "This is a sample sentence for calibration.",
        "Machine learning models benefit from proper quantization calibration.",
        "Multiple examples ensure representative activation distributions."
        ]
        )
    
    # Create quantization configuration
        quantization_config = QuantizationConfig()))))))))
        precision="int8",
        scheme="symmetric",
        per_channel=True,
        mixed_precision=True
        )
    
    # Quantize the model
        quantized_model = quantization_engine.quantize()))))))))
        model_name=model_name,
        hardware_profile=hardware_profile,
        quantization_config=quantization_config,
        calibration_dataset=calibration_dataset
        )
    
        print()))))))))f"Model quantized to {quantization_config.precision} precision")
        print()))))))))f"Compression ratio: {quantized_model.get()))))))))'compression_ratio', 0):.2f}x")
        print()))))))))f"Performance improvement: {quantized_model.get()))))))))'performance_improvement', 0):.2f}x")
        print()))))))))f"Accuracy impact: {quantized_model.get()))))))))'accuracy_impact', 0)*100:.2f}%")
        print())))))))))
    
    # Step 10: Compare quantized vs unquantized performance
        print()))))))))"Step 10: Comparing quantized vs unquantized performance...")
        comparison = quantization_engine.benchmark_comparison()))))))))
        model_name=model_name,
        quantized_model=quantized_model,
        hardware_profile=hardware_profile,
        metrics=[]]],,,"latency", "memory", "accuracy"]
        )
    
        latency_comp = comparison.get()))))))))"comparison", {}).get()))))))))"latency", {})
        memory_comp = comparison.get()))))))))"comparison", {}).get()))))))))"memory", {})
        accuracy_comp = comparison.get()))))))))"comparison", {}).get()))))))))"accuracy", {})
    
        print()))))))))"Performance comparison:")
    if latency_comp:
        print()))))))))f"  - Latency: {latency_comp.get()))))))))'baseline', 0):.2f} ms → {latency_comp.get()))))))))'quantized', 0):.2f} ms ())))))))){latency_comp.get()))))))))'improvement', 0):.2f}% improvement)")
    if memory_comp:
        print()))))))))f"  - Memory: {memory_comp.get()))))))))'baseline', 0):.2f} MB → {memory_comp.get()))))))))'quantized', 0):.2f} MB ())))))))){memory_comp.get()))))))))'reduction', 0):.2f}% reduction)")
    if accuracy_comp:
        print()))))))))f"  - Accuracy: {accuracy_comp.get()))))))))'baseline', 0)*100:.2f}% → {accuracy_comp.get()))))))))'quantized', 0)*100:.2f}% ())))))))){accuracy_comp.get()))))))))'relative_loss', 0):.2f}% loss)")
        print())))))))))
    
        print()))))))))"Example completed successfully!")

if __name__ == "__main__":
    main())))))))))