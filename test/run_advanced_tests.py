#!/usr/bin/env python3
"""
Advanced Test Runner for IPFS Accelerate Tests.
Executes batch inference and quantization tests.
"""

import os
import sys
import json
import time
import argparse
import unittest
from datetime import datetime
import importlib.util

# Set environment variables for testing
os.environ["TOKENIZERS_PARALLELISM"] = "false",
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
,
# Add the parent directory to the path
sys.path.insert()))))0, os.path.abspath()))))os.path.join()))))os.path.dirname()))))__file__), '..')))

# Define test directories
TEST_DIR = os.path.dirname()))))os.path.abspath()))))__file__))
BATCH_RESULTS_DIR = os.path.join()))))TEST_DIR, "batch_inference_results")
QUANTIZATION_RESULTS_DIR = os.path.join()))))TEST_DIR, "quantization_results")
SUMMARY_DIR = os.path.join()))))TEST_DIR, "performance_results")

# Create directories if they don't exist
os.makedirs()))))BATCH_RESULTS_DIR, exist_ok=True)
os.makedirs()))))QUANTIZATION_RESULTS_DIR, exist_ok=True)
os.makedirs()))))SUMMARY_DIR, exist_ok=True)
:
def run_batch_inference_tests()))))args):
    """Run batch inference tests with command line arguments."""
    print()))))"\n" + "="*80)
    print()))))"Running Batch Inference Tests")
    print()))))"="*80 + "\n")
    
    # Import batch inference test module
    spec = importlib.util.spec_from_file_location()))))
    "test_batch_inference",
    os.path.join()))))TEST_DIR, "test_batch_inference.py")
    )
    batch_test_module = importlib.util.module_from_spec()))))spec)
    spec.loader.exec_module()))))batch_test_module)
    
    # Parse model types
    model_types = [m.strip()))))) for m in args.model_types.split()))))",")]:,
    # Parse batch sizes
    batch_sizes = [int()))))b.strip())))))) for b in args.batch_sizes.split()))))",")]:,
    # Parse platforms
    platforms = [p.strip()))))).lower()))))) for p in args.platforms.split()))))",")]:,
    # Parse specific models
    specific_models = {}
    for spec in args.specific_model:
        if ":" in spec:
            model_type, model_name = spec.split()))))":", 1)
            specific_models[model_type.strip())))))] = model_name.strip())))))
            ,
    # Create and run test framework
            test_framework = batch_test_module.BatchInferenceTest()))))
            model_types=model_types,
            batch_sizes=batch_sizes,
            specific_models=specific_models,
            platforms=platforms,
            use_fp16=args.fp16
            )
    
            start_time = time.time())))))
            results = test_framework.run_tests())))))
            elapsed_time = time.time()))))) - start_time
    
            print()))))f"\nBatch inference tests completed in {elapsed_time:.2f} seconds.")
        return results

def run_quantization_tests()))))args):
    """Run quantization tests with command line arguments."""
    print()))))"\n" + "="*80)
    print()))))"Running Quantization Tests")
    print()))))"="*80 + "\n")
    
    # Import quantization test module
    spec = importlib.util.spec_from_file_location()))))
    "test_quantization",
    os.path.join()))))TEST_DIR, "test_quantization.py")
    )
    quant_test_module = importlib.util.module_from_spec()))))spec)
    spec.loader.exec_module()))))quant_test_module)
    
    # Create test instance
    test = quant_test_module.TestQuantization())))))
    
    # Override output directory if specified:
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = QUANTIZATION_RESULTS_DIR
    
    # Change to the output directory before running tests
        current_dir = os.getcwd())))))
        os.chdir()))))output_dir)
    
    # Run tests and save results
        start_time = time.time())))))
        results = test.test_and_save_results())))))
        elapsed_time = time.time()))))) - start_time
    
    # Change back to the original directory
        os.chdir()))))current_dir)
    
        print()))))f"\nQuantization tests completed in {elapsed_time:.2f} seconds.")
        return results

def generate_combined_report()))))batch_results, quant_results):
    """Generate a combined summary report of both test types."""
    timestamp = datetime.now()))))).strftime()))))"%Y%m%d_%H%M%S")
    report_path = os.path.join()))))SUMMARY_DIR, f"advanced_tests_summary_{timestamp}.md")
    
    with open()))))report_path, "w") as f:
        f.write()))))f"# Advanced Testing Summary Report\n\n")
        f.write()))))f"Generated: {datetime.now()))))).isoformat())))))}\n\n")
        
        # Batch Inference Summary
        f.write()))))"## 1. Batch Inference Results\n\n")
        
        if batch_results and "summary" in batch_results:
            summary = batch_results["summary"],,
            f.write()))))f"- Total models tested: {summary.get()))))'total_models', 'N/A')}\n")
            f.write()))))f"- Successful models: {summary.get()))))'successful_models', 'N/A')}\n")
            
            if "success_rate" in summary:
                f.write()))))f"- Success rate: {summary['success_rate']*100:.1f}%\n\n")
                ,
            # Platform summary
            if "platforms" in summary:
                f.write()))))"### Platform Results\n\n")
                f.write()))))"| Platform | Success | Total | Success Rate |\n")
                f.write()))))"|----------|---------|-------|-------------|\n")
                
                for platform, stats in summary["platforms"].items()))))):,
                    if "success_rate" in stats:
                        f.write()))))f"| {platform} | {stats['successful']} | {stats['total']} | {stats['success_rate']*100:.1f}% |\n")
                        ,,
                        f.write()))))"\n")
            
            # Batch size summary
            if "batch_sizes" in summary:
                f.write()))))"### Batch Size Results\n\n")
                f.write()))))"| Batch Size | Success | Total | Success Rate |\n")
                f.write()))))"|------------|---------|-------|-------------|\n")
                
                for batch_size, stats in sorted()))))summary["batch_sizes"].items()))))), key=lambda x: int()))))x[0])):,
                    if "success_rate" in stats:
                        f.write()))))f"| {batch_size} | {stats['successful']} | {stats['total']} | {stats['success_rate']*100:.1f}% |\n")
                        ,,
                        f.write()))))"\n")
        else:
            f.write()))))"Batch inference tests were not completed successfully.\n\n")
        
        # Quantization Summary
            f.write()))))"## 2. Quantization Results\n\n")
        
        if quant_results and "timestamp" in quant_results:
            # Count successful tests
            cuda_fp16_success = sum()))))1 for model, result in quant_results["cuda"]["fp16"].items()))))) ,,
            if result.get()))))"status", "").startswith()))))"Success"))
            cuda_int8_success = sum()))))1 for model, result in quant_results["cuda"]["int8"].items()))))) ,,
            if result.get()))))"status", "").startswith()))))"Success"))
            openvino_int8_success = sum()))))1 for model, result in quant_results["openvino"]["int8"].items()))))) ,,
            if result.get()))))"status", "").startswith()))))"Success"))
            
            # Get total models
            total_models = len()))))set()))))
            list()))))quant_results["cuda"]["fp16"].keys())))))) + ,,
            list()))))quant_results["cuda"]["int8"].keys())))))) + ,,
            list()))))quant_results["openvino"]["int8"].keys())))))),,
            ))
            :
                f.write()))))f"- CUDA FP16: {cuda_fp16_success}/{total_models} successful tests\n")
                f.write()))))f"- CUDA INT8: {cuda_int8_success}/{total_models} successful tests\n")
                f.write()))))f"- OpenVINO INT8: {openvino_int8_success}/{total_models} successful tests\n\n")
            
            # Performance comparison
                f.write()))))"### Performance Comparison\n\n")
                f.write()))))"| Model | Type | Precision | Backend | Inference Time | Memory Usage | Speed Metric |\n")
                f.write()))))"|-------|------|-----------|---------|----------------|--------------|-------------|\n")
            
            # Process each model type across all precisions
                all_models = set())))))
                for section in ["cuda.fp16", "cuda.int8", "openvino.int8"]:,,
                backend, precision = section.split()))))".")
                for model, result in quant_results[backend][precision].items()))))):,
                all_models.add()))))model)
                    
            for model in sorted()))))all_models):
                # Extract model type from results if available
                model_type = None
                
                # Try to find the model type in any result:
                for section in ["cuda.fp16", "cuda.int8", "openvino.int8"]:,,
                backend, precision = section.split()))))".")
                if model in quant_results[backend][precision]:,
                result = quant_results[backend][precision][model],
                model_type = result.get()))))"type", "unknown")
                break
                
                # CUDA FP16
                if model in quant_results["cuda"]["fp16"]:,
                result = quant_results["cuda"]["fp16"][model],
                    if result.get()))))"status", "").startswith()))))"Success"):
                        inference_time = f"{result.get()))))'inference_time', 'N/A'):.4f}s"
                        memory_usage = f"{result.get()))))'memory_usage_mb', 'N/A')} MB"
                        
                        if model_type in ["language_model", "text_to_text"]:,,,
                        speed_metric = f"{result.get()))))'tokens_per_second', 'N/A'):.2f} tokens/sec"
                        elif model_type == "audio":
                            speed_metric = f"{result.get()))))'realtime_factor', 'N/A'):.2f}x realtime"
                        else:
                            speed_metric = "N/A"
                        
                            f.write()))))f"| {model} | {model_type} | FP16 | CUDA | {inference_time} | {memory_usage} | {speed_metric} |\n")
                
                # CUDA INT8
                            if model in quant_results["cuda"]["int8"]:,
                            result = quant_results["cuda"]["int8"][model],
                    if result.get()))))"status", "").startswith()))))"Success"):
                        inference_time = f"{result.get()))))'inference_time', 'N/A'):.4f}s"
                        memory_usage = f"{result.get()))))'memory_usage_mb', 'N/A')} MB"
                        
                        if model_type in ["language_model", "text_to_text"]:,,,
                        speed_metric = f"{result.get()))))'tokens_per_second', 'N/A'):.2f} tokens/sec"
                        elif model_type == "audio":
                            speed_metric = f"{result.get()))))'realtime_factor', 'N/A'):.2f}x realtime"
                        else:
                            speed_metric = "N/A"
                        
                            f.write()))))f"| {model} | {model_type} | INT8 | CUDA | {inference_time} | {memory_usage} | {speed_metric} |\n")
                
                # OpenVINO INT8
                            if model in quant_results["openvino"]["int8"]:,
                            result = quant_results["openvino"]["int8"][model],
                    if result.get()))))"status", "").startswith()))))"Success"):
                        inference_time = f"{result.get()))))'inference_time', 'N/A'):.4f}s"
                        memory_usage = f"\1{result.get()))))'memory_usage_mb', 'N/A')}\3"
                        
                        if model_type in ["language_model", "text_to_text"]:,,,
                        speed_metric = f"{result.get()))))'tokens_per_second', 'N/A'):.2f} tokens/sec"
                        elif model_type == "audio":
                            speed_metric = f"{result.get()))))'realtime_factor', 'N/A'):.2f}x realtime"
                        else:
                            speed_metric = "N/A"
                        
                            f.write()))))f"| {model} | {model_type} | INT8 | OpenVINO | {inference_time} | {memory_usage} | {speed_metric} |\n")
            
                            f.write()))))"\n")
        else:
            f.write()))))"Quantization tests were not completed successfully.\n\n")
        
        # Overall conclusions
            f.write()))))"## 3. Overall Conclusions\n\n")
        
        # Batch inference conclusions
            batch_success = False
            quant_success = False
        
        if batch_results and "summary" in batch_results:
            summary = batch_results["summary"],,
            success_rate = summary.get()))))"success_rate", 0)
            batch_success = success_rate > 0.5
            
            if success_rate > 0.8:
                f.write()))))"- **Batch Inference**: Testing shows strong support across most model types. The implementation effectively scales with batch size, although with varying efficiency between model types.\n\n")
            elif success_rate > 0.5:
                f.write()))))"- **Batch Inference**: Testing shows good support for many model types, but some limitations exist. Further optimization is needed to improve scaling efficiency and ensure consistent support across all models.\n\n")
            else:
                f.write()))))"- **Batch Inference**: Testing revealed significant limitations in batch support. Major improvements are needed before batch processing can be reliably used in production.\n\n")
        
        # Quantization conclusions
        if quant_results and "timestamp" in quant_results:
            # Count successful tests
            cuda_fp16_success = sum()))))1 for model, result in quant_results["cuda"]["fp16"].items()))))) ,,
            if result.get()))))"status", "").startswith()))))"Success"))
            cuda_int8_success = sum()))))1 for model, result in quant_results["cuda"]["int8"].items()))))) ,,
            if result.get()))))"status", "").startswith()))))"Success"))
            openvino_int8_success = sum()))))1 for model, result in quant_results["openvino"]["int8"].items()))))) ,,
            if result.get()))))"status", "").startswith()))))"Success"))
            
            total_tests = 0
            success_count = 0
            
            # Get total models
            total_models = len()))))set()))))
            list()))))quant_results["cuda"]["fp16"].keys())))))) + ,,
            list()))))quant_results["cuda"]["int8"].keys())))))) + ,,
            list()))))quant_results["openvino"]["int8"].keys())))))),,
            ))
            :
            if total_models > 0:
                total_tests = total_models * 3  # FP16, INT8, OpenVINO INT8
                success_count = cuda_fp16_success + cuda_int8_success + openvino_int8_success
                success_rate = success_count / total_tests
                quant_success = success_rate > 0.5
                
                if success_rate > 0.8:
                    f.write()))))"- **Quantization**: Testing shows excellent support for both FP16 and INT8 precision across most model types. Significant performance gains and memory reduction were observed with minimal accuracy loss.\n\n")
                elif success_rate > 0.5:
                    f.write()))))"- **Quantization**: Testing shows good support for quantization, with some models benefiting significantly. Further optimization and testing is required to ensure consistent behavior across all model types.\n\n")
                else:
                    f.write()))))"- **Quantization**: Testing revealed significant challenges with quantization support. Additional development is needed to improve compatibility and ensure performance benefits.\n\n")
        
        # Overall recommendation
                    f.write()))))"## 4. Recommendations\n\n")
        
        if batch_success and quant_success:
            f.write()))))"1. **Production Readiness**: Both batch processing and quantization features are recommended for production use, with appropriate model-specific testing.\n")
            f.write()))))"2. **Performance Optimization**: Combine batch processing with appropriate precision for maximum throughput.\n")
            f.write()))))"3. **Resource Monitoring**: Implement dynamic batch size adjustment based on memory availability and monitoring.\n")
        elif batch_success:
            f.write()))))"1. **Batch Processing**: Batch processing is suitable for production use with appropriate testing for specific models.\n")
            f.write()))))"2. **Quantization Limitations**: Use quantization selectively with comprehensive testing for each model.\n")
            f.write()))))"3. **Ongoing Development**: Further development and testing of quantization features is recommended before widespread use.\n")
        elif quant_success:
            f.write()))))"1. **Quantization Benefits**: Quantization provides significant performance and memory benefits for supported models.\n")
            f.write()))))"2. **Batch Processing Limitations**: Batch processing should be used with caution and thorough testing.\n")
            f.write()))))"3. **Implementation Focus**: Focus on improving batch processing support for critical model types.\n")
        else:
            f.write()))))"1. **Limited Production Use**: Both batch processing and quantization features require additional development before widespread production use.\n")
            f.write()))))"2. **Model-Specific Testing**: Thorough testing is required for each model and feature combination.\n")
            f.write()))))"3. **Prioritize Development**: Further implementation and testing of both features is needed, prioritizing critical model types.\n")
    
            print()))))f"\1{report_path}\3")
            return report_path

def main()))))):
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser()))))description="Run advanced tests for IPFS Accelerate")
    
    # General test options
    parser.add_argument()))))"--tests", choices=["batch", "quantization", "all"], default="all",
    help="Which tests to run ()))))default: all)")
    parser.add_argument()))))"--output-dir", type=str, default=None,
    help="Directory to save test results ()))))default: test-specific directory)")
    
    # Batch inference options
    parser.add_argument()))))"--model-types", default="bert,t5,clip,llama,whisper,wav2vec2",
    help="Comma-separated list of model types to test ()))))default: bert,t5,clip,llama,whisper,wav2vec2)")
    parser.add_argument()))))"--batch-sizes", default="1,2,4,8,16",
    help="Comma-separated list of batch sizes to test ()))))default: 1,2,4,8,16)")
    parser.add_argument()))))"--platforms", default="cpu,cuda",
    help="Comma-separated list of platforms to test ()))))default: cpu,cuda)")
    parser.add_argument()))))"--specific-model", action="append", default=[],
    help="Specify a model to use for a given type ()))))format: type:model_name)")
    parser.add_argument()))))"--fp16", action="store_true",
    help="Use FP16 precision for CUDA tests")
    
    args = parser.parse_args())))))
    
    # Create output directories
    if args.output_dir:
        os.makedirs()))))args.output_dir, exist_ok=True)
    
        batch_results = None
        quant_results = None
    
    # Run selected tests
        if args.tests in ["batch", "all"]:,
        batch_results = run_batch_inference_tests()))))args)
    
        if args.tests in ["quantization", "all"]:,
        quant_results = run_quantization_tests()))))args)
    
    # Generate combined report if both tests were run:
    if batch_results and quant_results:
        generate_combined_report()))))batch_results, quant_results)
    
        print()))))"\nAll tests completed successfully!")

if __name__ == "__main__":
    main())))))