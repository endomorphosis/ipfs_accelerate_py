#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for the refactored benchmark suite.

This script validates that the enhanced benchmark code correctly implements
hardware-aware metrics and provides detailed insights across different hardware platforms.
"""

import os
import sys
import re
import logging
import json
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("validation")

# Add the current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def validate_flops_metric():
    """Validate the hardware-aware FLOPs metric implementation."""
    logger.info("Validating hardware-aware FLOPs metric implementation...")
    
    # Import the FLOPs metric module
    try:
        from metrics.flops import FLOPsMetric, FLOPsMetricFactory
        
        # Define validation points for the hardware-aware FLOPs metric
        validation_points = {
            "Hardware Efficiency Factors": [
                "_get_hardware_efficiency_factor",
                "hardware_efficiency",
                "device_type"
            ],
            "Architecture-Specific Calculations": [
                "_detect_model_type", 
                "_estimate_attention_flops",
                "_estimate_feed_forward_flops",
                "_get_transformer_config"
            ],
            "Tensor Core Detection": [
                "_has_tensor_core_operations",
                "tensor_core_eligible"
            ],
            "Component-Level Breakdown": [
                "attention",
                "feed_forward",
                "embedding"
            ],
            "Attention Type Detection": [
                "multi_query_attn",
                "grouped_query_attn",
                "kv_heads"
            ],
            "Precision Detection": [
                "_get_model_precision_info",
                "dominant_precision",
                "precision_float"
            ]
        }
        
        # Read the metric code
        flops_file = os.path.join(current_dir, "metrics", "flops.py")
        with open(flops_file, 'r') as f:
            flops_code = f.read()
        
        all_validated = True
        
        # Check for each validation point
        for feature, checklist in validation_points.items():
            feature_present = all(item in flops_code for item in checklist)
            status = "✓" if feature_present else "✗"
            logger.info(f"{status} {feature} {'present' if feature_present else 'missing'}")
            
            if not feature_present:
                all_validated = False
                missing = [item for item in checklist if item not in flops_code]
                logger.info(f"  Missing: {', '.join(missing)}")
        
        # Check for specific patterns indicating enhanced implementation
        enhanced_patterns = {
            "Hardware-Specific Optimization": r"hardware_efficiency.*cuda.*gpu|efficiency.*tensor_core",
            "Model Type Detection": r"_detect_model_type.*return.*transformer|return.*cnn|return.*multimodal",
            "Attention Type Detection": r"multi_query_attn|grouped_query_attn|config.kv_heads",
            "Detailed Metrics": r"get_detailed_metrics.*detailed_flops"
        }
        
        for feature, pattern in enhanced_patterns.items():
            pattern_match = re.search(pattern, flops_code, re.DOTALL)
            status = "✓" if pattern_match else "✗"
            logger.info(f"{status} {feature} implementation {'found' if pattern_match else 'missing'}")
            
            if not pattern_match:
                all_validated = False
        
        # Check for tests validating the enhanced features
        test_file = os.path.join(current_dir, "tests", "test_flops_metric.py")
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_code = f.read()
            
            test_patterns = {
                "Hardware Efficiency Test": r"test_hardware_efficiency|gpu_efficiency.*cpu_efficiency",
                "Attention Type Test": r"test_attention|multi.*query.*attention|grouped.*query.*attention",
                "Component Breakdown Test": r"test_detailed_metrics.*attention.*feed_forward",
                "Precision Detection Test": r"test_precision|fp16|float16"
            }
            
            for test_feature, test_pattern in test_patterns.items():
                test_match = re.search(test_pattern, test_code, re.DOTALL)
                status = "✓" if test_match else "✗"
                logger.info(f"{status} {test_feature} {'present' if test_match else 'missing'}")
                
                if not test_match:
                    all_validated = False
        else:
            logger.error(f"Test file not found at {test_file}")
            all_validated = False
        
        return all_validated
        
    except ImportError as e:
        logger.error(f"Import error validating FLOPs metric: {e}")
        return False
    except Exception as e:
        logger.error(f"Error validating FLOPs metric: {e}")
        return False

def validate_dashboard_visualization():
    """Validate dashboard visualization to ensure it handles hardware-aware metrics."""
    logger.info("Validating dashboard visualization code...")
    
    # Check if the dashboard.py file exists
    dashboard_file = os.path.join(current_dir, "visualizers", "dashboard.py")
    if not os.path.exists(dashboard_file):
        logger.error(f"Dashboard file not found at {dashboard_file}")
        return False
    
    # Read the dashboard.py file
    try:
        with open(dashboard_file, 'r') as f:
            dashboard_code = f.read()
        
        # Define validation points
        validation_points = {
            "Latency Percentiles": [
                "latency_p90_ms", 
                "latency_p95_ms", 
                "latency_p99_ms"
            ],
            "Detailed Memory Metrics": [
                "memory_peak_mb", 
                "memory_allocated_end_mb",
                "memory_reserved_end_mb",
                "cpu_memory_end_mb"
            ],
            "FLOPs Metrics": ["flops", "gflops"],
            "Chart Type Selection": ["chartTypeSelect", "selectedChartType"],
            "Enhanced Visualization Functions": [
                "updateMainChart", 
                "updateLatencyPercentileChart", 
                "updateMemoryBreakdownChart"
            ],
            "Hardware-Aware Table Columns": [
                "<th>p90 (ms)</th>",
                "<th>p99 (ms)</th>",
                "<th>Peak Memory (MB)</th>",
                "<th>GFLOPs</th>"
            ],
            "Hardware-Specific Labels": [
                "hardware.toUpperCase()",
                "CPU</th>",
                "CUDA</th>",
                "GPU</th>"
            ],
            "FLOPs Visualization": [
                "gflops",
                "formatNumber(d.gflops)",
                "flops_per_token",
                "flops_per_parameter",
                "hardware_efficiency"
            ]
        }
        
        all_validated = True
        
        # Check for each feature
        for feature, checklist in validation_points.items():
            feature_present = all(item in dashboard_code for item in checklist)
            status = "✓" if feature_present else "✗"
            logger.info(f"{status} {feature} {'present' if feature_present else 'missing'}")
            
            if not feature_present:
                all_validated = False
                missing = [item for item in checklist if item not in dashboard_code]
                logger.info(f"  Missing: {', '.join(missing)}")
        
        # Check for specific patterns indicating function enhancement
        enhanced_patterns = {
            "Dynamic Chart Type": r"selectedChartType.*case\s+['\"]bar['\"]|case\s+['\"]scatter['\"]",
            "Latency Percentile Visualization": r"latency_p\d+_ms.*\.map\(d\s+=>\s+d\.batch_size\)",
            "Memory Breakdown Chart": r"memory_peak_mb.*stacked\s+bar\s+chart|barmode:\s+['\"]stack['\"]",
            "GFLOPs Formatting": r"gflops.*formatNumber|gflops_value.*1000"
        }
        
        for feature, pattern in enhanced_patterns.items():
            pattern_match = re.search(pattern, dashboard_code, re.DOTALL)
            status = "✓" if pattern_match else "✗"
            logger.info(f"{status} {feature} implementation {'found' if pattern_match else 'missing'}")
            
            if not pattern_match:
                all_validated = False
        
        # Check for hardware-specific filtering and visualization
        hardware_patterns = {
            "Hardware Filtering": r"selectedHardware.*hardware.*filter",
            "Hardware Comparison": r"hardware_tested.*comparison|multiple\s+hardware.*comparison",
            "Hardware-Specific Metrics": r"hardware_efficiency|cuda_device_name|tensor_core_eligible"
        }
        
        for feature, pattern in hardware_patterns.items():
            pattern_match = re.search(pattern, dashboard_code, re.DOTALL)
            status = "✓" if pattern_match else "✗"
            logger.info(f"{status} {feature} visualization {'found' if pattern_match else 'missing'}")
            
            if not pattern_match:
                all_validated = False
        
        return all_validated
        
    except Exception as e:
        logger.error(f"Error validating dashboard code: {e}")
        return False

def validate_benchmark_metrics():
    """Validate that benchmark.py correctly implements hardware-aware metrics."""
    logger.info("Validating benchmark metrics implementation...")
    
    # Check if benchmark.py exists
    benchmark_file = os.path.join(current_dir, "benchmark.py")
    if not os.path.exists(benchmark_file):
        logger.error(f"Benchmark file not found at {benchmark_file}")
        return False
    
    try:
        with open(benchmark_file, 'r') as f:
            benchmark_code = f.read()
        
        # Define validation points
        validation_points = {
            "Hardware Detection": [
                "get_available_hardware",
                "initialize_hardware"
            ],
            "Hardware-Aware Metrics": [
                "TimingMetricFactory",
                "MemoryMetricFactory",
                "FLOPsMetricFactory"
            ],
            "Factory Pattern": [
                "create_latency_metric",
                "create_throughput_metric",
                "FLOPsMetricFactory.create"
            ],
            "Multiple Hardware Support": [
                "hardware: List[str]",
                "for hw in self.config.hardware"
            ],
            "Hardware Info": [
                "get_hardware_info",
                "gpu_theoretical_tflops",
                "hardware_efficiency"
            ]
        }
        
        all_validated = True
        
        # Check for each feature
        for feature, checklist in validation_points.items():
            feature_present = all(item in benchmark_code for item in checklist)
            status = "✓" if feature_present else "✗"
            logger.info(f"{status} {feature} {'present' if feature_present else 'missing'}")
            
            if not feature_present:
                all_validated = False
                missing = [item for item in checklist if item not in benchmark_code]
                logger.info(f"  Missing: {', '.join(missing)}")
        
        # Check for specific patterns indicating enhanced implementation
        enhanced_patterns = {
            "Hardware Configuration Validation": r"validate.*hardware|available_hw.*hardware",
            "Hardware-Specific Metric Creation": r"metric.*hardware|device.*metric",
            "Hardware-Aware Results Export": r"hardware_tested|results\[hw\]|device.*hardware",
            "Hardware Efficiency Reporting": r"hardware_efficiency|speedup.*cpu.*gpu|efficiency"
        }
        
        for feature, pattern in enhanced_patterns.items():
            pattern_match = re.search(pattern, benchmark_code, re.DOTALL)
            status = "✓" if pattern_match else "✗"
            logger.info(f"{status} {feature} implementation {'found' if pattern_match else 'missing'}")
            
            if not pattern_match:
                all_validated = False
        
        return all_validated
        
    except Exception as e:
        logger.error(f"Error validating benchmark code: {e}")
        return False

def validate_power_metric():
    """Validate power metrics implementation."""
    logger.info("Validating power metrics implementation...")
    
    # Check if the power.py file exists
    power_file = os.path.join(current_dir, "metrics", "power.py")
    if not os.path.exists(power_file):
        logger.error(f"Power metrics file not found at {power_file}")
        return False
    
    try:
        with open(power_file, 'r') as f:
            power_code = f.read()
        
        # Define validation points
        validation_points = {
            "Platform Detection": [
                "_check_nvidia_smi",
                "_check_intel_rapl",
                "_check_rocm_smi",
                "_check_powermetrics"
            ],
            "Power Sampling": [
                "_sample_power",
                "sampling_thread",
                "sampling_rate_ms"
            ],
            "Platform-Specific Power Reading": [
                "_get_nvidia_power",
                "_get_intel_rapl_power",
                "_get_rocm_power",
                "_get_current_power"
            ],
            "Efficiency Metrics": [
                "set_operations_count",
                "set_throughput",
                "ops_per_watt",
                "gflops_per_watt",
                "throughput_per_watt"
            ],
            "Factory Pattern": [
                "PowerMetricFactory",
                "_get_device_type"
            ]
        }
        
        all_validated = True
        
        # Check for each feature
        for feature, checklist in validation_points.items():
            feature_present = all(item in power_code for item in checklist)
            status = "✓" if feature_present else "✗"
            logger.info(f"{status} {feature} {'present' if feature_present else 'missing'}")
            
            if not feature_present:
                all_validated = False
                missing = [item for item in checklist if item not in power_code]
                logger.info(f"  Missing: {', '.join(missing)}")
        
        # Check for specific patterns indicating enhanced implementation
        enhanced_patterns = {
            "Threaded Sampling": r"threading\.Thread|daemon\s*=\s*True",
            "Power Calculation": r"avg_power\s*=\s*sum\(.*\)\s*/\s*len\(.*\)",
            "Energy Calculation": r"energy_joules\s*=\s*avg_power\s*\*\s*duration",
            "Efficiency Calculation": r"ops_per_watt\s*=.*operations_count.*avg_power",
            "Hardware-Specific Detection": r"device_type.*cuda.*nvidia|device_type.*cpu.*intel_rapl"
        }
        
        for feature, pattern in enhanced_patterns.items():
            pattern_match = re.search(pattern, power_code, re.DOTALL)
            status = "✓" if pattern_match else "✗"
            logger.info(f"{status} {feature} implementation {'found' if pattern_match else 'missing'}")
            
            if not pattern_match:
                all_validated = False
        
        # Check for integration with benchmark
        benchmark_file = os.path.join(current_dir, "benchmark.py")
        if os.path.exists(benchmark_file):
            with open(benchmark_file, 'r') as f:
                benchmark_code = f.read()
            
            power_integration_points = [
                "PowerMetric",
                "PowerMetricFactory",
                "power_metric",
                "power_avg_watts",
                "energy_joules",
                "gflops_per_watt"
            ]
            
            integration_present = any(point in benchmark_code for point in power_integration_points)
            status = "✓" if integration_present else "✗"
            logger.info(f"{status} Power metrics integration with benchmark {'present' if integration_present else 'missing'}")
            
            if not integration_present:
                all_validated = False
        
        # Check for tests validating power metrics
        test_file = os.path.join(current_dir, "tests", "test_power_metric.py")
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_code = f.read()
            
            test_patterns = {
                "Platform Detection Test": r"test_platform_detection|check_.*smi",
                "Power Sampling Test": r"test_power_sampling|sample_power",
                "Efficiency Metrics Test": r"test_efficiency_metrics|test_.*per_watt",
                "Integration Test": r"test_integration_with_benchmark|benchmark.*power"
            }
            
            for test_feature, test_pattern in test_patterns.items():
                test_match = re.search(test_pattern, test_code, re.DOTALL)
                status = "✓" if test_match else "✗"
                logger.info(f"{status} {test_feature} {'present' if test_match else 'missing'}")
                
                if not test_match:
                    all_validated = False
        else:
            logger.warning(f"Test file not found at {test_file}")
            logger.warning("You should create tests for power metrics")
            all_validated = False
        
        return all_validated
        
    except Exception as e:
        logger.error(f"Error validating power metrics: {e}")
        return False

def validate_bandwidth_metric():
    """Validate memory bandwidth metrics implementation."""
    logger.info("Validating memory bandwidth metrics implementation...")
    
    # Check if the bandwidth.py file exists
    bandwidth_file = os.path.join(current_dir, "metrics", "bandwidth.py")
    if not os.path.exists(bandwidth_file):
        logger.error(f"Bandwidth metrics file not found at {bandwidth_file}")
        return False
    
    try:
        with open(bandwidth_file, 'r') as f:
            bandwidth_code = f.read()
        
        # Define validation points
        validation_points = {
            "Platform Detection": [
                "_get_theoretical_peak_bandwidth",
                "_get_cpu_peak_bandwidth",
                "_get_cuda_peak_bandwidth",
                "_get_rocm_peak_bandwidth"
            ],
            "Bandwidth Measurement": [
                "_sample_bandwidth",
                "_get_current_bandwidth",
                "bandwidth_samples"
            ],
            "Memory Transfer Estimation": [
                "set_memory_transfers",
                "estimate_memory_transfers",
                "memory_transfers_bytes"
            ],
            "Roofline Model": [
                "get_arithmetic_intensity",
                "is_compute_bound",
                "get_roofline_data",
                "ridge_point"
            ],
            "Factory Pattern": [
                "BandwidthMetricFactory",
                "_get_device_type"
            ]
        }
        
        all_validated = True
        
        # Check for each feature
        for feature, checklist in validation_points.items():
            feature_present = all(item in bandwidth_code for item in checklist)
            status = "✓" if feature_present else "✗"
            logger.info(f"{status} {feature} {'present' if feature_present else 'missing'}")
            
            if not feature_present:
                all_validated = False
                missing = [item for item in checklist if item not in bandwidth_code]
                logger.info(f"  Missing: {', '.join(missing)}")
        
        # Check for specific patterns indicating enhanced implementation
        enhanced_patterns = {
            "Dynamic Sampling": r"threading\.Thread|daemon\s*=\s*True|sampling_loop",
            "Hardware Specific Calculations": r"cuda.*bandwidth|rocm.*bandwidth|theoretical_peak_bandwidth",
            "Arithmetic Intensity Calculation": r"compute_operations\s*/\s*.*memory_transfers_bytes",
            "Roofline Model Analysis": r"ridge_point|compute_bound|memory_bound"
        }
        
        for feature, pattern in enhanced_patterns.items():
            pattern_match = re.search(pattern, bandwidth_code, re.DOTALL)
            status = "✓" if pattern_match else "✗"
            logger.info(f"{status} {feature} implementation {'found' if pattern_match else 'missing'}")
            
            if not pattern_match:
                all_validated = False
        
        # Check for integration with benchmark
        benchmark_file = os.path.join(current_dir, "benchmark.py")
        if os.path.exists(benchmark_file):
            with open(benchmark_file, 'r') as f:
                benchmark_code = f.read()
            
            bandwidth_integration_points = [
                "BandwidthMetric",
                "BandwidthMetricFactory",
                "bandwidth_metric",
                "avg_bandwidth_gbps",
                "bandwidth_utilization_percent",
                "arithmetic_intensity"
            ]
            
            integration_present = any(point in benchmark_code for point in bandwidth_integration_points)
            status = "✓" if integration_present else "✗"
            logger.info(f"{status} Bandwidth metrics integration with benchmark {'present' if integration_present else 'missing'}")
            
            if not integration_present:
                all_validated = False
        
        # Check for tests validating bandwidth metrics
        test_file = os.path.join(current_dir, "tests", "test_bandwidth_metric.py")
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_code = f.read()
            
            test_patterns = {
                "Peak Bandwidth Test": r"test_peak_bandwidth|theoretical.*bandwidth",
                "Memory Transfer Test": r"test_memory_transfer|estimate_memory_transfers",
                "Roofline Model Test": r"test_roofline|arithmetic_intensity|compute_bound",
                "Integration Test": r"test_integration_with_benchmark|benchmark.*bandwidth"
            }
            
            for test_feature, test_pattern in test_patterns.items():
                test_match = re.search(test_pattern, test_code, re.DOTALL)
                status = "✓" if test_match else "✗"
                logger.info(f"{status} {test_feature} {'present' if test_match else 'missing'}")
                
                if not test_match:
                    all_validated = False
        else:
            logger.warning(f"Test file not found at {test_file}")
            logger.warning("You should create tests for bandwidth metrics")
            all_validated = False
        
        return all_validated
        
    except Exception as e:
        logger.error(f"Error validating bandwidth metrics: {e}")
        return False

def validate_all():
    """Run all validation checks."""
    logger.info("Running validation for hardware-aware benchmark metrics")
    
    # Track validation results
    flops_valid = validate_flops_metric()
    dashboard_valid = validate_dashboard_visualization()
    benchmark_valid = validate_benchmark_metrics()
    power_valid = validate_power_metric()
    bandwidth_valid = validate_bandwidth_metric()
    
    # Summarize result
    logger.info("\n--- Validation Summary ---")
    logger.info(f"{'✓' if flops_valid else '✗'} Hardware-Aware FLOPs Metric")
    logger.info(f"{'✓' if dashboard_valid else '✗'} Dashboard Visualization")
    logger.info(f"{'✓' if benchmark_valid else '✗'} Benchmark Implementation")
    logger.info(f"{'✓' if power_valid else '✗'} Power Metrics Implementation")
    logger.info(f"{'✓' if bandwidth_valid else '✗'} Memory Bandwidth Metrics")
    
    if flops_valid and dashboard_valid and benchmark_valid and power_valid and bandwidth_valid:
        logger.info("\n🎉 Hardware-aware benchmark metrics validation passed!")
        logger.info("The refactored benchmark suite correctly implements hardware-aware metrics.")
        return True
    else:
        logger.error("\n❌ Hardware-aware benchmark metrics validation failed.")
        logger.error("Some features are not properly implemented.")
        
        # Provide specific guidance on what needs to be fixed
        if not flops_valid:
            logger.error("\nFLOPs Metric Issues:")
            logger.error("- Ensure hardware-specific optimizations are correctly implemented")
            logger.error("- Add support for different attention mechanisms (MHA, MQA, GQA)")
            logger.error("- Implement detailed component-level breakdowns")
        
        if not dashboard_valid:
            logger.error("\nDashboard Visualization Issues:")
            logger.error("- Add support for visualizing hardware-specific metrics")
            logger.error("- Implement detailed breakdown charts for different hardware")
            logger.error("- Add visualization for hardware efficiency factors")
        
        if not benchmark_valid:
            logger.error("\nBenchmark Implementation Issues:")
            logger.error("- Ensure factory pattern correctly creates hardware-specific metrics")
            logger.error("- Implement comprehensive hardware detection and configuration")
            logger.error("- Add support for hardware efficiency reporting")
            
        if not power_valid:
            logger.error("\nPower Metrics Issues:")
            logger.error("- Implement platform-specific power monitoring")
            logger.error("- Add efficiency metrics calculation (operations per watt)")
            logger.error("- Ensure integration with the benchmark orchestration")
            logger.error("- Create tests for power metrics implementation")
            
        if not bandwidth_valid:
            logger.error("\nBandwidth Metrics Issues:")
            logger.error("- Implement platform-specific bandwidth measurement")
            logger.error("- Add memory transfer estimation for accurate bandwidth calculation")
            logger.error("- Implement roofline model analysis")
            logger.error("- Create tests for bandwidth metrics implementation")
        
        return False

def main():
    """Entry point for validation script."""
    parser = argparse.ArgumentParser(description="Validate refactored benchmark suite")
    parser.add_argument("--flops", action="store_true", help="Validate FLOPs metric only")
    parser.add_argument("--dashboard", action="store_true", help="Validate dashboard only")
    parser.add_argument("--benchmark", action="store_true", help="Validate benchmark only")
    parser.add_argument("--power", action="store_true", help="Validate power metrics only")
    parser.add_argument("--bandwidth", action="store_true", help="Validate bandwidth metrics only")
    args = parser.parse_args()
    
    # If no specific validation requested, run all
    if not (args.flops or args.dashboard or args.benchmark or args.power or args.bandwidth):
        success = validate_all()
    else:
        success = True
        
        if args.flops:
            flops_valid = validate_flops_metric()
            success = success and flops_valid
            
        if args.dashboard:
            dashboard_valid = validate_dashboard_visualization()
            success = success and dashboard_valid
            
        if args.benchmark:
            benchmark_valid = validate_benchmark_metrics()
            success = success and benchmark_valid
            
        if args.power:
            power_valid = validate_power_metric()
            success = success and power_valid
            
        if args.bandwidth:
            bandwidth_valid = validate_bandwidth_metric()
            success = success and bandwidth_valid
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())