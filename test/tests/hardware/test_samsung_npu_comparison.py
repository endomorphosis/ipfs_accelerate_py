#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Samsung NPU Comparison Tool

This module provides comprehensive benchmarking and comparison tools for
Samsung Neural Processing Unit (NPU) hardware against other hardware accelerators.

It supports comparisons between:
- Samsung Exynos NPU
- Qualcomm QNN
- CPU (as baseline)
- Optional: GPU if available

Features:
- Hardware detection and capability comparison
- Model compatibility assessment
- Performance benchmarking
- Power efficiency analysis
- Thermal impact comparison
- Optimization recommendations

Date: March 2025
"""

import os
import sys
import json
import argparse
import logging
import datetime
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import Samsung support
try:
    from samsung_support import (
        SamsungDetector,
        SamsungChipset,
        SamsungBenchmarkRunner,
        SamsungModelConverter
    )
    SAMSUNG_SUPPORT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    logger.warning("Samsung NPU support not available")
    SAMSUNG_SUPPORT_AVAILABLE = False

# Import Qualcomm support
try:
    from hardware_detection.qnn_support import (
        QNNWrapper,
        QNNCapabilityDetector
    )
    QUALCOMM_SUPPORT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    logger.warning("Qualcomm QNN support not available")
    QUALCOMM_SUPPORT_AVAILABLE = False

# Import centralized hardware detection
try:
    from centralized_hardware_detection.hardware_detection import (
        HardwareManager
    )
    CENTRALIZED_HARDWARE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    logger.warning("Centralized hardware detection not available")
    CENTRALIZED_HARDWARE_AVAILABLE = False


class HardwareComparisonTool:
    """Comprehensive tool for comparing mobile NPU hardware."""
    
    def __init__(self, 
                 use_centralized_hardware: bool = True,
                 samsung_simulation: bool = False,
                 qualcomm_simulation: bool = False,
                 db_path: Optional[str] = None):
        """
        Initialize the hardware comparison tool.
        
        Args:
            use_centralized_hardware: Whether to use centralized hardware detection
            samsung_simulation: Force Samsung simulation mode
            qualcomm_simulation: Force Qualcomm simulation mode
            db_path: Optional path to benchmark database
        """
        self.use_centralized_hardware = use_centralized_hardware
        self.samsung_simulation = samsung_simulation
        self.qualcomm_simulation = qualcomm_simulation
        self.db_path = db_path
        
        # Initialize hardware detection
        self._initialize_hardware_detection()
        
        # Store available hardware
        self.available_hardware = self._detect_available_hardware()
        
        # Initialize benchmark runners
        self._initialize_benchmark_runners()
    
    def _initialize_hardware_detection(self):
        """Initialize hardware detection systems."""
        # Set up environment for simulation if requested
        if self.samsung_simulation:
            os.environ["TEST_SAMSUNG_CHIPSET"] = "exynos_2400"
        
        if self.qualcomm_simulation:
            os.environ["QNN_SIMULATION_MODE"] = "1"
        
        # Initialize centralized hardware detection if requested
        if self.use_centralized_hardware and CENTRALIZED_HARDWARE_AVAILABLE:
            self.hardware_manager = HardwareManager()
            logger.info("Using centralized hardware detection")
        else:
            self.hardware_manager = None
            logger.info("Using direct hardware detection")
        
        # Initialize direct hardware detectors
        if SAMSUNG_SUPPORT_AVAILABLE:
            self.samsung_detector = SamsungDetector()
        else:
            self.samsung_detector = None
        
        if QUALCOMM_SUPPORT_AVAILABLE:
            self.qualcomm_detector = QNNCapabilityDetector()
        else:
            self.qualcomm_detector = None
    
    def _detect_available_hardware(self) -> Dict[str, bool]:
        """
        Detect available hardware.
        
        Returns:
            Dictionary of available hardware
        """
        available = {
            "samsung": False,
            "qualcomm": False,
            "cpu": True,  # CPU is always available
            "gpu": False  # GPU detection would be added here
        }
        
        # Try centralized hardware detection first
        if self.hardware_manager is not None:
            capabilities = self.hardware_manager.get_capabilities()
            available["samsung"] = capabilities.get("samsung_npu", False)
            available["qualcomm"] = capabilities.get("qnn", False)
            available["gpu"] = capabilities.get("gpu", False)
            
            # Get simulation status
            if available["samsung"]:
                available["samsung_simulation"] = capabilities.get("samsung_npu_simulation", False)
            
            if available["qualcomm"]:
                available["qualcomm_simulation"] = capabilities.get("qnn_simulation", False)
            
            logger.info(f"Hardware detected via centralized system: {available}")
            return available
        
        # Fall back to direct detection
        if self.samsung_detector is not None:
            samsung_chipset = self.samsung_detector.detect_samsung_hardware()
            available["samsung"] = samsung_chipset is not None
            available["samsung_simulation"] = "TEST_SAMSUNG_CHIPSET" in os.environ
            
            if available["samsung"]:
                self.samsung_chipset = samsung_chipset
                logger.info(f"Samsung NPU detected: {samsung_chipset.name}")
            
        if self.qualcomm_detector is not None:
            available["qualcomm"] = self.qualcomm_detector.is_qnn_available()
            available["qualcomm_simulation"] = "QNN_SIMULATION_MODE" in os.environ
            
            if available["qualcomm"]:
                qnn_info = self.qualcomm_detector.get_qnn_device_info()
                logger.info(f"Qualcomm QNN detected: {qnn_info}")
        
        logger.info(f"Hardware detected via direct detection: {available}")
        return available
    
    def _initialize_benchmark_runners(self):
        """Initialize benchmark runners for available hardware."""
        self.benchmark_runners = {}
        
        # Initialize Samsung benchmark runner if available
        if self.available_hardware.get("samsung", False) and SAMSUNG_SUPPORT_AVAILABLE:
            try:
                self.benchmark_runners["samsung"] = SamsungBenchmarkRunner(
                    db_path=self.db_path
                )
            except Exception as e:
                logger.warning(f"Could not initialize Samsung benchmark runner: {e}")
        
        # Qualcomm benchmark runner would be initialized here
        # if self.available_hardware["qualcomm"] and QUALCOMM_SUPPORT_AVAILABLE:
        #     self.benchmark_runners["qualcomm"] = QualcommBenchmarkRunner()
        
        logger.info(f"Initialized benchmark runners for: {list(self.benchmark_runners.keys())}")
    
    def get_hardware_capability_comparison(self) -> Dict[str, Any]:
        """
        Get detailed comparison of hardware capabilities.
        
        Returns:
            Dictionary with hardware capability comparison
        """
        comparison = {
            "timestamp": datetime.datetime.now().isoformat(),
            "available_hardware": self.available_hardware,
            "hardware_details": {},
            "capability_comparison": {},
            "model_compatibility": {},
            "power_efficiency": {},
            "thermal_impact": {},
            "optimization_recommendations": {}
        }
        
        # Get hardware details
        if self.available_hardware["samsung"] and self.samsung_detector is not None:
            if hasattr(self, 'samsung_chipset'):
                samsung_chipset = self.samsung_chipset
            else:
                samsung_chipset = self.samsung_detector.detect_samsung_hardware()
            
            if samsung_chipset is not None:
                # Basic info
                comparison["hardware_details"]["samsung"] = samsung_chipset.to_dict()
                
                # Detailed capability analysis
                capability_analysis = self.samsung_detector.get_capability_analysis(samsung_chipset)
                comparison["capability_comparison"]["samsung"] = capability_analysis
                
                # Extract model compatibility from capability analysis
                comparison["model_compatibility"]["samsung"] = capability_analysis["model_capabilities"]
                
                # Extract power efficiency from capability analysis
                comparison["power_efficiency"]["samsung"] = capability_analysis["power_efficiency"]
                
                # Extract optimization recommendations from capability analysis
                comparison["optimization_recommendations"]["samsung"] = capability_analysis["recommended_optimizations"]
        
        # Get Qualcomm details
        if self.available_hardware["qualcomm"] and self.qualcomm_detector is not None:
            # Basic info
            qnn_info = self.qualcomm_detector.get_qnn_device_info()
            comparison["hardware_details"]["qualcomm"] = qnn_info
            
            # Other Qualcomm-specific details would be added here
            # comparison["capability_comparison"]["qualcomm"] = ...
            # comparison["model_compatibility"]["qualcomm"] = ...
            # comparison["power_efficiency"]["qualcomm"] = ...
            # comparison["optimization_recommendations"]["qualcomm"] = ...
        
        # Generate comparative recommendations
        if len(comparison["hardware_details"]) > 1:
            comparison["comparative_recommendations"] = self._generate_comparative_recommendations(comparison)
        
        return comparison
    
    def _generate_comparative_recommendations(self, comparison_data: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on comparative analysis.
        
        Args:
            comparison_data: Hardware comparison data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check what hardware is available for comparison
        samsung_available = "samsung" in comparison_data["hardware_details"]
        qualcomm_available = "qualcomm" in comparison_data["hardware_details"]
        
        if samsung_available and qualcomm_available:
            # Compare Samsung and Qualcomm capabilities
            samsung_tops = comparison_data["hardware_details"]["samsung"]["npu_tops"]
            
            # This is a simplified comparison; in a real implementation, we would
            # do a more detailed analysis of capabilities
            
            # Performance recommendations
            if samsung_tops > 25.0:
                recommendations.append(
                    "Samsung Exynos NPU offers higher TOPS performance, prefer it for "
                    "compute-intensive models like dense transformers and large vision models."
                )
            else:
                recommendations.append(
                    "Qualcomm QNN may offer better overall performance for most models. "
                    "Consider using it as the default backend."
                )
            
            # Precision recommendations
            samsung_precisions = comparison_data["hardware_details"]["samsung"]["supported_precisions"]
            if "INT4" in samsung_precisions:
                recommendations.append(
                    "Samsung Exynos NPU supports INT4 precision, which can significantly "
                    "reduce memory usage for large models. Consider using Samsung for "
                    "memory-constrained deployments."
                )
            
            # Power efficiency recommendations
            if "power_efficiency" in comparison_data and "samsung" in comparison_data["power_efficiency"]:
                samsung_efficiency = comparison_data["power_efficiency"]["samsung"]
                if samsung_efficiency["efficiency_rating"] == "High":
                    recommendations.append(
                        "Samsung Exynos NPU has high power efficiency. Prefer it for "
                        "battery-sensitive applications or sustained workloads."
                    )
                elif samsung_efficiency["efficiency_rating"] == "Low":
                    recommendations.append(
                        "Consider using Qualcomm QNN for better power efficiency in "
                        "battery-sensitive applications."
                    )
            
            # Model type recommendations
            if "model_compatibility" in comparison_data and "samsung" in comparison_data["model_compatibility"]:
                samsung_model_compat = comparison_data["model_compatibility"]["samsung"]
                
                # Text generation models
                if samsung_model_compat["text_generation"]["suitable"]:
                    if samsung_model_compat["text_generation"]["performance"] == "High":
                        recommendations.append(
                            "Samsung Exynos NPU shows high performance for text generation models. "
                            "Consider using it for LLM inference."
                        )
                
                # Vision models
                if samsung_model_compat["vision_models"]["suitable"]:
                    recommendations.append(
                        "Both Samsung and Qualcomm offer good performance for vision models. "
                        "Consider testing both to determine the best for your specific model."
                    )
        
        # Samsung-specific recommendations
        elif samsung_available:
            # Add Samsung-only recommendations
            if "optimization_recommendations" in comparison_data and "samsung" in comparison_data["optimization_recommendations"]:
                samsung_opts = comparison_data["optimization_recommendations"]["samsung"]
                recommendations.append(
                    f"For optimal Samsung Exynos NPU performance, consider these optimizations: {', '.join(samsung_opts[:3])}"
                )
        
        # Qualcomm-specific recommendations
        elif qualcomm_available:
            # Add Qualcomm-only recommendations
            recommendations.append(
                "Qualcomm QNN is the only available NPU hardware. Consider using INT8 quantization "
                "and model partitioning to optimize performance."
            )
        
        # Generic recommendations
        recommendations.append(
            "For best cross-platform compatibility, maintain models in ONNX format "
            "and apply hardware-specific optimizations at deployment time."
        )
        
        return recommendations
    
    def run_model_compatibility_assessment(self, model_path: str) -> Dict[str, Any]:
        """
        Assess model compatibility across available hardware.
        
        Args:
            model_path: Path to the model file (ONNX, TensorFlow, or PyTorch)
            
        Returns:
            Dictionary with compatibility assessment results
        """
        results = {
            "model_path": model_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "compatibility": {}
        }
        
        # Check Samsung compatibility
        if self.available_hardware.get("samsung", False) and "samsung" in self.benchmark_runners:
            try:
                # Create model converter
                converter = SamsungModelConverter()
                
                # Analyze compatibility
                if hasattr(self, 'samsung_chipset'):
                    target_chipset = self.samsung_chipset.name.lower().replace(" ", "_")
                else:
                    target_chipset = "exynos_2400"  # Default to high-end chipset
                
                samsung_compat = converter.analyze_model_compatibility(
                    model_path=model_path,
                    target_chipset=target_chipset
                )
                
                results["compatibility"]["samsung"] = samsung_compat
            except Exception as e:
                logger.warning(f"Could not analyze Samsung compatibility: {e}")
                results["compatibility"]["samsung"] = {"error": str(e)}
        
        # Qualcomm compatibility would be checked here
        
        # Generate comparative recommendations
        if len(results["compatibility"]) > 0:
            results["recommendations"] = self._generate_model_specific_recommendations(results)
        
        return results
    
    def _generate_model_specific_recommendations(self, compatibility_results: Dict[str, Any]) -> List[str]:
        """
        Generate model-specific recommendations based on compatibility results.
        
        Args:
            compatibility_results: Model compatibility results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Extract model info
        model_path = compatibility_results["model_path"]
        model_format = model_path.split(".")[-1]
        
        # Check what hardware compatibility results we have
        samsung_compat = "samsung" in compatibility_results["compatibility"]
        qualcomm_compat = "qualcomm" in compatibility_results["compatibility"]
        
        # General format recommendations
        if model_format.lower() != "onnx":
            recommendations.append(
                "Convert the model to ONNX format for better cross-platform "
                "compatibility across mobile NPUs."
            )
        
        # Samsung-specific recommendations
        if samsung_compat:
            samsung_results = compatibility_results["compatibility"]["samsung"]
            
            # Check if model is supported
            if samsung_results["compatibility"]["supported"]:
                # Precision recommendations
                recommendations.append(
                    f"For Samsung Exynos NPU, use {samsung_results['compatibility']['recommended_precision']} "
                    "precision for optimal performance."
                )
                
                # Check for potential issues
                if "potential_issues" in samsung_results["compatibility"]:
                    issues = samsung_results["compatibility"]["potential_issues"]
                    if issues and issues[0] != "No significant issues detected":
                        recommendations.append(
                            f"Potential issues with Samsung Exynos NPU: {issues[0]}"
                        )
                
                # Optimization opportunities
                if "optimization_opportunities" in samsung_results["compatibility"]:
                    opts = samsung_results["compatibility"]["optimization_opportunities"]
                    if opts:
                        recommendations.append(
                            f"Samsung optimization opportunities: {', '.join(opts[:3])}"
                        )
        
        # Qualcomm-specific recommendations would be added here
        
        # Overall recommendations
        if samsung_compat and qualcomm_compat:
            # Compare expected performance and make recommendations
            # This would be expanded in a real implementation
            recommendations.append(
                "Test the model on both Samsung and Qualcomm hardware to determine "
                "the best performance for your specific model and use case."
            )
        
        return recommendations
    
    def run_performance_benchmark(self, 
                                 model_path: str,
                                 hardware_types: Optional[List[str]] = None,
                                 batch_sizes: Optional[List[int]] = None,
                                 precision: str = "INT8",
                                 duration_seconds: int = 10,
                                 iterations: int = 5,
                                 monitor_thermals: bool = True) -> Dict[str, Any]:
        """
        Run performance benchmark across available hardware.
        
        Args:
            model_path: Path to the model file
            hardware_types: List of hardware types to benchmark (default: all available)
            batch_sizes: List of batch sizes to test (default: [1, 4, 8])
            precision: Precision to use for benchmarking
            duration_seconds: Duration of each benchmark in seconds
            iterations: Number of iterations per benchmark
            monitor_thermals: Whether to monitor thermal impact
            
        Returns:
            Dictionary with benchmark results
        """
        if hardware_types is None:
            hardware_types = [hw for hw, available in self.available_hardware.items() if available]
        
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]
        
        results = {
            "model_path": model_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "precision": precision,
            "duration_seconds": duration_seconds,
            "iterations": iterations,
            "hardware_results": {},
            "comparative_results": {}
        }
        
        # Run benchmarks for each hardware type
        for hw_type in hardware_types:
            if hw_type in self.benchmark_runners:
                logger.info(f"Running benchmark on {hw_type} with batch sizes {batch_sizes}")
                
                # Run benchmark
                hw_results = self.benchmark_runners[hw_type].run_benchmark(
                    model_path=model_path,
                    batch_sizes=batch_sizes,
                    precision=precision,
                    duration_seconds=duration_seconds,
                    monitor_thermals=monitor_thermals
                )
                
                results["hardware_results"][hw_type] = hw_results
        
        # Generate comparative results
        if len(results["hardware_results"]) > 1:
            results["comparative_results"] = self._generate_comparative_benchmark_results(results)
        
        return results
    
    def _generate_comparative_benchmark_results(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparative analysis of benchmark results.
        
        Args:
            benchmark_results: Benchmark results
            
        Returns:
            Dictionary with comparative analysis
        """
        comparative = {
            "performance_comparison": {},
            "power_efficiency_comparison": {},
            "optimal_batch_size": {},
            "recommendations": []
        }
        
        # Extract results for each hardware type
        hardware_results = benchmark_results["hardware_results"]
        hardware_types = list(hardware_results.keys())
        batch_sizes = list(hardware_results[hardware_types[0]]["batch_results"].keys())
        
        # Compare performance for each batch size
        for batch_size in batch_sizes:
            comparative["performance_comparison"][batch_size] = {}
            comparative["power_efficiency_comparison"][batch_size] = {}
            
            # Get reference hardware (use first one as reference)
            ref_hw = hardware_types[0]
            ref_throughput = hardware_results[ref_hw]["batch_results"][batch_size]["throughput_items_per_second"]
            ref_latency = hardware_results[ref_hw]["batch_results"][batch_size]["latency_ms"]["avg"]
            ref_power = hardware_results[ref_hw]["batch_results"][batch_size]["power_metrics"]["power_consumption_mw"]
            
            # Calculate relative performance for each hardware
            for hw_type in hardware_types:
                hw_throughput = hardware_results[hw_type]["batch_results"][batch_size]["throughput_items_per_second"]
                hw_latency = hardware_results[hw_type]["batch_results"][batch_size]["latency_ms"]["avg"]
                hw_power = hardware_results[hw_type]["batch_results"][batch_size]["power_metrics"]["power_consumption_mw"]
                
                # Calculate relative performance
                comparative["performance_comparison"][batch_size][hw_type] = {
                    "throughput_items_per_second": hw_throughput,
                    "latency_ms": hw_latency,
                    "throughput_vs_reference": hw_throughput / ref_throughput if ref_throughput > 0 else 0,
                    "latency_vs_reference": ref_latency / hw_latency if hw_latency > 0 else 0
                }
                
                # Calculate power efficiency
                power_efficiency = hw_throughput / hw_power if hw_power > 0 else 0
                ref_power_efficiency = ref_throughput / ref_power if ref_power > 0 else 0
                
                comparative["power_efficiency_comparison"][batch_size][hw_type] = {
                    "power_consumption_mw": hw_power,
                    "throughput_per_watt": power_efficiency * 1000,  # Items per second per watt
                    "efficiency_vs_reference": power_efficiency / ref_power_efficiency if ref_power_efficiency > 0 else 0
                }
        
        # Determine optimal batch size for each hardware
        for hw_type in hardware_types:
            max_efficiency = 0
            optimal_batch = batch_sizes[0]
            
            for batch_size in batch_sizes:
                hw_results = hardware_results[hw_type]["batch_results"][batch_size]
                throughput = hw_results["throughput_items_per_second"]
                power = hw_results["power_metrics"]["power_consumption_mw"]
                
                efficiency = throughput / power if power > 0 else 0
                
                if efficiency > max_efficiency:
                    max_efficiency = efficiency
                    optimal_batch = batch_size
            
            comparative["optimal_batch_size"][hw_type] = {
                "batch_size": optimal_batch,
                "throughput_items_per_second": hardware_results[hw_type]["batch_results"][optimal_batch]["throughput_items_per_second"],
                "latency_ms": hardware_results[hw_type]["batch_results"][optimal_batch]["latency_ms"]["avg"],
                "power_consumption_mw": hardware_results[hw_type]["batch_results"][optimal_batch]["power_metrics"]["power_consumption_mw"],
                "throughput_per_watt": max_efficiency * 1000  # Items per second per watt
            }
        
        # Generate recommendations
        if "samsung" in hardware_types and "qualcomm" in hardware_types:
            # Compare Samsung and Qualcomm
            samsung_opt = comparative["optimal_batch_size"]["samsung"]
            qualcomm_opt = comparative["optimal_batch_size"]["qualcomm"]
            
            if samsung_opt["throughput_items_per_second"] > qualcomm_opt["throughput_items_per_second"]:
                comparative["recommendations"].append(
                    f"Samsung Exynos NPU offers higher throughput ({samsung_opt['throughput_items_per_second']:.2f} vs "
                    f"{qualcomm_opt['throughput_items_per_second']:.2f} items/sec). Use Samsung for throughput-sensitive applications."
                )
            else:
                comparative["recommendations"].append(
                    f"Qualcomm QNN offers higher throughput ({qualcomm_opt['throughput_items_per_second']:.2f} vs "
                    f"{samsung_opt['throughput_items_per_second']:.2f} items/sec). Use Qualcomm for throughput-sensitive applications."
                )
            
            if samsung_opt["latency_ms"] < qualcomm_opt["latency_ms"]:
                comparative["recommendations"].append(
                    f"Samsung Exynos NPU offers lower latency ({samsung_opt['latency_ms']:.2f} ms vs "
                    f"{qualcomm_opt['latency_ms']:.2f} ms). Use Samsung for latency-sensitive applications."
                )
            else:
                comparative["recommendations"].append(
                    f"Qualcomm QNN offers lower latency ({qualcomm_opt['latency_ms']:.2f} ms vs "
                    f"{samsung_opt['latency_ms']:.2f} ms). Use Qualcomm for latency-sensitive applications."
                )
            
            if samsung_opt["throughput_per_watt"] > qualcomm_opt["throughput_per_watt"]:
                comparative["recommendations"].append(
                    f"Samsung Exynos NPU offers better power efficiency ({samsung_opt['throughput_per_watt']:.2f} vs "
                    f"{qualcomm_opt['throughput_per_watt']:.2f} items/sec/watt). Use Samsung for battery-sensitive applications."
                )
            else:
                comparative["recommendations"].append(
                    f"Qualcomm QNN offers better power efficiency ({qualcomm_opt['throughput_per_watt']:.2f} vs "
                    f"{samsung_opt['throughput_per_watt']:.2f} items/sec/watt). Use Qualcomm for battery-sensitive applications."
                )
        
        # Add batch size recommendations
        for hw_type in hardware_types:
            opt_batch = comparative["optimal_batch_size"][hw_type]["batch_size"]
            comparative["recommendations"].append(
                f"For {hw_type}, the optimal batch size is {opt_batch} for best throughput-power efficiency."
            )
        
        return comparative

    def run_thermal_impact_analysis(self,
                                  model_path: str,
                                  hardware_types: Optional[List[str]] = None,
                                  batch_size: int = 1,
                                  precision: str = "INT8",
                                  duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Run thermal impact analysis across available hardware.
        
        Args:
            model_path: Path to the model file
            hardware_types: List of hardware types to analyze (default: all available)
            batch_size: Batch size to use for the analysis
            precision: Precision to use for the analysis
            duration_minutes: Duration of the analysis in minutes
            
        Returns:
            Dictionary with thermal impact analysis results
        """
        if hardware_types is None:
            hardware_types = [hw for hw, available in self.available_hardware.items() if available]
        
        results = {
            "model_path": model_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "batch_size": batch_size,
            "precision": precision,
            "duration_minutes": duration_minutes,
            "hardware_results": {},
            "comparative_results": {}
        }
        
        # For now, this is a simplified implementation that delegates to hardware-specific runners
        for hw_type in hardware_types:
            if hw_type in self.benchmark_runners and hasattr(self.benchmark_runners[hw_type], 'run_thermal_analysis'):
                logger.info(f"Running thermal analysis on {hw_type}")
                
                # Run thermal analysis
                hw_results = self.benchmark_runners[hw_type].run_thermal_analysis(
                    model_path=model_path,
                    batch_size=batch_size,
                    precision=precision,
                    duration_minutes=duration_minutes
                )
                
                results["hardware_results"][hw_type] = hw_results
        
        # Generate comparative results
        if len(results["hardware_results"]) > 1:
            # This would be implemented in a real thermal analysis system
            results["comparative_results"] = {
                "thermal_comparison": {},
                "recommendations": []
            }
        
        return results

    def generate_hardware_report(self, output_path: Optional[str] = None, format: str = "json") -> str:
        """
        Generate a comprehensive hardware comparison report.
        
        Args:
            output_path: Path to save the report (default: stdout)
            format: Report format ("json" or "text")
            
        Returns:
            Path to the generated report or report content if output_path is None
        """
        # Get hardware capability comparison
        comparison = self.get_hardware_capability_comparison()
        
        # Format the report
        if format.lower() == "json":
            report_content = json.dumps(comparison, indent=2)
        else:
            # Generate text report
            report_content = self._generate_text_report(comparison)
        
        # Save or return the report
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Hardware report saved to {output_path}")
            return output_path
        else:
            return report_content
    
    def _generate_text_report(self, comparison: Dict[str, Any]) -> str:
        """
        Generate a text report from comparison data.
        
        Args:
            comparison: Hardware comparison data
            
        Returns:
            Text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MOBILE NPU HARDWARE COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {comparison['timestamp']}")
        lines.append("")
        
        # Available hardware
        lines.append("-" * 80)
        lines.append("AVAILABLE HARDWARE")
        lines.append("-" * 80)
        for hw_type, available in comparison["available_hardware"].items():
            if hw_type.endswith("_simulation"):
                continue
            status = "AVAILABLE" if available else "NOT AVAILABLE"
            if available and hw_type + "_simulation" in comparison["available_hardware"]:
                if comparison["available_hardware"][hw_type + "_simulation"]:
                    status += " (SIMULATION MODE)"
            lines.append(f"{hw_type.upper(): <10}: {status}")
        lines.append("")
        
        # Hardware details
        if comparison["hardware_details"]:
            lines.append("-" * 80)
            lines.append("HARDWARE DETAILS")
            lines.append("-" * 80)
            
            for hw_type, details in comparison["hardware_details"].items():
                lines.append(f"{hw_type.upper()} DETAILS:")
                
                if hw_type == "samsung":
                    lines.append(f"  Model:            {details['name']}")
                    lines.append(f"  NPU Cores:        {details['npu_cores']}")
                    lines.append(f"  NPU Performance:  {details['npu_tops']} TOPS")
                    lines.append(f"  Max Precision:    {details['max_precision']}")
                    lines.append(f"  Supported Prec.:  {', '.join(details['supported_precisions'])}")
                    lines.append(f"  Max Power Draw:   {details['max_power_draw']} W")
                    lines.append(f"  Typical Power:    {details['typical_power']} W")
                elif hw_type == "qualcomm":
                    # Format Qualcomm details
                    for key, value in details.items():
                        lines.append(f"  {key}: {value}")
                
                lines.append("")
        
        # Model compatibility
        if comparison["model_compatibility"]:
            lines.append("-" * 80)
            lines.append("MODEL COMPATIBILITY")
            lines.append("-" * 80)
            
            for hw_type, compat in comparison["model_compatibility"].items():
                lines.append(f"{hw_type.upper()} MODEL COMPATIBILITY:")
                
                for model_type, details in compat.items():
                    suitability = "Suitable" if details["suitable"] else "Not Suitable"
                    lines.append(f"  {model_type: <20}: {suitability}")
                    lines.append(f"    Max Size:         {details['max_size']}")
                    lines.append(f"    Performance:      {details['performance']}")
                    if "notes" in details:
                        lines.append(f"    Notes:            {details['notes']}")
                
                lines.append("")
        
        # Power efficiency
        if comparison["power_efficiency"]:
            lines.append("-" * 80)
            lines.append("POWER EFFICIENCY")
            lines.append("-" * 80)
            
            for hw_type, efficiency in comparison["power_efficiency"].items():
                lines.append(f"{hw_type.upper()} POWER EFFICIENCY:")
                
                if "tops_per_watt" in efficiency:
                    lines.append(f"  TOPS per Watt:     {efficiency['tops_per_watt']:.2f}")
                if "efficiency_rating" in efficiency:
                    lines.append(f"  Efficiency Rating: {efficiency['efficiency_rating']}")
                if "battery_impact" in efficiency:
                    lines.append(f"  Battery Impact:    {efficiency['battery_impact']}")
                
                lines.append("")
        
        # Optimization recommendations
        if comparison["optimization_recommendations"]:
            lines.append("-" * 80)
            lines.append("OPTIMIZATION RECOMMENDATIONS")
            lines.append("-" * 80)
            
            for hw_type, recommendations in comparison["optimization_recommendations"].items():
                lines.append(f"{hw_type.upper()} OPTIMIZATIONS:")
                
                for i, recommendation in enumerate(recommendations, 1):
                    lines.append(f"  {i}. {recommendation}")
                
                lines.append("")
        
        # Comparative recommendations
        if "comparative_recommendations" in comparison:
            lines.append("-" * 80)
            lines.append("COMPARATIVE RECOMMENDATIONS")
            lines.append("-" * 80)
            
            for i, recommendation in enumerate(comparison["comparative_recommendations"], 1):
                lines.append(f"{i}. {recommendation}")
            
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main function to run the hardware comparison tool from command line."""
    # Check for required packages
    try:
        import importlib.util
        missing_packages = []
        for package in ["duckdb", "pandas", "fastapi", "uvicorn", "pydantic"]:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Warning: Optional packages {', '.join(missing_packages)} not installed.")
            print("Some functionality may be limited, but basic report generation will work.")
            print()
    except ImportError:
        pass
    
    parser = argparse.ArgumentParser(description="Samsung NPU Comparison Tool")
    
    # General options
    parser.add_argument("--use-centralized", action="store_true", help="Use centralized hardware detection")
    parser.add_argument("--samsung-simulation", action="store_true", help="Force Samsung simulation mode")
    parser.add_argument("--qualcomm-simulation", action="store_true", help="Force Qualcomm simulation mode")
    parser.add_argument("--db-path", type=str, help="Path to benchmark database")
    
    # Command options
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Hardware report command
    report_parser = subparsers.add_parser("report", help="Generate hardware comparison report")
    report_parser.add_argument("--output", type=str, help="Output path for report")
    report_parser.add_argument("--format", type=str, choices=["json", "text"], default="text", help="Report format")
    
    # Model compatibility command
    compat_parser = subparsers.add_parser("compatibility", help="Assess model compatibility")
    compat_parser.add_argument("--model", type=str, required=True, help="Path to model file")
    compat_parser.add_argument("--output", type=str, help="Output path for compatibility report")
    compat_parser.add_argument("--format", type=str, choices=["json", "text"], default="json", help="Report format")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    benchmark_parser.add_argument("--model", type=str, required=True, help="Path to model file")
    benchmark_parser.add_argument("--hardware", type=str, nargs="+", help="Hardware types to benchmark")
    benchmark_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8], help="Batch sizes to test")
    benchmark_parser.add_argument("--precision", type=str, default="INT8", help="Precision to use")
    benchmark_parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    benchmark_parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    benchmark_parser.add_argument("--no-thermals", action="store_true", help="Disable thermal monitoring")
    benchmark_parser.add_argument("--output", type=str, help="Output path for benchmark report")
    benchmark_parser.add_argument("--format", type=str, choices=["json", "text"], default="json", help="Report format")
    
    # Thermal analysis command
    thermal_parser = subparsers.add_parser("thermal", help="Run thermal impact analysis")
    thermal_parser.add_argument("--model", type=str, required=True, help="Path to model file")
    thermal_parser.add_argument("--hardware", type=str, nargs="+", help="Hardware types to analyze")
    thermal_parser.add_argument("--batch-size", type=int, default=1, help="Batch size to use")
    thermal_parser.add_argument("--precision", type=str, default="INT8", help="Precision to use")
    thermal_parser.add_argument("--duration", type=int, default=5, help="Duration in minutes")
    thermal_parser.add_argument("--output", type=str, help="Output path for thermal analysis report")
    thermal_parser.add_argument("--format", type=str, choices=["json", "text"], default="json", help="Report format")
    
    args = parser.parse_args()
    
    # Create hardware comparison tool
    tool = HardwareComparisonTool(
        use_centralized_hardware=args.use_centralized,
        samsung_simulation=args.samsung_simulation,
        qualcomm_simulation=args.qualcomm_simulation,
        db_path=args.db_path
    )
    
    # Run requested command
    if args.command == "report":
        result = tool.generate_hardware_report(
            output_path=args.output,
            format=args.format
        )
        if not args.output:
            print(result)
    
    elif args.command == "compatibility":
        result = tool.run_model_compatibility_assessment(
            model_path=args.model
        )
        
        if args.format.lower() == "json":
            output = json.dumps(result, indent=2)
        else:
            # Format as text (simplified for now)
            output = str(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Compatibility report saved to {args.output}")
        else:
            print(output)
    
    elif args.command == "benchmark":
        result = tool.run_performance_benchmark(
            model_path=args.model,
            hardware_types=args.hardware,
            batch_sizes=args.batch_sizes,
            precision=args.precision,
            duration_seconds=args.duration,
            iterations=args.iterations,
            monitor_thermals=not args.no_thermals
        )
        
        if args.format.lower() == "json":
            output = json.dumps(result, indent=2)
        else:
            # Format as text (simplified for now)
            output = str(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Benchmark report saved to {args.output}")
        else:
            print(output)
    
    elif args.command == "thermal":
        result = tool.run_thermal_impact_analysis(
            model_path=args.model,
            hardware_types=args.hardware,
            batch_size=args.batch_size,
            precision=args.precision,
            duration_minutes=args.duration
        )
        
        if args.format.lower() == "json":
            output = json.dumps(result, indent=2)
        else:
            # Format as text (simplified for now)
            output = str(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Thermal analysis report saved to {args.output}")
        else:
            print(output)
    
    else:
        # No command specified, print hardware report by default
        result = tool.generate_hardware_report(format="text")
        print(result)


if __name__ == "__main__":
    main()