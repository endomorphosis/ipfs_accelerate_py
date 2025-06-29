#!/usr/bin/env python3
"""
Hardware Optimization Analyzer

This module analyzes benchmark and predictive performance data to generate
optimization recommendations specific to hardware platforms and models.
It leverages historical performance data to identify optimization opportunities
and suggest configuration changes to improve throughput, latency, and memory usage.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hardware_optimization_analyzer")

# Import clients
try:
    from test.api_client.predictive_performance_client import (
        PredictivePerformanceClient,
        HardwarePlatform,
        PrecisionType,
        ModelMode
    )
    PREDICTIVE_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("Predictive Performance client not available")
    PREDICTIVE_CLIENT_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available")
    DUCKDB_AVAILABLE = False

class HardwareOptimizationAnalyzer:
    """
    Analyzer for hardware-specific optimization recommendations.
    
    This class provides functionality to:
    1. Analyze performance data for specific model-hardware combinations
    2. Identify optimization opportunities based on performance patterns
    3. Generate actionable optimization recommendations
    4. Provide guidance on hardware-specific configuration
    5. Estimate performance improvements from applying recommendations
    """
    
    def __init__(
        self,
        benchmark_db_path: str = "benchmark_db.duckdb",
        predictive_api_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            benchmark_db_path: Path to the benchmark DuckDB database
            predictive_api_url: URL of the Predictive Performance API
            api_key: Optional API key for authenticated endpoints
            config: Optional configuration dictionary
        """
        self.benchmark_db_path = benchmark_db_path
        self.predictive_api_url = predictive_api_url
        self.api_key = api_key
        self.config = config or {}
        
        # Connect to benchmark database
        self.benchmark_conn = None
        if DUCKDB_AVAILABLE:
            try:
                self.benchmark_conn = duckdb.connect(benchmark_db_path)
                logger.info(f"Connected to benchmark database at {benchmark_db_path}")
            except Exception as e:
                logger.error(f"Error connecting to benchmark database: {e}")
        
        # Initialize predictive performance client
        self.predictive_client = None
        if PREDICTIVE_CLIENT_AVAILABLE:
            try:
                self.predictive_client = PredictivePerformanceClient(
                    base_url=predictive_api_url,
                    api_key=api_key
                )
                logger.info(f"Initialized Predictive Performance client with URL {predictive_api_url}")
            except Exception as e:
                logger.error(f"Error initializing Predictive Performance client: {e}")
        
        # Hardware-specific optimization strategies
        self.optimization_strategies = self._load_optimization_strategies()
        
        # Performance thresholds for optimization recommendations
        self.thresholds = self.config.get("thresholds", {
            "throughput_improvement": 0.15,  # 15% improvement threshold
            "latency_reduction": 0.20,       # 20% latency reduction threshold
            "memory_reduction": 0.10,        # 10% memory usage reduction threshold
            "min_data_points": 3             # Minimum data points for reliable analysis
        })
    
    def _load_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Load hardware-specific optimization strategies.
        
        Returns:
            Dictionary of optimization strategies by hardware type
        """
        # Default strategies
        strategies = {
            "cpu": {
                "name": "CPU",
                "strategies": [
                    {
                        "name": "Quantization",
                        "description": "Convert model weights to lower precision",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.3,
                            "latency": 0.3,
                            "memory": 0.5
                        },
                        "configuration": {
                            "precision": "int8",
                            "quantization_method": "dynamic"
                        },
                        "implementation": "Use torch.quantization.quantize_dynamic for PyTorch models or TensorFlow Lite for TF models"
                    },
                    {
                        "name": "Thread Optimization",
                        "description": "Optimize number of threads for CPU inference",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.2,
                            "latency": 0.15,
                            "memory": 0.0
                        },
                        "configuration": {
                            "num_threads": "cores_minus_one",
                            "intra_op_parallelism": "cores_divided_by_2"
                        },
                        "implementation": "Set OMP_NUM_THREADS and MKL_NUM_THREADS environment variables"
                    },
                    {
                        "name": "Batch Size Optimization",
                        "description": "Find optimal batch size for throughput",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.4,
                            "latency": -0.2,  # May increase latency
                            "memory": -0.3    # May increase memory usage
                        },
                        "configuration": {
                            "batch_size": "dynamic_based_on_model_size"
                        },
                        "implementation": "Adjust batch size based on model size and available memory"
                    }
                ]
            },
            "cuda": {
                "name": "NVIDIA GPU (CUDA)",
                "strategies": [
                    {
                        "name": "Mixed Precision",
                        "description": "Use FP16 (half precision) for GPU inference",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.5,
                            "latency": 0.3,
                            "memory": 0.4
                        },
                        "configuration": {
                            "precision": "fp16",
                            "mixed_precision": True
                        },
                        "implementation": "Use torch.cuda.amp for PyTorch or TF mixed precision API for TensorFlow"
                    },
                    {
                        "name": "CUDA Graphs",
                        "description": "Use CUDA Graphs for static models",
                        "applicable_to": ["static_models"],
                        "expected_improvements": {
                            "throughput": 0.3,
                            "latency": 0.3,
                            "memory": 0.0
                        },
                        "configuration": {
                            "use_cuda_graphs": True,
                            "static_shapes": True
                        },
                        "implementation": "Use torch.cuda.make_graphed_callables for PyTorch"
                    },
                    {
                        "name": "TensorRT Integration",
                        "description": "Convert model to TensorRT for faster inference",
                        "applicable_to": ["cnn", "transformer_encoder"],
                        "expected_improvements": {
                            "throughput": 0.7,
                            "latency": 0.6,
                            "memory": 0.0
                        },
                        "configuration": {
                            "precision": "fp16",
                            "workspace_size": "8G",
                            "dynamic_shapes": False
                        },
                        "implementation": "Convert using torch2trt or TF-TRT conversion API"
                    },
                    {
                        "name": "Optimal Batch Size",
                        "description": "Find optimal batch size for GPU utilization",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.5,
                            "latency": 0.0,
                            "memory": -0.2
                        },
                        "configuration": {
                            "batch_size": "dynamic_based_on_gpu_memory"
                        },
                        "implementation": "Scale batch size based on available GPU memory"
                    }
                ]
            },
            "rocm": {
                "name": "AMD GPU (ROCm)",
                "strategies": [
                    {
                        "name": "Mixed Precision",
                        "description": "Use FP16 for ROCm inference",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.4,
                            "latency": 0.3,
                            "memory": 0.4
                        },
                        "configuration": {
                            "precision": "fp16",
                            "mixed_precision": True
                        },
                        "implementation": "Use torch.cuda.amp with ROCm backend"
                    },
                    {
                        "name": "MIOpen Tuning",
                        "description": "Tune MIOpen parameters for AMD GPUs",
                        "applicable_to": ["cnn", "transformer_encoder"],
                        "expected_improvements": {
                            "throughput": 0.3,
                            "latency": 0.2,
                            "memory": 0.0
                        },
                        "configuration": {
                            "find_db_path": "/path/to/miopen_find_db",
                            "tuning_iterations": 100
                        },
                        "implementation": "Run MIOpen find to generate tuning database"
                    }
                ]
            },
            "mps": {
                "name": "Apple Silicon (MPS)",
                "strategies": [
                    {
                        "name": "MPS Backend",
                        "description": "Use Metal Performance Shaders backend",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.4,
                            "latency": 0.3,
                            "memory": 0.0
                        },
                        "configuration": {
                            "device": "mps",
                            "debug": False
                        },
                        "implementation": "Use torch.device('mps') for PyTorch or CoreML conversion for TF models"
                    },
                    {
                        "name": "CoreML Integration",
                        "description": "Convert model to CoreML format",
                        "applicable_to": ["vision", "nlp"],
                        "expected_improvements": {
                            "throughput": 0.6,
                            "latency": 0.5,
                            "memory": 0.2
                        },
                        "configuration": {
                            "compute_units": "all",
                            "precision": "fp16"
                        },
                        "implementation": "Convert using coremltools or onnx-coreml"
                    }
                ]
            },
            "openvino": {
                "name": "Intel OpenVINO",
                "strategies": [
                    {
                        "name": "INT8 Quantization",
                        "description": "Quantize model to INT8 with OpenVINO",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.4,
                            "latency": 0.4,
                            "memory": 0.7
                        },
                        "configuration": {
                            "precision": "int8",
                            "calibration_method": "default"
                        },
                        "implementation": "Use OpenVINO POT (Post-Training Optimization Tool)"
                    },
                    {
                        "name": "OpenVINO Streams",
                        "description": "Use multiple inference streams",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.5,
                            "latency": 0.0,
                            "memory": -0.1
                        },
                        "configuration": {
                            "num_streams": "dynamic_based_on_cores",
                            "inference_num_threads": "cores_minus_one"
                        },
                        "implementation": "Configure ExecutableNetwork with optimal num_streams setting"
                    }
                ]
            },
            "webgpu": {
                "name": "WebGPU",
                "strategies": [
                    {
                        "name": "Shader Precompilation",
                        "description": "Precompile WebGPU shaders",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.2,
                            "latency": 0.4,
                            "memory": 0.0
                        },
                        "configuration": {
                            "precompile_shaders": True,
                            "shader_cache": True
                        },
                        "implementation": "Implement shader caching and precompilation in WebGPU code"
                    },
                    {
                        "name": "WebGPU Compute Optimization",
                        "description": "Optimize compute shader workgroups",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.3,
                            "latency": 0.2,
                            "memory": 0.0
                        },
                        "configuration": {
                            "workgroup_size": "device_optimal",
                            "pipeline_caching": True
                        },
                        "implementation": "Tune workgroup size and use pipeline caching"
                    }
                ]
            },
            "webnn": {
                "name": "WebNN",
                "strategies": [
                    {
                        "name": "WebNN Graph Optimization",
                        "description": "Optimize neural network graph",
                        "applicable_to": ["all"],
                        "expected_improvements": {
                            "throughput": 0.3,
                            "latency": 0.3,
                            "memory": 0.1
                        },
                        "configuration": {
                            "optimize_for": "latency",
                            "power_preference": "default"
                        },
                        "implementation": "Use WebNN MLGraphBuilder with appropriate options"
                    }
                ]
            }
        }
        
        # Load custom strategies from config if available
        if "optimization_strategies" in self.config:
            custom_strategies = self.config.get("optimization_strategies", {})
            for hardware, strategy in custom_strategies.items():
                if hardware in strategies:
                    # Merge with existing strategies
                    strategies[hardware]["strategies"].extend(strategy.get("strategies", []))
                else:
                    # Add new hardware type
                    strategies[hardware] = strategy
        
        return strategies
    
    def get_optimization_strategies(self, hardware: str) -> List[Dict[str, Any]]:
        """
        Get optimization strategies for a specific hardware type.
        
        Args:
            hardware: Hardware type
            
        Returns:
            List of optimization strategies
        """
        hardware = hardware.lower()
        if hardware in self.optimization_strategies:
            return self.optimization_strategies[hardware]["strategies"]
        else:
            logger.warning(f"No optimization strategies found for hardware: {hardware}")
            return []
    
    def analyze_performance_data(
        self,
        model_name: str,
        hardware_platform: str,
        batch_size: Optional[int] = None,
        days: int = 90,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze performance data for a specific model and hardware.
        
        Args:
            model_name: Model name
            hardware_platform: Hardware platform
            batch_size: Optional batch size filter
            days: Number of days to look back
            limit: Maximum number of records to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not self.predictive_client:
            return {"error": "Predictive Performance client not available"}
        
        try:
            # Get measurements for the model and hardware
            measurements = self.predictive_client.list_measurements(
                model_name=model_name,
                hardware=hardware_platform,
                batch_size=batch_size,
                days=days,
                limit=limit
            )
            
            if "error" in measurements:
                return measurements
            
            if not measurements.get("results", []):
                return {
                    "model_name": model_name,
                    "hardware_platform": hardware_platform,
                    "message": "No measurements found",
                    "data_points": 0
                }
            
            # Extract measurement data
            data_points = measurements.get("results", [])
            data_count = len(data_points)
            
            # Calculate performance metrics
            throughput_values = [m.get("throughput") for m in data_points if m.get("throughput") is not None]
            latency_values = [m.get("latency") for m in data_points if m.get("latency") is not None]
            memory_values = [m.get("memory_usage") for m in data_points if m.get("memory_usage") is not None]
            
            # Calculate statistics if data is available
            analysis = {
                "model_name": model_name,
                "hardware_platform": hardware_platform,
                "data_points": data_count,
                "time_range": f"Last {days} days",
                "batch_sizes": self._extract_unique_values(data_points, "batch_size"),
                "precision_types": self._extract_unique_values(data_points, "precision"),
                "throughput": self._calculate_statistics(throughput_values),
                "latency": self._calculate_statistics(latency_values),
                "memory_usage": self._calculate_statistics(memory_values),
                "correlation": {}
            }
            
            # Calculate correlations if enough data points
            if data_count >= self.thresholds.get("min_data_points", 3):
                # Correlation between batch size and throughput
                if len(throughput_values) >= 3:
                    batch_sizes = [m.get("batch_size", 1) for m in data_points if m.get("throughput") is not None]
                    analysis["correlation"]["batch_size_throughput"] = self._calculate_correlation(batch_sizes, throughput_values)
                
                # Correlation between batch size and latency
                if len(latency_values) >= 3:
                    batch_sizes = [m.get("batch_size", 1) for m in data_points if m.get("latency") is not None]
                    analysis["correlation"]["batch_size_latency"] = self._calculate_correlation(batch_sizes, latency_values)
                
                # Correlation between precision and throughput
                precision_impact = self._analyze_precision_impact(data_points)
                if precision_impact:
                    analysis["precision_impact"] = precision_impact
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance data: {e}")
            return {"error": str(e)}
    
    def _extract_unique_values(self, data_points: List[Dict[str, Any]], key: str) -> List[Any]:
        """
        Extract unique values for a key from data points.
        
        Args:
            data_points: List of data point dictionaries
            key: Key to extract
            
        Returns:
            List of unique values
        """
        values = [point.get(key) for point in data_points if key in point]
        return list(set(values))
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            Dictionary with statistics
        """
        if not values:
            return {"count": 0}
        
        # Sort values for percentile calculation
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        # Calculate statistics
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "median": sorted_values[count // 2] if count % 2 == 1 else (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2,
            "p25": sorted_values[int(count * 0.25)],
            "p75": sorted_values[int(count * 0.75)],
            "p90": sorted_values[int(count * 0.90)],
            "std_dev": (sum((x - (sum(sorted_values) / count)) ** 2 for x in sorted_values) / count) ** 0.5
        }
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient between two lists of values.
        
        Args:
            x_values: List of x values
            y_values: List of y values
            
        Returns:
            Correlation coefficient
        """
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator_x = sum((x - x_mean) ** 2 for x in x_values) ** 0.5
        denominator_y = sum((y - y_mean) ** 2 for y in y_values) ** 0.5
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
        
        return numerator / (denominator_x * denominator_y)
    
    def _analyze_precision_impact(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the impact of precision on performance metrics.
        
        Args:
            data_points: List of performance data points
            
        Returns:
            Dictionary with precision impact analysis
        """
        # Group data points by precision
        precision_groups = {}
        for point in data_points:
            precision = point.get("precision", "unknown")
            if precision not in precision_groups:
                precision_groups[precision] = []
            precision_groups[precision].append(point)
        
        # Only compare if we have at least two precision types
        if len(precision_groups) < 2:
            return None
        
        # Calculate mean metrics for each precision
        precision_metrics = {}
        for precision, points in precision_groups.items():
            throughput_values = [p.get("throughput") for p in points if p.get("throughput") is not None]
            latency_values = [p.get("latency") for p in points if p.get("latency") is not None]
            memory_values = [p.get("memory_usage") for p in points if p.get("memory_usage") is not None]
            
            precision_metrics[precision] = {
                "count": len(points),
                "throughput_mean": sum(throughput_values) / len(throughput_values) if throughput_values else None,
                "latency_mean": sum(latency_values) / len(latency_values) if latency_values else None,
                "memory_mean": sum(memory_values) / len(memory_values) if memory_values else None
            }
        
        # Compare precision types
        comparisons = []
        precisions = list(precision_metrics.keys())
        for i in range(len(precisions)):
            for j in range(i + 1, len(precisions)):
                p1, p2 = precisions[i], precisions[j]
                m1, m2 = precision_metrics[p1], precision_metrics[p2]
                
                comparison = {
                    "precision_1": p1,
                    "precision_2": p2,
                    "sample_size": {"precision_1": m1["count"], "precision_2": m2["count"]}
                }
                
                # Calculate relative differences
                metrics = ["throughput_mean", "latency_mean", "memory_mean"]
                for metric in metrics:
                    if m1[metric] is not None and m2[metric] is not None and m1[metric] > 0:
                        rel_diff = (m2[metric] - m1[metric]) / m1[metric]
                        comparison[f"{metric}_relative_diff"] = rel_diff
                
                comparisons.append(comparison)
        
        return {
            "precision_metrics": precision_metrics,
            "comparisons": comparisons
        }
    
    def get_optimization_recommendations(
        self,
        model_name: str,
        hardware_platform: str,
        model_family: Optional[str] = None,
        batch_size: Optional[int] = None,
        current_precision: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations for a specific model and hardware.
        
        Args:
            model_name: Model name
            hardware_platform: Hardware platform
            model_family: Optional model family
            batch_size: Optional batch size
            current_precision: Optional current precision being used
            
        Returns:
            Dictionary with optimization recommendations
        """
        # Normalize hardware platform
        hardware_platform = hardware_platform.lower()
        
        # Get analysis
        analysis = self.analyze_performance_data(
            model_name=model_name,
            hardware_platform=hardware_platform,
            batch_size=batch_size
        )
        
        if "error" in analysis:
            return analysis
        
        # Get hardware strategies
        strategies = self.get_optimization_strategies(hardware_platform)
        if not strategies:
            return {
                "model_name": model_name,
                "hardware_platform": hardware_platform,
                "message": f"No optimization strategies available for {hardware_platform}",
                "recommendations": []
            }
        
        # Determine model family if not provided
        if not model_family and self.predictive_client:
            try:
                # Try to get model family from existing predictions
                predictions = self.predictive_client.list_predictions(
                    model_name=model_name,
                    limit=1
                )
                
                if "results" in predictions and predictions["results"]:
                    model_family = predictions["results"][0].get("model_family")
            except:
                pass
        
        # Infer model family from name if still not available
        if not model_family:
            model_family = self._infer_model_family(model_name)
        
        # Determine current precision if not provided
        if not current_precision and "precision_types" in analysis:
            precision_types = analysis.get("precision_types", [])
            if precision_types:
                current_precision = precision_types[0]  # Use the first precision type found
        
        # Default to fp32 if still not available
        current_precision = current_precision or "fp32"
        
        # Generate recommendations
        recommendations = []
        for strategy in strategies:
            # Check if strategy is applicable to this model family
            applicable_to = strategy.get("applicable_to", [])
            if "all" not in applicable_to and model_family not in applicable_to:
                continue
            
            # Skip strategies that are already applied
            if self._is_strategy_already_applied(strategy, current_precision):
                continue
            
            # Calculate expected improvements
            improvements = strategy.get("expected_improvements", {})
            improvement_metrics = self._calculate_improvement_metrics(
                analysis, improvements, current_precision
            )
            
            # Add recommendation
            recommendation = {
                "name": strategy.get("name"),
                "description": strategy.get("description"),
                "hardware_platform": hardware_platform,
                "current_metrics": {
                    "throughput": analysis.get("throughput", {}).get("mean"),
                    "latency": analysis.get("latency", {}).get("mean"),
                    "memory_usage": analysis.get("memory_usage", {}).get("mean")
                },
                "expected_improvements": improvement_metrics,
                "configuration": strategy.get("configuration"),
                "implementation": strategy.get("implementation"),
                "confidence": self._calculate_confidence_score(analysis, strategy)
            }
            
            recommendations.append(recommendation)
        
        # Sort recommendations by expected throughput improvement
        recommendations.sort(
            key=lambda x: x["expected_improvements"]["throughput_improvement"], 
            reverse=True
        )
        
        return {
            "model_name": model_name,
            "model_family": model_family,
            "hardware_platform": hardware_platform,
            "current_precision": current_precision,
            "batch_size": batch_size,
            "data_points": analysis.get("data_points", 0),
            "recommendations": recommendations
        }
    
    def _infer_model_family(self, model_name: str) -> str:
        """
        Infer model family from model name.
        
        Args:
            model_name: Model name
            
        Returns:
            Inferred model family
        """
        model_name_lower = model_name.lower()
        
        if any(term in model_name_lower for term in ["bert", "roberta", "distilbert", "albert"]):
            return "transformer_encoder"
        elif any(term in model_name_lower for term in ["gpt", "llama", "llm", "t5", "falcon", "claude"]):
            return "transformer"
        elif any(term in model_name_lower for term in ["resnet", "efficientnet", "vit", "deit", "convnext"]):
            return "cnn"
        elif any(term in model_name_lower for term in ["whisper", "wav2vec", "hubert"]):
            return "audio"
        elif any(term in model_name_lower for term in ["clip", "blip", "llava"]):
            return "multimodal"
        else:
            return "unknown"
    
    def _is_strategy_already_applied(self, strategy: Dict[str, Any], current_precision: str) -> bool:
        """
        Check if a strategy is already applied based on current configuration.
        
        Args:
            strategy: Strategy dictionary
            current_precision: Current precision being used
            
        Returns:
            True if strategy is already applied, False otherwise
        """
        # Check precision-based strategies
        if "configuration" in strategy and "precision" in strategy["configuration"]:
            strategy_precision = strategy["configuration"]["precision"]
            if strategy_precision == current_precision:
                return True
        
        return False
    
    def _calculate_improvement_metrics(
        self, 
        analysis: Dict[str, Any], 
        improvement_ratios: Dict[str, float],
        current_precision: str
    ) -> Dict[str, Any]:
        """
        Calculate improvement metrics based on analysis and improvement ratios.
        
        Args:
            analysis: Performance analysis
            improvement_ratios: Dictionary of improvement ratios
            current_precision: Current precision being used
            
        Returns:
            Dictionary with improvement metrics
        """
        # Get current metrics
        current_throughput = analysis.get("throughput", {}).get("mean")
        current_latency = analysis.get("latency", {}).get("mean")
        current_memory = analysis.get("memory_usage", {}).get("mean")
        
        # Calculate improvements
        throughput_ratio = improvement_ratios.get("throughput", 0.0)
        latency_ratio = improvement_ratios.get("latency", 0.0)
        memory_ratio = improvement_ratios.get("memory", 0.0)
        
        # Adjust based on precision if applicable
        if current_precision == "fp16" and "precision" in improvement_ratios:
            # Reduce expected gains for already optimized precision
            throughput_ratio *= 0.5
            latency_ratio *= 0.5
            memory_ratio *= 0.5
        
        # Calculate improved values
        improved_throughput = current_throughput * (1 + throughput_ratio) if current_throughput else None
        improved_latency = current_latency * (1 - latency_ratio) if current_latency else None
        improved_memory = current_memory * (1 - memory_ratio) if current_memory else None
        
        return {
            "throughput_improvement": throughput_ratio,
            "latency_reduction": latency_ratio,
            "memory_reduction": memory_ratio,
            "improved_throughput": improved_throughput,
            "improved_latency": improved_latency,
            "improved_memory": improved_memory
        }
    
    def _calculate_confidence_score(self, analysis: Dict[str, Any], strategy: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a recommendation.
        
        Args:
            analysis: Performance analysis
            strategy: Strategy dictionary
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence
        base_confidence = 0.7
        
        # Adjust based on data points
        data_points = analysis.get("data_points", 0)
        if data_points < self.thresholds.get("min_data_points", 3):
            base_confidence *= 0.8
        elif data_points >= 10:
            base_confidence *= 1.1
        
        # Adjust based on precision impact analysis
        if "precision_impact" in analysis and "precision" in strategy.get("configuration", {}):
            base_confidence *= 1.2
        
        # Adjust based on correlation data
        if "correlation" in analysis and "batch_size" in strategy.get("configuration", {}):
            batch_correlation = abs(analysis.get("correlation", {}).get("batch_size_throughput", 0))
            if batch_correlation > 0.7:
                base_confidence *= 1.1
        
        # Cap at 1.0
        return min(base_confidence, 1.0)
    
    def generate_optimization_report(
        self,
        model_names: List[str],
        hardware_platforms: List[str],
        batch_size: Optional[int] = None,
        current_precision: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization report for multiple models and hardware platforms.
        
        Args:
            model_names: List of model names
            hardware_platforms: List of hardware platforms
            batch_size: Optional batch size filter
            current_precision: Optional current precision
            
        Returns:
            Dictionary with optimization report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "models": len(model_names),
            "hardware_platforms": len(hardware_platforms),
            "recommendations_by_model": {},
            "recommendations_by_hardware": {},
            "top_recommendations": []
        }
        
        all_recommendations = []
        
        # Generate recommendations for each model and hardware combination
        for model_name in model_names:
            report["recommendations_by_model"][model_name] = {}
            
            for hardware in hardware_platforms:
                # Get recommendations
                recommendations = self.get_optimization_recommendations(
                    model_name=model_name,
                    hardware_platform=hardware,
                    batch_size=batch_size,
                    current_precision=current_precision
                )
                
                # Store in report
                report["recommendations_by_model"][model_name][hardware] = recommendations
                
                # Add to hardware groups
                if hardware not in report["recommendations_by_hardware"]:
                    report["recommendations_by_hardware"][hardware] = {}
                
                report["recommendations_by_hardware"][hardware][model_name] = recommendations
                
                # Add to all recommendations
                if "recommendations" in recommendations:
                    for rec in recommendations["recommendations"]:
                        all_recommendations.append({
                            "model_name": model_name,
                            "hardware_platform": hardware,
                            "recommendation": rec
                        })
        
        # Sort all recommendations by expected throughput improvement
        all_recommendations.sort(
            key=lambda x: x["recommendation"]["expected_improvements"]["throughput_improvement"], 
            reverse=True
        )
        
        # Pick top 10 recommendations
        report["top_recommendations"] = all_recommendations[:10]
        
        return report
    
    def close(self):
        """Close database connections."""
        if self.benchmark_conn:
            self.benchmark_conn.close()
            self.benchmark_conn = None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Optimization Analyzer")
    parser.add_argument("--benchmark-db", type=str, default="benchmark_db.duckdb", 
                      help="Path to benchmark DuckDB database")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080",
                      help="URL of the Predictive Performance API")
    parser.add_argument("--api-key", type=str, help="API key for authenticated endpoints")
    parser.add_argument("--model", type=str, help="Analyze model")
    parser.add_argument("--hardware", type=str, help="Hardware platform")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8", "int4"], 
                      help="Current precision")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--models", type=str, nargs="+", help="List of models for report")
    parser.add_argument("--hardware-platforms", type=str, nargs="+", 
                      help="List of hardware platforms for report")
    parser.add_argument("--output", type=str, help="Path to write JSON output")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HardwareOptimizationAnalyzer(
        benchmark_db_path=args.benchmark_db,
        predictive_api_url=args.api_url,
        api_key=args.api_key
    )
    
    try:
        output = None
        
        # Generate report for multiple models and hardware
        if args.report:
            if not args.models or not args.hardware_platforms:
                print("ERROR: --models and --hardware-platforms are required for report generation")
                return 1
            
            print(f"Generating optimization report for {len(args.models)} models and {len(args.hardware_platforms)} hardware platforms...")
            report = analyzer.generate_optimization_report(
                model_names=args.models,
                hardware_platforms=args.hardware_platforms,
                batch_size=args.batch_size,
                current_precision=args.precision
            )
            
            print(f"Report generated with {len(report['top_recommendations'])} top recommendations")
            output = report
            
        # Get recommendations for specific model and hardware
        elif args.model and args.hardware:
            print(f"Analyzing {args.model} on {args.hardware}...")
            
            recommendations = analyzer.get_optimization_recommendations(
                model_name=args.model,
                hardware_platform=args.hardware,
                batch_size=args.batch_size,
                current_precision=args.precision
            )
            
            num_recommendations = len(recommendations.get("recommendations", []))
            print(f"Found {num_recommendations} optimization recommendations")
            
            output = recommendations
            
        else:
            print("ERROR: Please specify either --model and --hardware for single analysis or --report, --models, and --hardware-platforms for comprehensive report")
            return 1
        
        # Write output to file
        if args.output and output:
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Output written to {args.output}")
        elif output:
            print(json.dumps(output, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    finally:
        # Close connections
        analyzer.close()

if __name__ == "__main__":
    sys.exit(main())