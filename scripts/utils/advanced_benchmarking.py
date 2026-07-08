#!/usr/bin/env python3
"""
Advanced Benchmarking Suite for IPFS Accelerate Python

This module provides comprehensive benchmarking capabilities including
cross-platform performance comparison, regression detection, and optimization
recommendations for production deployment.
"""

import os
import sys
import time
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path

# Safe imports
try:
    from .safe_imports import safe_import
    from .model_compatibility import get_optimal_hardware, benchmark_model_performance
    from .performance_modeling import simulate_model_performance, HardwareType, PrecisionMode
    from ..hardware_detection import HardwareDetector
except ImportError:
    # Fallback imports for standalone execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.safe_imports import safe_import
    from utils.model_compatibility import get_optimal_hardware, benchmark_model_performance
    from utils.performance_modeling import simulate_model_performance, HardwareType, PrecisionMode
    from hardware_detection import HardwareDetector

# Optional visualization
matplotlib = safe_import('matplotlib')
if matplotlib:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks available."""
    LATENCY = "latency"           # Inference latency benchmarks
    THROUGHPUT = "throughput"     # Throughput benchmarks  
    MEMORY = "memory"             # Memory usage benchmarks
    ACCURACY = "accuracy"         # Model accuracy benchmarks
    POWER = "power"               # Power consumption benchmarks
    SCALABILITY = "scalability"   # Scalability benchmarks

class OptimizationTarget(Enum):
    """Optimization targets for benchmarking."""
    SPEED = "speed"               # Optimize for speed
    MEMORY = "memory"             # Optimize for memory usage
    POWER = "power"               # Optimize for power efficiency
    ACCURACY = "accuracy"         # Optimize for accuracy
    BALANCED = "balanced"         # Balanced optimization

@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    model: str
    hardware: str
    precision: str
    batch_size: int
    benchmark_type: BenchmarkType
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""
    run_id: str
    timestamp: float
    configuration: Dict[str, Any]
    results: List[BenchmarkResult]
    summary: Dict[str, Any]
    duration_seconds: float

@dataclass 
class PerformanceComparison:
    """Performance comparison between different configurations."""
    baseline_config: str
    comparison_config: str
    improvement_percent: float
    significant: bool
    confidence: float
    details: Dict[str, Any]

class AdvancedBenchmarkSuite:
    """Comprehensive benchmarking suite with advanced analysis."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ipfs_accelerate" / "benchmarks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.detector = HardwareDetector()
        self.benchmark_history = []
        
    def run_comprehensive_benchmark(
        self,
        models: Optional[List[str]] = None,
        hardware: Optional[List[str]] = None,
        benchmark_types: Optional[List[BenchmarkType]] = None,
        optimization_target: OptimizationTarget = OptimizationTarget.BALANCED,
        iterations: int = 3,
        parallel: bool = True
    ) -> BenchmarkRun:
        """Run comprehensive benchmark suite."""
        
        run_id = f"benchmark_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"Starting comprehensive benchmark run: {run_id}")
        
        # Default configurations
        if models is None:
            models = ["bert-base-uncased", "gpt2", "distilbert-base-uncased"]
        
        if hardware is None:
            hardware = self.detector.get_available_hardware()
        
        if benchmark_types is None:
            benchmark_types = [BenchmarkType.LATENCY, BenchmarkType.THROUGHPUT, BenchmarkType.MEMORY]
        
        configuration = {
            "models": models,
            "hardware": hardware,
            "benchmark_types": [bt.value for bt in benchmark_types],
            "optimization_target": optimization_target.value,
            "iterations": iterations,
            "parallel": parallel
        }
        
        results = []
        
        # Run benchmarks
        if parallel and len(models) * len(hardware) > 2:
            results = self._run_parallel_benchmarks(
                models, hardware, benchmark_types, iterations
            )
        else:
            results = self._run_sequential_benchmarks(
                models, hardware, benchmark_types, iterations  
            )
        
        # Generate summary
        summary = self._generate_benchmark_summary(results, optimization_target)
        
        duration = time.time() - start_time
        
        benchmark_run = BenchmarkRun(
            run_id=run_id,
            timestamp=start_time,
            configuration=configuration,
            results=results,
            summary=summary,
            duration_seconds=duration
        )
        
        # Save results
        self._save_benchmark_run(benchmark_run)
        self.benchmark_history.append(benchmark_run)
        
        logger.info(f"Benchmark run completed in {duration:.2f}s with {len(results)} results")
        
        return benchmark_run
    
    def _run_parallel_benchmarks(
        self,
        models: List[str],
        hardware: List[str], 
        benchmark_types: List[BenchmarkType],
        iterations: int
    ) -> List[BenchmarkResult]:
        """Run benchmarks in parallel for better performance."""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all benchmark tasks
            futures = []
            
            for model in models:
                for hw in hardware:
                    for benchmark_type in benchmark_types:
                        future = executor.submit(
                            self._run_single_benchmark,
                            model, hw, benchmark_type, iterations
                        )
                        futures.append((future, model, hw, benchmark_type))
            
            # Collect results as they complete
            for future, model, hw, benchmark_type in futures:
                try:
                    benchmark_results = future.result(timeout=60)
                    results.extend(benchmark_results)
                except Exception as e:
                    logger.error(f"Benchmark failed for {model}/{hw}/{benchmark_type.value}: {e}")
                    # Add failed result placeholder
                    results.append(BenchmarkResult(
                        model=model,
                        hardware=hw,
                        precision="fp32",
                        batch_size=1,
                        benchmark_type=benchmark_type,
                        value=0.0,
                        unit="error",
                        timestamp=time.time(),
                        metadata={"error": str(e)}
                    ))
        
        return results
    
    def _run_sequential_benchmarks(
        self,
        models: List[str],
        hardware: List[str],
        benchmark_types: List[BenchmarkType], 
        iterations: int
    ) -> List[BenchmarkResult]:
        """Run benchmarks sequentially."""
        
        results = []
        
        for model in models:
            for hw in hardware:
                for benchmark_type in benchmark_types:
                    try:
                        benchmark_results = self._run_single_benchmark(
                            model, hw, benchmark_type, iterations
                        )
                        results.extend(benchmark_results)
                    except Exception as e:
                        logger.error(f"Benchmark failed for {model}/{hw}/{benchmark_type.value}: {e}")
                        # Add failed result placeholder
                        results.append(BenchmarkResult(
                            model=model,
                            hardware=hw,
                            precision="fp32",
                            batch_size=1,
                            benchmark_type=benchmark_type,
                            value=0.0,
                            unit="error",
                            timestamp=time.time(),
                            metadata={"error": str(e)}
                        ))
        
        return results
    
    def _run_single_benchmark(
        self,
        model: str,
        hardware: str,
        benchmark_type: BenchmarkType,
        iterations: int
    ) -> List[BenchmarkResult]:
        """Run a single benchmark configuration multiple times."""
        
        results = []
        precisions = ["fp32", "fp16"] if hardware != "cpu" else ["fp32"]
        batch_sizes = [1, 4, 8] if benchmark_type == BenchmarkType.THROUGHPUT else [1]
        
        for precision in precisions:
            for batch_size in batch_sizes:
                iteration_results = []
                
                for i in range(iterations):
                    try:
                        if benchmark_type == BenchmarkType.LATENCY:
                            value, unit = self._benchmark_latency(model, hardware, precision, batch_size)
                        elif benchmark_type == BenchmarkType.THROUGHPUT:
                            value, unit = self._benchmark_throughput(model, hardware, precision, batch_size)
                        elif benchmark_type == BenchmarkType.MEMORY:
                            value, unit = self._benchmark_memory(model, hardware, precision, batch_size)
                        elif benchmark_type == BenchmarkType.POWER:
                            value, unit = self._benchmark_power(model, hardware, precision, batch_size)
                        elif benchmark_type == BenchmarkType.SCALABILITY:
                            value, unit = self._benchmark_scalability(model, hardware, precision, batch_size)
                        else:
                            value, unit = 0.0, "unknown"
                        
                        iteration_results.append(value)
                        
                    except Exception as e:
                        logger.warning(f"Iteration {i} failed for {model}/{hardware}: {e}")
                        continue
                
                # Calculate statistics from iterations
                if iteration_results:
                    avg_value = statistics.mean(iteration_results)
                    std_dev = statistics.stdev(iteration_results) if len(iteration_results) > 1 else 0.0
                    
                    metadata = {
                        "iterations": len(iteration_results),
                        "std_dev": std_dev,
                        "min_value": min(iteration_results),
                        "max_value": max(iteration_results),
                        "raw_results": iteration_results
                    }
                    
                    results.append(BenchmarkResult(
                        model=model,
                        hardware=hardware,
                        precision=precision,
                        batch_size=batch_size,
                        benchmark_type=benchmark_type,
                        value=avg_value,
                        unit=unit,
                        timestamp=time.time(),
                        metadata=metadata
                    ))
        
        return results
    
    def _benchmark_latency(self, model: str, hardware: str, precision: str, batch_size: int) -> Tuple[float, str]:
        """Benchmark inference latency."""
        result = simulate_model_performance(model, hardware, batch_size=batch_size, precision=precision)
        return result.inference_time_ms, "ms"
    
    def _benchmark_throughput(self, model: str, hardware: str, precision: str, batch_size: int) -> Tuple[float, str]:
        """Benchmark throughput (tokens/second)."""
        result = simulate_model_performance(model, hardware, batch_size=batch_size, precision=precision)
        
        # Estimate throughput based on latency and batch size
        if result.inference_time_ms > 0:
            throughput = (batch_size * 1000) / result.inference_time_ms
        else:
            throughput = 0.0
            
        return throughput, "samples/sec"
    
    def _benchmark_memory(self, model: str, hardware: str, precision: str, batch_size: int) -> Tuple[float, str]:
        """Benchmark memory usage."""
        result = simulate_model_performance(model, hardware, batch_size=batch_size, precision=precision)
        return result.memory_usage_mb, "MB"
    
    def _benchmark_power(self, model: str, hardware: str, precision: str, batch_size: int) -> Tuple[float, str]:
        """Benchmark power consumption (estimated)."""
        result = simulate_model_performance(model, hardware, batch_size=batch_size, precision=precision)
        
        # Estimate power based on hardware type and utilization
        base_power = {
            "cpu": 65.0,    # Watts
            "cuda": 250.0,
            "mps": 150.0,
            "webgpu": 100.0,
            "webnn": 50.0
        }.get(hardware, 100.0)
        
        # Scale by utilization (estimated from efficiency score)
        utilization = min(1.0, result.efficiency_score)
        estimated_power = base_power * utilization
        
        return estimated_power, "W"
    
    def _benchmark_scalability(self, model: str, hardware: str, precision: str, batch_size: int) -> Tuple[float, str]:
        """Benchmark scalability (performance scaling with batch size)."""
        
        # Test multiple batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        latencies = []
        
        for bs in batch_sizes:
            if bs <= batch_size:  # Only test up to requested batch size
                result = simulate_model_performance(model, hardware, batch_size=bs, precision=precision)
                latencies.append(result.inference_time_ms)
        
        # Calculate scalability score (lower is better scaling)
        if len(latencies) > 1:
            # Ideal scaling would be linear: latency(n) = latency(1) * n
            ideal_latencies = [latencies[0] * bs for bs in batch_sizes[:len(latencies)]]
            
            # Calculate how close we are to ideal scaling (0-1, higher is better)
            scalability_score = 0.0
            for i, (actual, ideal) in enumerate(zip(latencies, ideal_latencies)):
                if ideal > 0:
                    scaling_efficiency = min(1.0, ideal / actual)
                    scalability_score += scaling_efficiency
            
            scalability_score /= len(latencies)
        else:
            scalability_score = 1.0
            
        return scalability_score, "score"
    
    def _generate_benchmark_summary(
        self, 
        results: List[BenchmarkResult],
        optimization_target: OptimizationTarget
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary."""
        
        summary = {
            "total_benchmarks": len(results),
            "successful_benchmarks": len([r for r in results if r.unit != "error"]),
            "models_tested": list(set(r.model for r in results)),
            "hardware_tested": list(set(r.hardware for r in results)),
            "benchmark_types": list(set(r.benchmark_type.value for r in results))
        }
        
        # Calculate performance rankings
        summary["performance_rankings"] = self._calculate_performance_rankings(results, optimization_target)
        
        # Calculate optimization recommendations
        summary["optimization_recommendations"] = self._generate_optimization_recommendations(results)
        
        # Calculate statistics by category
        summary["statistics"] = self._calculate_benchmark_statistics(results)
        
        # Identify best configurations
        summary["best_configurations"] = self._identify_best_configurations(results, optimization_target)
        
        return summary
    
    def _calculate_performance_rankings(
        self,
        results: List[BenchmarkResult],
        optimization_target: OptimizationTarget
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Calculate performance rankings by hardware and model."""
        
        rankings = {
            "hardware": {},
            "models": {},
            "configurations": {}
        }
        
        # Group results by hardware
        hardware_scores = {}
        for result in results:
            if result.unit == "error":
                continue
                
            if result.hardware not in hardware_scores:
                hardware_scores[result.hardware] = []
            
            # Convert to normalized score (0-100, higher is better)
            score = self._calculate_normalized_score(result, optimization_target)
            hardware_scores[result.hardware].append(score)
        
        # Average scores by hardware
        for hardware, scores in hardware_scores.items():
            avg_score = statistics.mean(scores)
            rankings["hardware"][hardware] = avg_score
        
        # Sort hardware by performance
        sorted_hardware = sorted(rankings["hardware"].items(), key=lambda x: x[1], reverse=True)
        rankings["hardware_ranked"] = sorted_hardware
        
        # Similar for models
        model_scores = {}
        for result in results:
            if result.unit == "error":
                continue
                
            if result.model not in model_scores:
                model_scores[result.model] = []
            
            score = self._calculate_normalized_score(result, optimization_target)
            model_scores[result.model].append(score)
        
        for model, scores in model_scores.items():
            avg_score = statistics.mean(scores)
            rankings["models"][model] = avg_score
        
        sorted_models = sorted(rankings["models"].items(), key=lambda x: x[1], reverse=True)
        rankings["models_ranked"] = sorted_models
        
        return rankings
    
    def _calculate_normalized_score(self, result: BenchmarkResult, optimization_target: OptimizationTarget) -> float:
        """Calculate normalized performance score (0-100)."""
        
        if result.benchmark_type == BenchmarkType.LATENCY:
            # Lower latency is better, so invert
            if result.value > 0:
                score = max(0, 100 - (result.value / 10))  # Normalize around 100ms
            else:
                score = 0
        elif result.benchmark_type == BenchmarkType.THROUGHPUT:
            # Higher throughput is better
            score = min(100, result.value * 2)  # Normalize around 50 samples/sec
        elif result.benchmark_type == BenchmarkType.MEMORY:
            # Lower memory usage is better for memory optimization
            if optimization_target == OptimizationTarget.MEMORY:
                score = max(0, 100 - (result.value / 50))  # Normalize around 5GB
            else:
                score = 50  # Neutral score if not optimizing for memory
        elif result.benchmark_type == BenchmarkType.POWER:
            # Lower power is better for power optimization
            if optimization_target == OptimizationTarget.POWER:
                score = max(0, 100 - (result.value / 5))  # Normalize around 500W
            else:
                score = 50
        elif result.benchmark_type == BenchmarkType.SCALABILITY:
            # Higher scalability score is better
            score = result.value * 100
        else:
            score = 50  # Default neutral score
        
        return max(0, min(100, score))
    
    def _generate_optimization_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate optimization recommendations based on benchmark results."""
        
        recommendations = []
        
        # Analyze latency results
        latency_results = [r for r in results if r.benchmark_type == BenchmarkType.LATENCY and r.unit != "error"]
        if latency_results:
            avg_latency = statistics.mean(r.value for r in latency_results)
            if avg_latency > 100:  # ms
                recommendations.append("Consider using hardware acceleration or model optimization to reduce latency")
            
            # Find best hardware for latency
            hardware_latencies = {}
            for result in latency_results:
                if result.hardware not in hardware_latencies:
                    hardware_latencies[result.hardware] = []
                hardware_latencies[result.hardware].append(result.value)
            
            best_hardware = min(hardware_latencies.items(), key=lambda x: statistics.mean(x[1]))
            recommendations.append(f"For lowest latency, use {best_hardware[0]} hardware")
        
        # Analyze memory results
        memory_results = [r for r in results if r.benchmark_type == BenchmarkType.MEMORY and r.unit != "error"]
        if memory_results:
            max_memory = max(r.value for r in memory_results)
            if max_memory > 4000:  # MB
                recommendations.append("Consider using lower precision (fp16/int8) or smaller models to reduce memory usage")
        
        # Analyze precision impact
        fp32_results = [r for r in results if r.precision == "fp32" and r.unit != "error"]
        fp16_results = [r for r in results if r.precision == "fp16" and r.unit != "error"]
        
        if fp32_results and fp16_results:
            fp32_avg = statistics.mean(r.value for r in fp32_results if r.benchmark_type == BenchmarkType.LATENCY)
            fp16_avg = statistics.mean(r.value for r in fp16_results if r.benchmark_type == BenchmarkType.LATENCY)
            
            if fp16_avg < fp32_avg * 0.8:
                speedup = ((fp32_avg - fp16_avg) / fp32_avg) * 100
                recommendations.append(f"Using FP16 precision can provide {speedup:.1f}% latency improvement")
        
        # Batch size recommendations
        throughput_results = [r for r in results if r.benchmark_type == BenchmarkType.THROUGHPUT and r.unit != "error"]
        if throughput_results:
            # Find optimal batch size
            batch_throughputs = {}
            for result in throughput_results:
                batch_size = result.batch_size
                if batch_size not in batch_throughputs:
                    batch_throughputs[batch_size] = []
                batch_throughputs[batch_size].append(result.value)
            
            if batch_throughputs:
                best_batch = max(batch_throughputs.items(), key=lambda x: statistics.mean(x[1]))
                recommendations.append(f"For maximum throughput, use batch size {best_batch[0]}")
        
        return recommendations
    
    def _calculate_benchmark_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate detailed benchmark statistics."""
        
        stats = {}
        
        # Statistics by benchmark type
        for benchmark_type in BenchmarkType:
            type_results = [r for r in results if r.benchmark_type == benchmark_type and r.unit != "error"]
            if not type_results:
                continue
                
            values = [r.value for r in type_results]
            stats[benchmark_type.value] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values)
            }
        
        # Overall statistics
        all_successful = [r for r in results if r.unit != "error"]
        stats["overall"] = {
            "success_rate": len(all_successful) / len(results) * 100 if results else 0,
            "total_benchmarks": len(results),
            "unique_configurations": len(set((r.model, r.hardware, r.precision) for r in results))
        }
        
        return stats
    
    def _identify_best_configurations(
        self,
        results: List[BenchmarkResult], 
        optimization_target: OptimizationTarget
    ) -> Dict[str, Any]:
        """Identify best performing configurations."""
        
        configurations = {}
        
        # Group results by configuration
        config_groups = {}
        for result in results:
            if result.unit == "error":
                continue
                
            config_key = f"{result.model}_{result.hardware}_{result.precision}"
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)
        
        # Score each configuration
        config_scores = {}
        for config_key, config_results in config_groups.items():
            total_score = 0
            for result in config_results:
                score = self._calculate_normalized_score(result, optimization_target)
                total_score += score
            
            avg_score = total_score / len(config_results)
            config_scores[config_key] = avg_score
        
        # Find best configurations
        if config_scores:
            sorted_configs = sorted(config_scores.items(), key=lambda x: x[1], reverse=True)
            
            configurations["best_overall"] = {
                "configuration": sorted_configs[0][0],
                "score": sorted_configs[0][1]
            }
            
            configurations["top_5"] = sorted_configs[:5]
            
            # Best by hardware type
            hardware_best = {}
            for config_key, score in sorted_configs:
                hardware = config_key.split('_')[1]
                if hardware not in hardware_best:
                    hardware_best[hardware] = (config_key, score)
            
            configurations["best_by_hardware"] = hardware_best
        
        return configurations
    
    def _save_benchmark_run(self, benchmark_run: BenchmarkRun) -> None:
        """Save benchmark run to cache."""
        try:
            output_file = self.cache_dir / f"{benchmark_run.run_id}.json"
            with open(output_file, 'w') as f:
                json.dump(asdict(benchmark_run), f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def compare_benchmark_runs(
        self, 
        run1_id: str, 
        run2_id: str
    ) -> List[PerformanceComparison]:
        """Compare two benchmark runs and identify significant differences."""
        
        comparisons = []
        
        try:
            # Load benchmark runs
            run1 = self._load_benchmark_run(run1_id)
            run2 = self._load_benchmark_run(run2_id)
            
            if not run1 or not run2:
                logger.error("Could not load benchmark runs for comparison")
                return comparisons
            
            # Group results by configuration
            run1_configs = {}
            for result in run1.results:
                config_key = f"{result.model}_{result.hardware}_{result.precision}_{result.benchmark_type.value}"
                run1_configs[config_key] = result
            
            run2_configs = {}
            for result in run2.results:
                config_key = f"{result.model}_{result.hardware}_{result.precision}_{result.benchmark_type.value}"
                run2_configs[config_key] = result
            
            # Compare matching configurations
            common_configs = set(run1_configs.keys()) & set(run2_configs.keys())
            
            for config in common_configs:
                result1 = run1_configs[config]
                result2 = run2_configs[config]
                
                if result1.unit == "error" or result2.unit == "error":
                    continue
                
                # Calculate improvement percentage
                if result1.value != 0:
                    improvement = ((result2.value - result1.value) / result1.value) * 100
                else:
                    improvement = 0.0
                
                # Determine significance (simple threshold-based)
                significant = abs(improvement) > 5.0  # 5% threshold
                
                # Estimate confidence (placeholder - would need more sophisticated analysis)
                confidence = 0.8 if significant else 0.3
                
                comparison = PerformanceComparison(
                    baseline_config=f"{run1_id}:{config}",
                    comparison_config=f"{run2_id}:{config}",
                    improvement_percent=improvement,
                    significant=significant,
                    confidence=confidence,
                    details={
                        "baseline_value": result1.value,
                        "comparison_value": result2.value,
                        "unit": result1.unit,
                        "benchmark_type": result1.benchmark_type.value
                    }
                )
                
                comparisons.append(comparison)
        
        except Exception as e:
            logger.error(f"Failed to compare benchmark runs: {e}")
        
        return comparisons
    
    def _load_benchmark_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """Load benchmark run from cache."""
        try:
            run_file = self.cache_dir / f"{run_id}.json"
            if not run_file.exists():
                return None
                
            with open(run_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct BenchmarkRun object
            results = []
            for result_data in data.get("results", []):
                result = BenchmarkResult(
                    model=result_data["model"],
                    hardware=result_data["hardware"],
                    precision=result_data["precision"],
                    batch_size=result_data["batch_size"],
                    benchmark_type=BenchmarkType(result_data["benchmark_type"]),
                    value=result_data["value"],
                    unit=result_data["unit"],
                    timestamp=result_data["timestamp"],
                    metadata=result_data["metadata"]
                )
                results.append(result)
            
            return BenchmarkRun(
                run_id=data["run_id"],
                timestamp=data["timestamp"],
                configuration=data["configuration"],
                results=results,
                summary=data["summary"],
                duration_seconds=data["duration_seconds"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load benchmark run {run_id}: {e}")
            return None
    
    def generate_benchmark_report(
        self,
        benchmark_run: BenchmarkRun,
        output_file: Optional[str] = None,
        include_charts: bool = True
    ) -> str:
        """Generate comprehensive benchmark report."""
        
        report_lines = []
        
        # Header
        report_lines.append("="*80)
        report_lines.append("🚀 IPFS ACCELERATE PYTHON - BENCHMARK REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Run information
        report_lines.append(f"📊 Benchmark Run: {benchmark_run.run_id}")
        report_lines.append(f"🕐 Timestamp: {time.ctime(benchmark_run.timestamp)}")
        report_lines.append(f"⏱️  Duration: {benchmark_run.duration_seconds:.2f} seconds")
        report_lines.append("")
        
        # Configuration
        report_lines.append("⚙️  Configuration:")
        config = benchmark_run.configuration
        report_lines.append(f"  • Models: {', '.join(config.get('models', []))}")
        report_lines.append(f"  • Hardware: {', '.join(config.get('hardware', []))}")
        report_lines.append(f"  • Benchmark Types: {', '.join(config.get('benchmark_types', []))}")
        report_lines.append(f"  • Iterations: {config.get('iterations', 1)}")
        report_lines.append(f"  • Parallel: {config.get('parallel', False)}")
        report_lines.append("")
        
        # Summary statistics
        summary = benchmark_run.summary
        report_lines.append("📈 Summary Statistics:")
        report_lines.append(f"  • Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        report_lines.append(f"  • Successful: {summary.get('successful_benchmarks', 0)}")
        
        if "statistics" in summary:
            stats = summary["statistics"]
            if "overall" in stats:
                overall = stats["overall"]
                report_lines.append(f"  • Success Rate: {overall.get('success_rate', 0):.1f}%")
                report_lines.append(f"  • Unique Configurations: {overall.get('unique_configurations', 0)}")
        
        report_lines.append("")
        
        # Performance rankings
        if "performance_rankings" in summary:
            rankings = summary["performance_rankings"]
            
            if "hardware_ranked" in rankings:
                report_lines.append("🏆 Hardware Performance Rankings:")
                for i, (hardware, score) in enumerate(rankings["hardware_ranked"], 1):
                    report_lines.append(f"  {i}. {hardware}: {score:.1f}/100")
                report_lines.append("")
            
            if "models_ranked" in rankings:
                report_lines.append("🏆 Model Performance Rankings:")
                for i, (model, score) in enumerate(rankings["models_ranked"], 1):
                    report_lines.append(f"  {i}. {model}: {score:.1f}/100")
                report_lines.append("")
        
        # Best configurations
        if "best_configurations" in summary:
            best_configs = summary["best_configurations"]
            
            if "best_overall" in best_configs:
                best = best_configs["best_overall"]
                report_lines.append(f"🥇 Best Overall Configuration:")
                report_lines.append(f"  • {best.get('configuration', 'Unknown')}")
                report_lines.append(f"  • Score: {best.get('score', 0):.1f}/100")
                report_lines.append("")
        
        # Optimization recommendations
        if "optimization_recommendations" in summary:
            recommendations = summary["optimization_recommendations"]
            if recommendations:
                report_lines.append("💡 Optimization Recommendations:")
                for rec in recommendations:
                    report_lines.append(f"  • {rec}")
                report_lines.append("")
        
        # Detailed results by benchmark type
        results_by_type = {}
        for result in benchmark_run.results:
            if result.unit == "error":
                continue
            if result.benchmark_type not in results_by_type:
                results_by_type[result.benchmark_type] = []
            results_by_type[result.benchmark_type].append(result)
        
        for benchmark_type, results in results_by_type.items():
            report_lines.append(f"📊 {benchmark_type.value.title()} Results:")
            
            # Sort by value (best first)
            if benchmark_type in [BenchmarkType.LATENCY, BenchmarkType.MEMORY, BenchmarkType.POWER]:
                sorted_results = sorted(results, key=lambda r: r.value)
            else:
                sorted_results = sorted(results, key=lambda r: r.value, reverse=True)
            
            for i, result in enumerate(sorted_results[:10], 1):  # Top 10
                config = f"{result.model}/{result.hardware}/{result.precision}"
                report_lines.append(f"  {i:2d}. {config:<40} {result.value:8.2f} {result.unit}")
            
            report_lines.append("")
        
        # Footer
        report_lines.append("="*80)
        report_lines.append("Report generated by IPFS Accelerate Python Benchmark Suite")
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Benchmark report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report to {output_file}: {e}")
        
        return report_text

def run_quick_benchmark() -> BenchmarkRun:
    """Run a quick benchmark for immediate feedback."""
    
    suite = AdvancedBenchmarkSuite()
    
    # Quick configuration
    models = ["bert-base-uncased", "gpt2"]
    all_hardware = suite.detector.get_available_hardware()
    hardware = list(all_hardware)[:2]  # Limit to 2 best hardware (convert to list first)
    benchmark_types = [BenchmarkType.LATENCY, BenchmarkType.MEMORY]
    
    return suite.run_comprehensive_benchmark(
        models=models,
        hardware=hardware,
        benchmark_types=benchmark_types,
        iterations=1,  # Single iteration for speed
        parallel=True
    )

def run_full_benchmark() -> BenchmarkRun:
    """Run comprehensive benchmark suite."""
    
    suite = AdvancedBenchmarkSuite()
    
    return suite.run_comprehensive_benchmark(
        iterations=3,
        parallel=True
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IPFS Accelerate Python Advanced Benchmarking")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    parser.add_argument("--hardware", nargs="+", help="Specific hardware to test")
    parser.add_argument("--output", help="Save report to file")
    parser.add_argument("--compare", nargs=2, help="Compare two benchmark runs")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        if args.compare:
            suite = AdvancedBenchmarkSuite()
            comparisons = suite.compare_benchmark_runs(args.compare[0], args.compare[1])
            
            print("\n🔄 Benchmark Comparison Results:")
            print("="*60)
            
            for comp in comparisons:
                direction = "improvement" if comp.improvement_percent > 0 else "regression"
                significance = "significant" if comp.significant else "minor"
                
                print(f"📊 {comp.details['benchmark_type']}: {comp.improvement_percent:+.1f}% {direction} ({significance})")
                print(f"   Baseline: {comp.details['baseline_value']:.2f} {comp.details['unit']}")
                print(f"   Current:  {comp.details['comparison_value']:.2f} {comp.details['unit']}")
                print()
        
        elif args.quick:
            print("🚀 Running quick benchmark...")
            benchmark_run = run_quick_benchmark()
            
        elif args.full:
            print("🚀 Running full benchmark suite...")
            benchmark_run = run_full_benchmark()
            
        else:
            # Custom benchmark
            suite = AdvancedBenchmarkSuite()
            benchmark_run = suite.run_comprehensive_benchmark(
                models=args.models,
                hardware=args.hardware
            )
        
        if not args.compare:
            # Generate and display report
            if 'suite' not in locals():
                suite = AdvancedBenchmarkSuite()
            report = suite.generate_benchmark_report(benchmark_run, args.output)
            print(report)
            
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        logger.exception("Benchmark error details:")
        sys.exit(1)