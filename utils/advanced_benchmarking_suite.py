#!/usr/bin/env python3
"""
Advanced Benchmarking Suite
Comprehensive performance benchmarking with statistical analysis
"""

import time
import statistics
import math
import json
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Safe imports
try:
    from utils.safe_imports import safe_import
    from utils.enhanced_performance_modeling import EnhancedPerformanceModeling
    from hardware_detection import HardwareDetector
except ImportError:
    # Fallback imports
    def safe_import(module_name, fallback=None):
        try:
            return __import__(module_name)
        except ImportError:
            return fallback
    
    from enhanced_performance_modeling import EnhancedPerformanceModeling
    try:
        import sys
        sys.path.append('..')
        from hardware_detection import HardwareDetector
    except ImportError:
        HardwareDetector = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfiguration:
    """Benchmark configuration parameters."""
    model_name: str
    hardware_type: str
    batch_sizes: List[int]
    sequence_lengths: List[int]
    precisions: List[str]
    iterations: int
    warmup_iterations: int
    timeout_seconds: int
    statistical_confidence: float

@dataclass 
class BenchmarkResult:
    """Single benchmark execution result."""
    model_name: str
    hardware_type: str
    batch_size: int
    sequence_length: int
    precision: str
    inference_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    power_consumption_w: float
    efficiency_score: float
    timestamp: str

@dataclass
class StatisticalSummary:
    """Statistical summary of benchmark results."""
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    coefficient_of_variation: float
    confidence_interval_95: Tuple[float, float]
    sample_size: int

@dataclass
class BenchmarkAnalysis:
    """Comprehensive benchmark analysis."""
    model_name: str
    hardware_rankings: List[Tuple[str, float, str]]  # (hardware, score, reason)
    performance_trends: Dict[str, Dict[str, float]]
    optimization_impact: Dict[str, float]
    bottleneck_analysis: Dict[str, List[str]]
    scaling_analysis: Dict[str, float]  # batch_size -> throughput scaling
    memory_analysis: Dict[str, float]   # sequence_length -> memory usage
    power_efficiency: Dict[str, float]  # hardware -> efficiency score
    recommendations: List[str]

class AdvancedBenchmarkSuite:
    """Advanced benchmarking suite with statistical analysis."""
    
    # Predefined benchmark configurations
    STANDARD_CONFIGS = {
        "quick": BenchmarkConfiguration(
            model_name="bert-tiny",
            hardware_type="cpu",
            batch_sizes=[1, 4],
            sequence_lengths=[128, 512],
            precisions=["fp32"],
            iterations=5,
            warmup_iterations=2,
            timeout_seconds=30,
            statistical_confidence=0.90
        ),
        "comprehensive": BenchmarkConfiguration(
            model_name="bert-base",
            hardware_type="cuda",
            batch_sizes=[1, 4, 8, 16],
            sequence_lengths=[128, 256, 512, 1024],
            precisions=["fp32", "fp16", "int8"],
            iterations=10,
            warmup_iterations=3,
            timeout_seconds=300,
            statistical_confidence=0.95
        ),
        "production": BenchmarkConfiguration(
            model_name="gpt2-small",
            hardware_type="auto",
            batch_sizes=[1, 2, 4, 8],
            sequence_lengths=[512, 1024],
            precisions=["fp16", "int8"],
            iterations=20,
            warmup_iterations=5,
            timeout_seconds=600,
            statistical_confidence=0.99
        )
    }
    
    def __init__(self):
        """Initialize advanced benchmark suite."""
        logger.info("Initializing advanced benchmark suite...")
        
        try:
            self.performance_model = EnhancedPerformanceModeling()
            self.hardware_detector = HardwareDetector() if HardwareDetector else None
            self.results_cache = {}
            logger.info("Advanced benchmark suite initialized successfully")
        except Exception as e:
            logger.warning(f"Some components not available: {e}")
            self.performance_model = None
            self.hardware_detector = None
            self.results_cache = {}
    
    def run_benchmark_suite(
        self, 
        config: BenchmarkConfiguration,
        save_results: bool = True,
        parallel_execution: bool = False
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        
        logger.info(f"Starting benchmark suite for {config.model_name}")
        logger.info(f"Configuration: {config.iterations} iterations, "
                   f"{len(config.batch_sizes)} batch sizes, "
                   f"{len(config.precisions)} precisions")
        
        start_time = time.time()
        all_results = []
        
        # Get available hardware if auto-detection enabled
        if config.hardware_type == "auto" and self.hardware_detector:
            try:
                available_hardware = self.hardware_detector.get_available_hardware()
                hardware_list = available_hardware[:3]  # Test top 3
            except:
                hardware_list = ["cpu"]
        else:
            hardware_list = [config.hardware_type]
        
        # Generate all test combinations
        test_combinations = []
        for hardware in hardware_list:
            for batch_size in config.batch_sizes:
                for seq_length in config.sequence_lengths:
                    for precision in config.precisions:
                        test_combinations.append((hardware, batch_size, seq_length, precision))
        
        logger.info(f"Testing {len(test_combinations)} combinations")
        
        if parallel_execution and len(test_combinations) > 4:
            # Run benchmarks in parallel
            results = self._run_parallel_benchmarks(config, test_combinations)
        else:
            # Run benchmarks sequentially
            results = self._run_sequential_benchmarks(config, test_combinations)
        
        all_results.extend(results)
        
        # Statistical analysis
        analysis = self._perform_statistical_analysis(all_results)
        
        # Generate comprehensive report
        report = self._generate_benchmark_report(config, all_results, analysis)
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark suite completed in {total_time:.1f}s")
        
        if save_results:
            self._save_benchmark_results(config, report)
        
        return report
    
    def _run_sequential_benchmarks(
        self, config: BenchmarkConfiguration, test_combinations: List[Tuple]
    ) -> List[BenchmarkResult]:
        """Run benchmarks sequentially."""
        results = []
        
        for i, (hardware, batch_size, seq_length, precision) in enumerate(test_combinations):
            logger.info(f"Running test {i+1}/{len(test_combinations)}: "
                       f"{hardware}, batch={batch_size}, seq={seq_length}, {precision}")
            
            try:
                result = self._run_single_benchmark(
                    config.model_name, hardware, batch_size, seq_length, 
                    precision, config.iterations, config.warmup_iterations
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Benchmark failed for {hardware}: {e}")
                continue
        
        return results
    
    def _run_parallel_benchmarks(
        self, config: BenchmarkConfiguration, test_combinations: List[Tuple]
    ) -> List[BenchmarkResult]:
        """Run benchmarks in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(4, len(test_combinations))) as executor:
            # Submit all benchmark tasks
            future_to_config = {}
            for hardware, batch_size, seq_length, precision in test_combinations:
                future = executor.submit(
                    self._run_single_benchmark,
                    config.model_name, hardware, batch_size, seq_length,
                    precision, config.iterations, config.warmup_iterations
                )
                future_to_config[future] = (hardware, batch_size, seq_length, precision)
            
            # Collect results as they complete
            for future in as_completed(future_to_config, timeout=config.timeout_seconds):
                test_config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed benchmark: {test_config[0]}")
                except Exception as e:
                    logger.warning(f"Benchmark failed for {test_config}: {e}")
        
        return results
    
    def _run_single_benchmark(
        self, 
        model_name: str,
        hardware_type: str,
        batch_size: int,
        sequence_length: int,
        precision: str,
        iterations: int,
        warmup_iterations: int
    ) -> BenchmarkResult:
        """Run a single benchmark with statistical sampling."""
        
        if not self.performance_model:
            raise RuntimeError("Performance model not available")
        
        # Warmup runs (not measured)
        for _ in range(warmup_iterations):
            try:
                self.performance_model.simulate_inference_performance(
                    model_name, hardware_type, batch_size, sequence_length, precision
                )
            except:
                pass
        
        # Measured runs
        measurements = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            try:
                metrics = self.performance_model.simulate_inference_performance(
                    model_name, hardware_type, batch_size, sequence_length, precision
                )
                
                end_time = time.perf_counter()
                actual_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Use simulated metrics but real execution time for overhead
                final_time = metrics.inference_time_ms + (actual_time * 0.1)  # Add 10% overhead
                
                measurements.append({
                    'inference_time_ms': final_time,
                    'throughput': metrics.throughput_samples_per_sec,
                    'memory_usage': metrics.memory_usage_mb,
                    'power_consumption': metrics.power_consumption_w,
                    'efficiency_score': metrics.efficiency_score
                })
                
            except Exception as e:
                logger.warning(f"Measurement failed: {e}")
                continue
        
        if not measurements:
            raise RuntimeError("No successful measurements")
        
        # Calculate statistical averages
        avg_inference_time = statistics.mean([m['inference_time_ms'] for m in measurements])
        avg_throughput = statistics.mean([m['throughput'] for m in measurements])
        avg_memory = statistics.mean([m['memory_usage'] for m in measurements])
        avg_power = statistics.mean([m['power_consumption'] for m in measurements])
        avg_efficiency = statistics.mean([m['efficiency_score'] for m in measurements])
        
        return BenchmarkResult(
            model_name=model_name,
            hardware_type=hardware_type,
            batch_size=batch_size,
            sequence_length=sequence_length,
            precision=precision,
            inference_time_ms=avg_inference_time,
            throughput_samples_per_sec=avg_throughput,
            memory_usage_mb=avg_memory,
            power_consumption_w=avg_power,
            efficiency_score=avg_efficiency,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _perform_statistical_analysis(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        
        logger.info("Performing statistical analysis...")
        
        analysis = {
            'hardware_comparison': self._analyze_hardware_performance(results),
            'batch_size_scaling': self._analyze_batch_size_scaling(results),
            'precision_impact': self._analyze_precision_impact(results),
            'memory_scaling': self._analyze_memory_scaling(results),
            'power_efficiency': self._analyze_power_efficiency(results),
            'performance_variability': self._analyze_performance_variability(results)
        }
        
        return analysis
    
    def _analyze_hardware_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance across different hardware types."""
        
        hardware_performance = {}
        
        for result in results:
            if result.hardware_type not in hardware_performance:
                hardware_performance[result.hardware_type] = {
                    'inference_times': [],
                    'throughputs': [],
                    'efficiency_scores': []
                }
            
            hardware_performance[result.hardware_type]['inference_times'].append(
                result.inference_time_ms
            )
            hardware_performance[result.hardware_type]['throughputs'].append(
                result.throughput_samples_per_sec
            )
            hardware_performance[result.hardware_type]['efficiency_scores'].append(
                result.efficiency_score
            )
        
        # Calculate statistics for each hardware type
        hardware_stats = {}
        for hardware, data in hardware_performance.items():
            hardware_stats[hardware] = {
                'avg_inference_time': statistics.mean(data['inference_times']),
                'avg_throughput': statistics.mean(data['throughputs']),
                'avg_efficiency': statistics.mean(data['efficiency_scores']),
                'inference_time_std': statistics.stdev(data['inference_times']) if len(data['inference_times']) > 1 else 0,
                'sample_count': len(data['inference_times'])
            }
        
        # Rank hardware by overall performance score
        hardware_rankings = []
        for hardware, stats in hardware_stats.items():
            # Combined performance score (higher is better)
            score = (
                stats['avg_throughput'] * 0.4 +
                stats['avg_efficiency'] * 0.3 +
                (1000 / max(stats['avg_inference_time'], 1)) * 0.3
            )
            hardware_rankings.append((hardware, score, stats))
        
        hardware_rankings.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'statistics': hardware_stats,
            'rankings': hardware_rankings
        }
    
    def _analyze_batch_size_scaling(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze throughput scaling with batch size."""
        
        batch_scaling = {}
        
        for result in results:
            key = f"{result.hardware_type}_{result.precision}"
            if key not in batch_scaling:
                batch_scaling[key] = {}
            
            if result.batch_size not in batch_scaling[key]:
                batch_scaling[key][result.batch_size] = []
            
            batch_scaling[key][result.batch_size].append(result.throughput_samples_per_sec)
        
        # Calculate scaling efficiency
        scaling_analysis = {}
        for key, batch_data in batch_scaling.items():
            if len(batch_data) < 2:
                continue
                
            sorted_batches = sorted(batch_data.keys())
            base_batch = sorted_batches[0]
            base_throughput = statistics.mean(batch_data[base_batch])
            
            scaling_efficiency = {}
            for batch_size in sorted_batches:
                actual_throughput = statistics.mean(batch_data[batch_size])
                expected_throughput = base_throughput * batch_size / base_batch
                efficiency = actual_throughput / expected_throughput if expected_throughput > 0 else 0
                scaling_efficiency[batch_size] = min(efficiency, 2.0)  # Cap at 200%
            
            scaling_analysis[key] = scaling_efficiency
        
        return scaling_analysis
    
    def _analyze_precision_impact(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance impact of different precisions."""
        
        precision_impact = {}
        
        for result in results:
            key = f"{result.hardware_type}_{result.batch_size}"
            if key not in precision_impact:
                precision_impact[key] = {}
            
            if result.precision not in precision_impact[key]:
                precision_impact[key][result.precision] = {
                    'inference_times': [],
                    'throughputs': [],
                    'memory_usage': []
                }
            
            precision_impact[key][result.precision]['inference_times'].append(
                result.inference_time_ms
            )
            precision_impact[key][result.precision]['throughputs'].append(
                result.throughput_samples_per_sec
            )
            precision_impact[key][result.precision]['memory_usage'].append(
                result.memory_usage_mb
            )
        
        # Calculate relative improvements
        precision_analysis = {}
        for key, precision_data in precision_impact.items():
            if 'fp32' not in precision_data:
                continue  # Need fp32 baseline
                
            fp32_time = statistics.mean(precision_data['fp32']['inference_times'])
            fp32_memory = statistics.mean(precision_data['fp32']['memory_usage'])
            
            precision_analysis[key] = {}
            for precision in precision_data.keys():
                if precision == 'fp32':
                    continue
                    
                precision_time = statistics.mean(precision_data[precision]['inference_times'])
                precision_memory = statistics.mean(precision_data[precision]['memory_usage'])
                
                speedup = fp32_time / precision_time if precision_time > 0 else 1.0
                memory_reduction = fp32_memory / precision_memory if precision_memory > 0 else 1.0
                
                precision_analysis[key][precision] = {
                    'speedup': speedup,
                    'memory_reduction': memory_reduction
                }
        
        return precision_analysis
    
    def _analyze_memory_scaling(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze memory usage scaling with sequence length."""
        
        memory_scaling = {}
        
        for result in results:
            key = f"{result.hardware_type}_{result.precision}"
            if key not in memory_scaling:
                memory_scaling[key] = {}
            
            if result.sequence_length not in memory_scaling[key]:
                memory_scaling[key][result.sequence_length] = []
            
            memory_scaling[key][result.sequence_length].append(result.memory_usage_mb)
        
        # Analyze scaling patterns
        scaling_patterns = {}
        for key, seq_data in memory_scaling.items():
            if len(seq_data) < 2:
                continue
                
            sorted_seqs = sorted(seq_data.keys())
            scaling_factors = []
            
            for i in range(1, len(sorted_seqs)):
                prev_seq = sorted_seqs[i-1]
                curr_seq = sorted_seqs[i]
                
                prev_memory = statistics.mean(seq_data[prev_seq])
                curr_memory = statistics.mean(seq_data[curr_seq])
                
                seq_ratio = curr_seq / prev_seq
                memory_ratio = curr_memory / prev_memory if prev_memory > 0 else 1.0
                
                scaling_factor = memory_ratio / seq_ratio
                scaling_factors.append(scaling_factor)
            
            avg_scaling = statistics.mean(scaling_factors) if scaling_factors else 1.0
            scaling_patterns[key] = {
                'average_scaling_factor': avg_scaling,
                'memory_by_sequence': {seq: statistics.mean(mem_list) 
                                     for seq, mem_list in seq_data.items()}
            }
        
        return scaling_patterns
    
    def _analyze_power_efficiency(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze power efficiency across hardware types."""
        
        power_efficiency = {}
        
        for result in results:
            if result.hardware_type not in power_efficiency:
                power_efficiency[result.hardware_type] = {
                    'power_consumption': [],
                    'throughput': [],
                    'efficiency_scores': []
                }
            
            power_efficiency[result.hardware_type]['power_consumption'].append(
                result.power_consumption_w
            )
            power_efficiency[result.hardware_type]['throughput'].append(
                result.throughput_samples_per_sec
            )
            power_efficiency[result.hardware_type]['efficiency_scores'].append(
                result.efficiency_score
            )
        
        # Calculate efficiency metrics
        efficiency_analysis = {}
        for hardware, data in power_efficiency.items():
            avg_power = statistics.mean(data['power_consumption'])
            avg_throughput = statistics.mean(data['throughput'])
            avg_efficiency = statistics.mean(data['efficiency_scores'])
            
            # Performance per watt
            perf_per_watt = avg_throughput / avg_power if avg_power > 0 else 0
            
            efficiency_analysis[hardware] = {
                'average_power_w': avg_power,
                'average_throughput': avg_throughput,
                'performance_per_watt': perf_per_watt,
                'efficiency_score': avg_efficiency
            }
        
        return efficiency_analysis
    
    def _analyze_performance_variability(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance consistency and variability."""
        
        # Group results by configuration
        config_groups = {}
        for result in results:
            config_key = f"{result.hardware_type}_{result.batch_size}_{result.precision}"
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result.inference_time_ms)
        
        variability_analysis = {}
        for config, times in config_groups.items():
            if len(times) < 2:
                continue
                
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times)
            coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0
            
            variability_analysis[config] = {
                'mean_inference_time': mean_time,
                'standard_deviation': std_dev,
                'coefficient_of_variation': coefficient_of_variation,
                'consistency_score': max(0, 100 - coefficient_of_variation * 100)
            }
        
        return variability_analysis
    
    def _generate_benchmark_report(
        self, 
        config: BenchmarkConfiguration,
        results: List[BenchmarkResult],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        # Hardware rankings
        hardware_rankings = []
        if 'hardware_comparison' in analysis and 'rankings' in analysis['hardware_comparison']:
            for hardware, score, stats in analysis['hardware_comparison']['rankings']:
                reason = f"Avg throughput: {stats['avg_throughput']:.1f} samples/sec, "
                reason += f"Efficiency: {stats['avg_efficiency']:.1f}"
                hardware_rankings.append((hardware, score, reason))
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(analysis)
        
        report = {
            'configuration': asdict(config),
            'execution_summary': {
                'total_tests': len(results),
                'successful_tests': len([r for r in results if r.inference_time_ms > 0]),
                'hardware_types_tested': len(set(r.hardware_type for r in results)),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'performance_analysis': {
                'hardware_rankings': hardware_rankings,
                'best_throughput': max((r.throughput_samples_per_sec for r in results), default=0),
                'best_latency': min((r.inference_time_ms for r in results), default=float('inf')),
                'power_efficiency_leader': self._get_most_efficient_hardware(analysis),
            },
            'detailed_analysis': analysis,
            'optimization_recommendations': recommendations,
            'raw_results': [asdict(r) for r in results]
        }
        
        return report
    
    def _get_most_efficient_hardware(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Get the most power-efficient hardware."""
        if 'power_efficiency' not in analysis:
            return None
            
        best_hardware = None
        best_efficiency = 0
        
        for hardware, data in analysis['power_efficiency'].items():
            if data['performance_per_watt'] > best_efficiency:
                best_efficiency = data['performance_per_watt']
                best_hardware = hardware
        
        return best_hardware
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        
        recommendations = []
        
        # Hardware recommendations
        if 'hardware_comparison' in analysis and 'rankings' in analysis['hardware_comparison']:
            rankings = analysis['hardware_comparison']['rankings']
            if rankings:
                best_hardware = rankings[0][0]
                recommendations.append(
                    f"Use {best_hardware} for optimal performance"
                )
        
        # Precision recommendations
        if 'precision_impact' in analysis:
            for key, precisions in analysis['precision_impact'].items():
                best_speedup = 0
                best_precision = None
                for precision, metrics in precisions.items():
                    if metrics['speedup'] > best_speedup:
                        best_speedup = metrics['speedup']
                        best_precision = precision
                
                if best_precision and best_speedup > 1.2:
                    recommendations.append(
                        f"Use {best_precision} precision for {best_speedup:.1f}x speedup"
                    )
        
        # Batch size recommendations
        if 'batch_size_scaling' in analysis:
            for key, scaling in analysis['batch_size_scaling'].items():
                batch_sizes = sorted(scaling.keys())
                if len(batch_sizes) >= 2:
                    # Find optimal batch size (best efficiency)
                    best_batch = max(batch_sizes, key=lambda b: scaling[b])
                    if scaling[best_batch] > 0.8:  # Good efficiency
                        recommendations.append(
                            f"Use batch size {best_batch} for optimal throughput scaling"
                        )
        
        # Memory optimization recommendations
        if 'memory_scaling' in analysis:
            for key, scaling_data in analysis['memory_scaling'].items():
                if scaling_data['average_scaling_factor'] > 1.5:
                    recommendations.append(
                        "Consider memory optimization techniques for large sequences"
                    )
        
        # Power efficiency recommendations
        if 'power_efficiency' in analysis:
            efficient_hardware = []
            for hardware, data in analysis['power_efficiency'].items():
                if data['performance_per_watt'] > 10:  # Arbitrary threshold
                    efficient_hardware.append((hardware, data['performance_per_watt']))
            
            if efficient_hardware:
                efficient_hardware.sort(key=lambda x: x[1], reverse=True)
                best_efficient = efficient_hardware[0][0]
                recommendations.append(
                    f"Use {best_efficient} for best power efficiency"
                )
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _save_benchmark_results(self, config: BenchmarkConfiguration, report: Dict[str, Any]):
        """Save benchmark results to file."""
        try:
            results_dir = Path("benchmark_results")
            results_dir.mkdir(exist_ok=True)
            
            filename = f"benchmark_{config.model_name}_{int(time.time())}.json"
            filepath = results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")

def run_advanced_benchmark_demo():
    """Run advanced benchmark demonstration."""
    print("🚀 Advanced Benchmarking Suite Demo")
    print("=" * 50)
    
    suite = AdvancedBenchmarkSuite()
    
    # Run quick benchmark
    print("\n📊 Running Quick Benchmark...")
    config = suite.STANDARD_CONFIGS["quick"]
    
    try:
        report = suite.run_benchmark_suite(config, save_results=False)
        
        print(f"\n✅ Benchmark Results Summary:")
        print(f"   Total Tests: {report['execution_summary']['total_tests']}")
        print(f"   Successful Tests: {report['execution_summary']['successful_tests']}")
        
        if report['performance_analysis']['hardware_rankings']:
            best_hardware, score, reason = report['performance_analysis']['hardware_rankings'][0]
            print(f"   Best Hardware: {best_hardware} (score: {score:.1f})")
            print(f"   Best Latency: {report['performance_analysis']['best_latency']:.1f}ms")
            print(f"   Best Throughput: {report['performance_analysis']['best_throughput']:.1f} samples/sec")
        
        print(f"\n🎯 Top Optimization Recommendations:")
        for i, rec in enumerate(report['optimization_recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n✅ Advanced benchmarking demonstration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        logger.error(f"Benchmark error: {e}")
        return False

if __name__ == "__main__":
    success = run_advanced_benchmark_demo()
    exit(0 if success else 1)