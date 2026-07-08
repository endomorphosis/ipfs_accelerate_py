"""
Plotting utilities for benchmark results.

This module provides functions for creating visualizations of benchmark results,
including power efficiency and memory bandwidth metrics.
"""

import os
import logging
import numpy as np
from typing import Optional, List, Dict, Any

logger = logging.getLogger("benchmark.visualizers.plots")

def plot_latency_comparison(benchmark_results, output_path: Optional[str] = None, include_percentiles: bool = True):
    """
    Plot latency comparison across hardware platforms with optional percentiles.
    
    Args:
        benchmark_results: BenchmarkResults instance
        output_path: Path to save the plot
        include_percentiles: Whether to include p90, p95, p99 percentiles
        
    Returns:
        Path to the saved plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data
        hardware_platforms = []
        batch_sizes = []
        latencies = []
        # Also extract percentiles if available and requested
        p90_latencies = []
        p95_latencies = []
        p99_latencies = []
        
        for result in benchmark_results.results:
            if "latency_ms" in result.metrics:
                hardware_platforms.append(result.hardware)
                batch_sizes.append(result.batch_size)
                latencies.append(result.metrics["latency_ms"])
                
                # Extract percentiles if available
                if include_percentiles:
                    p90_latencies.append(result.metrics.get("latency_p90_ms", None))
                    p95_latencies.append(result.metrics.get("latency_p95_ms", None))
                    p99_latencies.append(result.metrics.get("latency_p99_ms", None))
        
        if not latencies:
            logger.warning("No latency data available for plotting")
            return None
            
        # Check if we have valid percentile data
        has_percentiles = include_percentiles and None not in p90_latencies and len(p90_latencies) == len(latencies)
        
        # Create default output path if not provided
        if output_path is None:
            os.makedirs(benchmark_results.config.output_dir, exist_ok=True)
            safe_model_id = benchmark_results.config.model_id.replace('/', '__')
            output_path = os.path.join(
                benchmark_results.config.output_dir,
                f"{safe_model_id}_latency_comparison.png"
            )
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Group by hardware platform
        unique_hardware = sorted(set(hardware_platforms))
        unique_batch_sizes = sorted(set(batch_sizes))
        
        # Plot barplot
        bar_width = 0.8 / len(unique_hardware)
        x = np.arange(len(unique_batch_sizes))
        
        # Create a figure with enough height for the legend
        if has_percentiles:
            plt.figure(figsize=(12, 8))
        
        for i, hw in enumerate(unique_hardware):
            hw_latencies = []
            hw_p90_latencies = []
            hw_p95_latencies = []
            hw_p99_latencies = []
            
            for bs in unique_batch_sizes:
                # Find matching results
                matches_indices = [
                    j for j in range(len(latencies))
                    if hardware_platforms[j] == hw and batch_sizes[j] == bs
                ]
                
                matches = [latencies[j] for j in matches_indices]
                
                # Use average if multiple matches
                if matches:
                    hw_latencies.append(sum(matches) / len(matches))
                    
                    # Get percentiles if available
                    if has_percentiles:
                        hw_p90_latencies.append(sum(p90_latencies[j] for j in matches_indices) / len(matches_indices))
                        hw_p95_latencies.append(sum(p95_latencies[j] for j in matches_indices) / len(matches_indices))
                        hw_p99_latencies.append(sum(p99_latencies[j] for j in matches_indices) / len(matches_indices))
                else:
                    hw_latencies.append(0)
                    if has_percentiles:
                        hw_p90_latencies.append(0)
                        hw_p95_latencies.append(0)
                        hw_p99_latencies.append(0)
            
            # Plot mean latency bars
            bar_positions = x + i * bar_width - (len(unique_hardware) - 1) * bar_width / 2
            bars = plt.bar(
                bar_positions,
                hw_latencies,
                width=bar_width,
                label=f"{hw.upper()} (Mean)",
                alpha=0.7
            )
            
            # Add percentile lines if available
            if has_percentiles:
                plt.plot(bar_positions, hw_p90_latencies, 'o-', color='green', alpha=0.7, 
                         linewidth=1.5, markersize=4, label=f"{hw.upper()} (p90)" if i == 0 else "")
                plt.plot(bar_positions, hw_p95_latencies, 'o-', color='orange', alpha=0.7, 
                         linewidth=1.5, markersize=4, label=f"{hw.upper()} (p95)" if i == 0 else "")
                plt.plot(bar_positions, hw_p99_latencies, 'o-', color='red', alpha=0.7, 
                         linewidth=1.5, markersize=4, label=f"{hw.upper()} (p99)" if i == 0 else "")
        
        # Customize plot
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (ms)')
        plt.title(f'Latency Comparison for {benchmark_results.config.model_id}')
        plt.xticks(x, unique_batch_sizes)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved latency comparison plot to {output_path}")
        return output_path
        
    except ImportError:
        logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
        return None
    except Exception as e:
        logger.error(f"Error plotting latency comparison: {e}")
        return None

def plot_throughput_scaling(benchmark_results, output_path: Optional[str] = None):
    """
    Plot throughput scaling with batch size.
    
    Args:
        benchmark_results: BenchmarkResults instance
        output_path: Path to save the plot
        
    Returns:
        Path to the saved plot
    """
    try:
        import matplotlib.pyplot as plt
        
        # Extract data
        hardware_platforms = []
        batch_sizes = []
        throughputs = []
        
        for result in benchmark_results.results:
            if "throughput_items_per_sec" in result.metrics:
                hardware_platforms.append(result.hardware)
                batch_sizes.append(result.batch_size)
                throughputs.append(result.metrics["throughput_items_per_sec"])
        
        if not throughputs:
            logger.warning("No throughput data available for plotting")
            return None
        
        # Create default output path if not provided
        if output_path is None:
            os.makedirs(benchmark_results.config.output_dir, exist_ok=True)
            safe_model_id = benchmark_results.config.model_id.replace('/', '__')
            output_path = os.path.join(
                benchmark_results.config.output_dir,
                f"{safe_model_id}_throughput_scaling.png"
            )
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Group by hardware platform
        unique_hardware = sorted(set(hardware_platforms))
        
        # Create line plot for each hardware platform
        for hw in unique_hardware:
            hw_batch_sizes = []
            hw_throughputs = []
            
            for i in range(len(throughputs)):
                if hardware_platforms[i] == hw:
                    hw_batch_sizes.append(batch_sizes[i])
                    hw_throughputs.append(throughputs[i])
            
            # Sort by batch size
            sorted_data = sorted(zip(hw_batch_sizes, hw_throughputs))
            hw_batch_sizes = [x[0] for x in sorted_data]
            hw_throughputs = [x[1] for x in sorted_data]
            
            plt.plot(hw_batch_sizes, hw_throughputs, marker='o', label=hw.upper())
        
        # Customize plot
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (items/sec)')
        plt.title(f'Throughput Scaling for {benchmark_results.config.model_id}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved throughput scaling plot to {output_path}")
        return output_path
        
    except ImportError:
        logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
        return None
    except Exception as e:
        logger.error(f"Error plotting throughput scaling: {e}")
        return None

def plot_memory_usage(benchmark_results, output_path: Optional[str] = None, detailed: bool = True):
    """
    Plot memory usage across hardware platforms with optional detailed breakdown.
    
    Args:
        benchmark_results: BenchmarkResults instance
        output_path: Path to save the plot
        detailed: Whether to include detailed memory metrics (allocated, reserved)
        
    Returns:
        Path to the saved plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data
        hardware_platforms = []
        batch_sizes = []
        memory_usages = []
        
        # Detailed memory metrics if available
        memory_allocated = []
        memory_reserved = []
        memory_peak = []
        cpu_memory = []
        
        for result in benchmark_results.results:
            if "memory_usage_mb" in result.metrics:
                hardware_platforms.append(result.hardware)
                batch_sizes.append(result.batch_size)
                memory_usages.append(result.metrics["memory_usage_mb"])
                
                # Extract detailed metrics if available and requested
                if detailed:
                    memory_allocated.append(result.metrics.get("memory_allocated_end_mb", None))
                    memory_reserved.append(result.metrics.get("memory_reserved_end_mb", None))
                    memory_peak.append(result.metrics.get("memory_peak_mb", None))
                    cpu_memory.append(result.metrics.get("cpu_memory_end_mb", None))
        
        if not memory_usages:
            logger.warning("No memory data available for plotting")
            return None
            
        # Check if we have valid detailed memory data
        has_allocated = detailed and None not in memory_allocated and len(memory_allocated) == len(memory_usages)
        has_reserved = detailed and None not in memory_reserved and len(memory_reserved) == len(memory_usages)
        has_peak = detailed and None not in memory_peak and len(memory_peak) == len(memory_usages)
        has_cpu = detailed and None not in cpu_memory and len(cpu_memory) == len(memory_usages)
        
        # Determine the number of subplots needed
        if detailed and (has_allocated or has_reserved or has_peak or has_cpu):
            # If we have detailed metrics, create a figure with subplots
            num_plots = 1 + has_allocated + has_reserved + has_peak + has_cpu
            fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots), sharex=True)
            if num_plots == 1:
                axes = [axes]  # Make it a list for consistent indexing
            current_ax = 0
        else:
            # Simple memory usage plot
            plt.figure(figsize=(10, 6))
        
        # Create default output path if not provided
        if output_path is None:
            os.makedirs(benchmark_results.config.output_dir, exist_ok=True)
            safe_model_id = benchmark_results.config.model_id.replace('/', '__')
            output_path = os.path.join(
                benchmark_results.config.output_dir,
                f"{safe_model_id}_memory_usage.png"
            )
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Group by hardware platform
        unique_hardware = sorted(set(hardware_platforms))
        unique_batch_sizes = sorted(set(batch_sizes))
        
        # Create helper function for plotting memory bar charts
        def plot_memory_bars(ax, data_array, title, ylabel):
            bar_width = 0.8 / len(unique_hardware)
            x = np.arange(len(unique_batch_sizes))
            
            for i, hw in enumerate(unique_hardware):
                hw_data = []
                for bs in unique_batch_sizes:
                    # Find matching results
                    matches_indices = [
                        j for j in range(len(data_array))
                        if hardware_platforms[j] == hw and batch_sizes[j] == bs
                    ]
                    
                    matches = [data_array[j] for j in matches_indices]
                    
                    # Use average if multiple matches
                    if matches:
                        hw_data.append(sum(matches) / len(matches))
                    else:
                        hw_data.append(0)
                
                ax.bar(
                    x + i * bar_width - (len(unique_hardware) - 1) * bar_width / 2,
                    hw_data,
                    width=bar_width,
                    label=hw.upper()
                )
            
            # Customize plot
            ax.set_xlabel('Batch Size')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(unique_batch_sizes)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Determine which axis to use for each plot
        if detailed and (has_allocated or has_reserved or has_peak or has_cpu):
            # Plot total memory usage
            ax = axes[current_ax]
            plot_memory_bars(ax, memory_usages, f'Total Memory Usage for {benchmark_results.config.model_id}', 'Memory Usage (MB)')
            current_ax += 1
            
            # Plot peak memory if available
            if has_peak:
                ax = axes[current_ax]
                plot_memory_bars(ax, memory_peak, 'Peak GPU Memory', 'Memory (MB)')
                current_ax += 1
            
            # Plot allocated memory if available
            if has_allocated:
                ax = axes[current_ax]
                plot_memory_bars(ax, memory_allocated, 'Allocated GPU Memory', 'Memory (MB)')
                current_ax += 1
            
            # Plot reserved memory if available
            if has_reserved:
                ax = axes[current_ax]
                plot_memory_bars(ax, memory_reserved, 'Reserved GPU Memory', 'Memory (MB)')
                current_ax += 1
            
            # Plot CPU memory if available
            if has_cpu:
                ax = axes[current_ax]
                plot_memory_bars(ax, cpu_memory, 'CPU Memory', 'Memory (MB)')
                current_ax += 1
            
            # Adjust layout
            plt.tight_layout()
            
        else:
            # Simple memory usage plot (just one plot)
            bar_width = 0.8 / len(unique_hardware)
            x = np.arange(len(unique_batch_sizes))
            
            for i, hw in enumerate(unique_hardware):
                hw_memory_usages = []
                for bs in unique_batch_sizes:
                    # Find matching results
                    matches = [
                        memory_usages[j] for j in range(len(memory_usages))
                        if hardware_platforms[j] == hw and batch_sizes[j] == bs
                    ]
                    
                    # Use average if multiple matches
                    if matches:
                        hw_memory_usages.append(sum(matches) / len(matches))
                    else:
                        hw_memory_usages.append(0)
                
                plt.bar(
                    x + i * bar_width - (len(unique_hardware) - 1) * bar_width / 2,
                    hw_memory_usages,
                    width=bar_width,
                    label=hw.upper()
                )
            
            # Customize plot
            plt.xlabel('Batch Size')
            plt.ylabel('Memory Usage (MB)')
            plt.title(f'Memory Usage for {benchmark_results.config.model_id}')
            plt.xticks(x, unique_batch_sizes)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved memory usage plot to {output_path}")
        return output_path
        
    except ImportError:
        logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
        return None
    except Exception as e:
        logger.error(f"Error plotting memory usage: {e}")
        return None

def plot_flops_comparison(benchmark_results, output_path: Optional[str] = None, detailed: bool = True):
    """
    Plot FLOPs comparison across hardware platforms and batch sizes.
    
    Args:
        benchmark_results: BenchmarkResults instance
        output_path: Path to save the plot
        detailed: Whether to include detailed FLOPs breakdown
        
    Returns:
        Path to the saved plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data
        hardware_platforms = []
        batch_sizes = []
        flops_values = []
        model_types = []
        detailed_breakdowns = []
        
        for result in benchmark_results.results:
            if "flops" in result.metrics:
                hardware_platforms.append(result.hardware)
                batch_sizes.append(result.batch_size)
                flops_values.append(result.metrics["flops"])
                model_types.append(result.metrics.get("model_type", "unknown"))
                if detailed and "detailed_flops" in result.metrics:
                    detailed_breakdowns.append(result.metrics["detailed_flops"])
                else:
                    detailed_breakdowns.append(None)
        
        if not flops_values:
            logger.warning("No FLOPs data available for plotting")
            return None
        
        # Create default output path if not provided
        if output_path is None:
            os.makedirs(benchmark_results.config.output_dir, exist_ok=True)
            safe_model_id = benchmark_results.config.model_id.replace('/', '__')
            output_path = os.path.join(
                benchmark_results.config.output_dir,
                f"{safe_model_id}_flops_comparison.png"
            )
        
        # Check if we have valid detailed breakdown
        has_detailed = detailed and None not in detailed_breakdowns
        
        # Create figure based on what data we have
        if has_detailed:
            # Create a figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1.5]})
        else:
            # Simple FLOPs plot
            fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot total FLOPs
        bar_width = 0.8 / len(set(hardware_platforms))
        x = np.arange(len(set(batch_sizes)))
        unique_hardware = sorted(set(hardware_platforms))
        unique_batch_sizes = sorted(set(batch_sizes))
        
        for i, hw in enumerate(unique_hardware):
            hw_flops = []
            for bs in unique_batch_sizes:
                # Find matching results
                matches = [
                    flops_values[j] for j in range(len(flops_values))
                    if hardware_platforms[j] == hw and batch_sizes[j] == bs
                ]
                
                # Use average if multiple matches
                if matches:
                    hw_flops.append(sum(matches) / len(matches))
                else:
                    hw_flops.append(0)
            
            ax1.bar(
                x + i * bar_width - (len(unique_hardware) - 1) * bar_width / 2,
                hw_flops,
                width=bar_width,
                label=hw.upper()
            )
        
        # Customize total FLOPs plot
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('FLOPs')
        ax1.set_title(f'FLOPs Comparison for {benchmark_results.config.model_id}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(unique_batch_sizes)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add detailed breakdown if available
        if has_detailed:
            # Find a sample with detailed breakdown to get keys
            sample_breakdown = next((b for b in detailed_breakdowns if b is not None), {})
            if sample_breakdown:
                breakdown_keys = list(sample_breakdown.keys())
                
                # Plot breakdown for each hardware platform
                for i, hw in enumerate(unique_hardware):
                    # Find the first matching result for this hardware
                    hw_indices = [j for j in range(len(hardware_platforms)) if hardware_platforms[j] == hw]
                    if not hw_indices:
                        continue
                        
                    sample_idx = hw_indices[0]
                    breakdown = detailed_breakdowns[sample_idx]
                    if not breakdown:
                        continue
                    
                    # Extract breakdown values
                    breakdown_values = [breakdown.get(key, 0) for key in breakdown_keys]
                    
                    # Create pie chart
                    pie_ax = ax2 if len(unique_hardware) == 1 else fig.add_subplot(2, len(unique_hardware), len(unique_hardware) + i + 1)
                    pie_ax.pie(breakdown_values, labels=breakdown_keys, autopct='%1.1f%%',
                            shadow=True, startangle=90)
                    pie_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    pie_ax.set_title(f'FLOPs Breakdown - {hw.upper()}')
            
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved FLOPs comparison plot to {output_path}")
        return output_path
        
    except ImportError:
        logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
        return None
    except Exception as e:
        logger.error(f"Error plotting FLOPs comparison: {e}")
        return None


def plot_power_efficiency(benchmark_results, output_path: Optional[str] = None):
    """
    Plot power efficiency metrics across hardware platforms.
    
    Args:
        benchmark_results: BenchmarkResults instance
        output_path: Path to save the plot
        
    Returns:
        Path to the saved plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data
        hardware_platforms = []
        batch_sizes = []
        power_values = []
        gflops_per_watt = []
        throughput_per_watt = []
        
        for result in benchmark_results.results:
            # Check if we have power efficiency metrics
            if "power_avg_watts" in result.metrics:
                hardware_platforms.append(result.hardware)
                batch_sizes.append(result.batch_size)
                power_values.append(result.metrics["power_avg_watts"])
                
                # Get efficiency metrics if available
                gflops_per_watt.append(result.metrics.get("gflops_per_watt", None))
                throughput_per_watt.append(result.metrics.get("throughput_per_watt", None))
        
        if not power_values:
            logger.warning("No power data available for plotting")
            return None
            
        # Check if we have efficiency metrics
        has_gflops_efficiency = None not in gflops_per_watt
        has_throughput_efficiency = None not in throughput_per_watt
            
        # Create default output path if not provided
        if output_path is None:
            os.makedirs(benchmark_results.config.output_dir, exist_ok=True)
            safe_model_id = benchmark_results.config.model_id.replace('/', '__')
            output_path = os.path.join(
                benchmark_results.config.output_dir,
                f"{safe_model_id}_power_efficiency.png"
            )
        
        # Create figure with subplots based on available metrics
        num_plots = 1 + (1 if has_gflops_efficiency else 0) + (1 if has_throughput_efficiency else 0)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
        
        # If only one plot, make sure axes is a list for consistent indexing
        if num_plots == 1:
            axes = [axes]
            
        current_ax = 0
        
        # Plot average power by hardware and batch size
        ax = axes[current_ax]
        current_ax += 1
        
        bar_width = 0.8 / len(set(hardware_platforms))
        x = np.arange(len(set(batch_sizes)))
        unique_hardware = sorted(set(hardware_platforms))
        unique_batch_sizes = sorted(set(batch_sizes))
        
        for i, hw in enumerate(unique_hardware):
            hw_power = []
            for bs in unique_batch_sizes:
                # Find matching results
                matches = [
                    power_values[j] for j in range(len(power_values))
                    if hardware_platforms[j] == hw and batch_sizes[j] == bs
                ]
                
                # Use average if multiple matches
                if matches:
                    hw_power.append(sum(matches) / len(matches))
                else:
                    hw_power.append(0)
            
            ax.bar(
                x + i * bar_width - (len(unique_hardware) - 1) * bar_width / 2,
                hw_power,
                width=bar_width,
                label=hw.upper()
            )
        
        # Customize power plot
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Power (W)')
        ax.set_title(f'Power Consumption for {benchmark_results.config.model_id}')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_batch_sizes)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot GFLOPs/Watt if available
        if has_gflops_efficiency:
            ax = axes[current_ax]
            current_ax += 1
            
            for i, hw in enumerate(unique_hardware):
                hw_efficiency = []
                for bs in unique_batch_sizes:
                    # Find matching results
                    matches = [
                        gflops_per_watt[j] for j in range(len(gflops_per_watt))
                        if hardware_platforms[j] == hw and batch_sizes[j] == bs
                    ]
                    
                    # Use average if multiple matches
                    if matches:
                        hw_efficiency.append(sum(matches) / len(matches))
                    else:
                        hw_efficiency.append(0)
                
                ax.bar(
                    x + i * bar_width - (len(unique_hardware) - 1) * bar_width / 2,
                    hw_efficiency,
                    width=bar_width,
                    label=hw.upper()
                )
            
            # Customize GFLOPs/Watt plot
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('GFLOPs/Watt')
            ax.set_title(f'Computational Efficiency for {benchmark_results.config.model_id}')
            ax.set_xticks(x)
            ax.set_xticklabels(unique_batch_sizes)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot throughput/Watt if available
        if has_throughput_efficiency:
            ax = axes[current_ax]
            
            for i, hw in enumerate(unique_hardware):
                hw_efficiency = []
                for bs in unique_batch_sizes:
                    # Find matching results
                    matches = [
                        throughput_per_watt[j] for j in range(len(throughput_per_watt))
                        if hardware_platforms[j] == hw and batch_sizes[j] == bs
                    ]
                    
                    # Use average if multiple matches
                    if matches:
                        hw_efficiency.append(sum(matches) / len(matches))
                    else:
                        hw_efficiency.append(0)
                
                ax.bar(
                    x + i * bar_width - (len(unique_hardware) - 1) * bar_width / 2,
                    hw_efficiency,
                    width=bar_width,
                    label=hw.upper()
                )
            
            # Customize throughput/Watt plot
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput/Watt (items/s/W)')
            ax.set_title(f'Throughput Efficiency for {benchmark_results.config.model_id}')
            ax.set_xticks(x)
            ax.set_xticklabels(unique_batch_sizes)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved power efficiency plot to {output_path}")
        return output_path
        
    except ImportError:
        logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
        return None
    except Exception as e:
        logger.error(f"Error plotting power efficiency: {e}")
        return None


def plot_bandwidth_utilization(benchmark_results, output_path: Optional[str] = None):
    """
    Plot memory bandwidth utilization metrics across hardware platforms.
    
    Args:
        benchmark_results: BenchmarkResults instance
        output_path: Path to save the plot
        
    Returns:
        Path to the saved plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data
        hardware_platforms = []
        batch_sizes = []
        avg_bandwidth = []
        peak_bandwidth = []
        utilization_percent = []
        arithmetic_intensity = []
        is_compute_bound = []
        
        for result in benchmark_results.results:
            # Check if we have bandwidth metrics
            if "avg_bandwidth_gbps" in result.metrics:
                hardware_platforms.append(result.hardware)
                batch_sizes.append(result.batch_size)
                avg_bandwidth.append(result.metrics["avg_bandwidth_gbps"])
                peak_bandwidth.append(result.metrics.get("peak_theoretical_bandwidth_gbps", 0))
                utilization_percent.append(result.metrics.get("bandwidth_utilization_percent", 0))
                arithmetic_intensity.append(result.metrics.get("arithmetic_intensity_flops_per_byte", 0))
                is_compute_bound.append(result.metrics.get("compute_bound", False))
        
        if not avg_bandwidth:
            logger.warning("No bandwidth data available for plotting")
            return None
            
        # Create default output path if not provided
        if output_path is None:
            os.makedirs(benchmark_results.config.output_dir, exist_ok=True)
            safe_model_id = benchmark_results.config.model_id.replace('/', '__')
            output_path = os.path.join(
                benchmark_results.config.output_dir,
                f"{safe_model_id}_bandwidth_utilization.png"
            )
        
        # Create figure with subplots for bandwidth metrics and roofline model
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot bandwidth utilization by hardware and batch size
        bar_width = 0.8 / len(set(hardware_platforms))
        x = np.arange(len(set(batch_sizes)))
        unique_hardware = sorted(set(hardware_platforms))
        unique_batch_sizes = sorted(set(batch_sizes))
        
        for i, hw in enumerate(unique_hardware):
            hw_utilization = []
            for bs in unique_batch_sizes:
                # Find matching results
                indices = [
                    j for j in range(len(utilization_percent))
                    if hardware_platforms[j] == hw and batch_sizes[j] == bs
                ]
                
                # Use average if multiple matches
                if indices:
                    hw_utilization.append(sum(utilization_percent[j] for j in indices) / len(indices))
                else:
                    hw_utilization.append(0)
            
            # Use a color gradient for the bars based on utilization percentage
            colors = plt.cm.RdYlGn(np.array(hw_utilization) / 100)
            
            ax1.bar(
                x + i * bar_width - (len(unique_hardware) - 1) * bar_width / 2,
                hw_utilization,
                width=bar_width,
                label=hw.upper(),
                color=colors
            )
        
        # Customize bandwidth utilization plot
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Bandwidth Utilization (%)')
        ax1.set_title(f'Memory Bandwidth Utilization for {benchmark_results.config.model_id}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(unique_batch_sizes)
        ax1.set_ylim(0, 100)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot the roofline model (simplified version)
        # Create x-axis for theoretical curves (log scale)
        x_intensity = np.logspace(-2, 2, 100)  # Range of arithmetic intensity values
        
        # Plot roofline for each hardware platform
        for hw in unique_hardware:
            # Get data points for this hardware
            hw_indices = [j for j in range(len(hardware_platforms)) if hardware_platforms[j] == hw]
            if not hw_indices:
                continue
                
            sample_idx = hw_indices[0]
            peak_bw = peak_bandwidth[sample_idx]
            
            # Create model data points for this hardware
            ai_values = []
            perf_values = []
            is_compute = []
            
            for j in hw_indices:
                ai_values.append(arithmetic_intensity[j])
                # Convert to TFLOPS for better visualization
                perf_values.append(arithmetic_intensity[j] * peak_bandwidth[j] * 1e9 / 1e12)
                is_compute.append(is_compute_bound[j])
            
            # Plot performance points
            ax2.scatter(
                ai_values, 
                perf_values,
                label=f"{hw.upper()} Models",
                marker='o',
                s=80,
                c=['blue' if comp else 'red' for comp in is_compute]
            )
            
            # Plot the roofline (memory bound part)
            memory_line = peak_bw * x_intensity / 1e3  # Convert to TFLOPS
            
            # Plot memory bound line
            ax2.loglog(x_intensity, memory_line, 'r--', alpha=0.7, linewidth=1)
        
        # Add labels for compute vs memory bound
        ax2.text(0.05, 0.95, "Red: Memory Bound", transform=ax2.transAxes, 
                color='red', fontsize=10, verticalalignment='top')
        ax2.text(0.05, 0.90, "Blue: Compute Bound", transform=ax2.transAxes, 
                color='blue', fontsize=10, verticalalignment='top')
        
        # Customize roofline plot
        ax2.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
        ax2.set_ylabel('Performance (TFLOPS)')
        ax2.set_title('Roofline Performance Model')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved bandwidth utilization plot to {output_path}")
        return output_path
        
    except ImportError:
        logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
        return None
    except Exception as e:
        logger.error(f"Error plotting bandwidth utilization: {e}")
        return None