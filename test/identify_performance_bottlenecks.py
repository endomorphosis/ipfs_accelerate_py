#!/usr/bin/env python3
"""
Performance Bottleneck Analyzer

This script identifies and documents performance bottlenecks in model-hardware combinations,
addressing the "Identify and document performance bottlenecks using real measurements" item
from NEXT_STEPS.md (line 167).

It analyzes benchmark data stored in DuckDB, identifies bottlenecks based on scaling behavior,
and generates a comprehensive report with optimization recommendations.

Usage:
    python identify_performance_bottlenecks.py --analyze
    python identify_performance_bottlenecks.py --format markdown --output bottlenecks.md
    python identify_performance_bottlenecks.py --model bert --hardware cuda
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import duckdb
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bottleneck_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

# Hardware descriptions
HARDWARE_DESCRIPTIONS = {
    "cpu": "CPU (Standard CPU processing)",
    "cuda": "CUDA (NVIDIA GPU acceleration)",
    "rocm": "ROCm (AMD GPU acceleration)",
    "mps": "MPS (Apple Silicon GPU acceleration)",
    "openvino": "OpenVINO (Intel acceleration)",
    "qnn": "QNN (Qualcomm AI Engine)",
    "webnn": "WebNN (Browser neural network API)",
    "webgpu": "WebGPU (Browser graphics API for ML)"
}

# Bottleneck types and descriptions
BOTTLENECK_TYPES = {
    "memory_bandwidth": {
        "name": "Memory Bandwidth",
        "description": "Model performance is limited by the rate at which data can be transferred between memory and compute units",
        "indicators": ["Poor batch scaling", "High memory access pattern sensitivity", "Increased latency with larger inputs"],
        "recommendations": [
            "Use hardware with higher memory bandwidth (HBM, GDDR6X)",
            "Optimize memory access patterns for better cache utilization",
            "Reduce precision to decrease memory requirements",
            "Apply kernel fusion to reduce memory transfers",
            "Use memory compression techniques",
        ]
    },
    "compute_bound": {
        "name": "Compute Bound",
        "description": "Model performance is limited by the computational capabilities of the hardware",
        "indicators": ["High compute unit utilization", "Good batch scaling but still slow", "Performance scales with compute capability"],
        "recommendations": [
            "Use hardware with more compute units or higher clock speeds",
            "Apply model pruning to reduce computational requirements",
            "Use specialized hardware accelerators (matrix multiplication units, tensor cores)",
            "Apply operator fusion to reduce computational overhead",
            "Consider lower precision for computation (FP16, INT8)",
        ]
    },
    "synchronization": {
        "name": "Synchronization Overhead",
        "description": "Model performance is limited by synchronization between operations or devices",
        "indicators": ["Poor scaling with parallel execution", "High launch overhead", "CPU-GPU transfer bottlenecks"],
        "recommendations": [
            "Minimize host-device synchronization points",
            "Batch operations to amortize launch overhead",
            "Use asynchronous execution when possible",
            "Apply operation fusion to reduce kernel launches",
            "Increase computation per kernel to improve efficiency",
        ]
    },
    "memory_capacity": {
        "name": "Memory Capacity",
        "description": "Model performance is limited by available memory, causing swapping or OOM errors",
        "indicators": ["OOM errors at larger batch sizes", "Significant performance drop at certain sizes", "High memory pressure"],
        "recommendations": [
            "Use hardware with more memory",
            "Apply model compression techniques (quantization, pruning)",
            "Implement gradient checkpointing for training",
            "Use model sharding or parallelism techniques",
            "Optimize memory usage with more efficient algorithms",
        ]
    },
    "io_bound": {
        "name": "I/O Bound",
        "description": "Model performance is limited by data loading or preprocessing",
        "indicators": ["CPU utilization spike during data loading", "Periods of GPU idling", "Bottleneck shifts with prefetching"],
        "recommendations": [
            "Implement data prefetching and caching",
            "Optimize data preprocessing pipeline",
            "Use memory-mapped files for large datasets",
            "Apply parallel data loading techniques",
            "Move preprocessing to GPU where possible",
        ]
    },
    "none": {
        "name": "No Significant Bottleneck",
        "description": "No clear performance bottleneck identified",
        "indicators": ["Good scaling across batch sizes", "Balanced resource utilization", "Performance meets expectations"],
        "recommendations": [
            "Continue monitoring performance as model evolves",
            "Explore advanced optimization techniques for specific use cases",
            "Consider specialized hardware for further gains if needed",
        ]
    }
}

class BottleneckAnalyzer:
    """Identify and analyze performance bottlenecks in model-hardware combinations."""
    
    def __init__(self, db_path=None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        self.conn = None
        self._connect_db()
        
    def _connect_db(self):
        """Connect to the DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            self.conn = None
    
    def _fetch_benchmark_data(self, model_filter=None, hardware_filter=None):
        """Fetch benchmark data from the database."""
        try:
            # Construct SQL query with optional filters
            query = """
                SELECT 
                    m.model_name,
                    m.model_family,
                    hp.hardware_type,
                    pr.batch_size,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb,
                    pr.inference_time_ms,
                    COALESCE(pr.test_timestamp, CURRENT_TIMESTAMP) as created_at,
                    pr.is_simulated,
                    m.modality
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                WHERE
                    1=1
            """
            
            # Add filters if provided
            if model_filter:
                model_filter_str = "','".join(model_filter)
                query += f" AND m.model_family IN ('{model_filter_str}')"
                
            if hardware_filter:
                hardware_filter_str = "','".join(hardware_filter)
                query += f" AND hp.hardware_type IN ('{hardware_filter_str}')"
                
            # Add order by clause
            query += """
                ORDER BY
                    m.model_family, hp.hardware_type, pr.batch_size, pr.test_timestamp DESC
            """
            
            # Try to execute the query
            if self.conn:
                result = self.conn.execute(query).fetchdf()
                if not result.empty:
                    logger.info(f"Found {len(result)} benchmark results in database")
                    return result
                else:
                    logger.warning("No benchmark data found with the specified filters")
            
            logger.warning("No data found in database, using sample data instead")
            return self._generate_sample_data(model_filter, hardware_filter)
            
        except Exception as e:
            logger.error(f"Failed to fetch benchmark data: {str(e)}")
            logger.warning("Using sample data instead")
            return self._generate_sample_data(model_filter, hardware_filter)
    
    def _generate_sample_data(self, model_filter=None, hardware_filter=None):
        """Generate sample benchmark data for testing."""
        logger.info("Generating sample benchmark data")
        
        # Create sample data structure
        sample_data = []
        
        # Define sample models and hardware
        models = model_filter or ["bert", "t5", "llama", "clip", "vit", "whisper", "llava"]
        hardware_types = hardware_filter or ["cpu", "cuda", "rocm", "mps", "openvino", "webgpu"]
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        # Generate sample data
        for model in models:
            # Assign a modality to each model
            if model in ["bert", "t5", "llama"]:
                modality = "text"
            elif model in ["vit", "detr"]:
                modality = "vision"
            elif model in ["whisper", "wav2vec2", "clap"]:
                modality = "audio"
            else:
                modality = "multimodal"
                
            for hardware in hardware_types:
                # Define different bottleneck patterns for hardware-model combinations
                if hardware == "cpu" and modality in ["vision", "multimodal"]:
                    # CPU is compute-bound for vision models - poor batch scaling
                    pattern = "compute_bound"
                    base_latency = 100
                    base_throughput = 10
                    base_memory = 1000
                    # Poor throughput scaling with batch size
                    scaling_factor = 1.2  # Only 1.2x throughput for 2x batch size
                elif hardware == "cuda" and modality == "text":
                    # CUDA is memory-bandwidth bound for text models
                    pattern = "memory_bandwidth"
                    base_latency = 20
                    base_throughput = 100
                    base_memory = 2000
                    # Good throughput scaling up to a point
                    scaling_factor = 1.8  # 1.8x throughput for 2x batch size
                elif hardware == "webgpu":
                    # WebGPU has synchronization overhead
                    pattern = "synchronization"
                    base_latency = 60
                    base_throughput = 30
                    base_memory = 1500
                    # Poor scaling due to overhead
                    scaling_factor = 1.3
                elif modality == "multimodal" and model == "llava":
                    # LLaVA has memory capacity issues
                    pattern = "memory_capacity"
                    base_latency = 80
                    base_throughput = 20
                    base_memory = 4000
                    # Dramatic memory increase with batch size
                    scaling_factor = 1.7
                else:
                    # Default - no major bottleneck
                    pattern = "none"
                    base_latency = 40
                    base_throughput = 50
                    base_memory = 1200
                    scaling_factor = 1.9  # Good scaling
                
                # Generate data points for different batch sizes
                for i, batch_size in enumerate(batch_sizes):
                    # Skip higher batch sizes for memory-constrained combinations
                    if pattern == "memory_capacity" and batch_size > 8:
                        continue
                        
                    # Calculate scaled metrics
                    scale = batch_size / batch_sizes[0] if i > 0 else 1.0
                    
                    # Apply scaling patterns
                    if pattern == "memory_bandwidth":
                        # Limited by memory bandwidth - throughput scaling diminishes with batch size
                        scaling_efficiency = scaling_factor * (1.0 - 0.1 * i) if i > 0 else 1.0
                        latency_scaling = scale * (1.0 + 0.2 * i)
                        memory_scaling = scale * (1.0 + 0.1 * i)
                    elif pattern == "compute_bound":
                        # Limited by compute - latency scales linearly with batch size
                        scaling_efficiency = scaling_factor
                        latency_scaling = scale * (1.0 + 0.1 * i)
                        memory_scaling = scale
                    elif pattern == "synchronization":
                        # Limited by synchronization - poor efficiency at all batch sizes
                        scaling_efficiency = scaling_factor * (1.0 - 0.05 * i) if i > 0 else 1.0
                        latency_scaling = scale * (1.0 + 0.15 * i)
                        memory_scaling = scale
                    elif pattern == "memory_capacity":
                        # Limited by memory capacity - dramatic memory increase
                        scaling_efficiency = scaling_factor * (1.0 - 0.2 * i) if i > 0 else 1.0
                        latency_scaling = scale * (1.0 + 0.3 * i)
                        memory_scaling = scale * (1.0 + 0.5 * i)
                    else:
                        # No major bottleneck - good scaling
                        scaling_efficiency = scaling_factor
                        latency_scaling = scale * (1.0 + 0.05 * i)
                        memory_scaling = scale
                    
                    # Calculate final metrics
                    latency = base_latency * latency_scaling
                    throughput = base_throughput * scale * scaling_efficiency
                    memory = base_memory * memory_scaling
                    
                    # Add random variation
                    latency *= (1.0 + 0.1 * np.random.randn())
                    throughput *= (1.0 + 0.1 * np.random.randn())
                    memory *= (1.0 + 0.05 * np.random.randn())
                    
                    # Add to sample data
                    sample_data.append({
                        'model_name': f"{model}-benchmark",
                        'model_family': model,
                        'hardware_type': hardware,
                        'batch_size': batch_size,
                        'average_latency_ms': max(1.0, latency),
                        'throughput_items_per_second': max(1.0, throughput),
                        'memory_peak_mb': max(100.0, memory),
                        'inference_time_ms': max(1.0, latency),
                        'created_at': datetime.now(),
                        'is_simulated': True,
                        'modality': modality,
                        'bottleneck_pattern': pattern  # Hidden ground truth for validation
                    })
        
        # Convert to DataFrame
        return pd.DataFrame(sample_data)
    
    def _get_latest_results_by_batch(self, df):
        """Get the latest results for each model-hardware-batch_size combination."""
        if df.empty:
            return df
            
        # Group by model, hardware, and batch size, keep latest result
        latest_results = df.sort_values('created_at', ascending=False).groupby(
            ['model_family', 'hardware_type', 'batch_size']).first().reset_index()
        
        return latest_results
    
    def _analyze_batch_scaling(self, df):
        """Analyze batch scaling behavior to identify bottlenecks."""
        if df.empty:
            return pd.DataFrame()
            
        # Get latest results by batch size
        latest_results = self._get_latest_results_by_batch(df)
        
        # Calculate bottleneck indicators
        bottlenecks = []
        
        # Group by model and hardware
        for (model, hardware), group in latest_results.groupby(['model_family', 'hardware_type']):
            # Need at least two batch sizes to analyze scaling
            if len(group) < 2:
                continue
                
            # Sort by batch size
            group = group.sort_values('batch_size')
            
            # Get batch sizes and metrics
            batch_sizes = group['batch_size'].tolist()
            latencies = group['average_latency_ms'].tolist()
            throughputs = group['throughput_items_per_second'].tolist()
            memories = group['memory_peak_mb'].tolist()
            
            # Calculate scaling metrics
            # 1. Throughput scaling efficiency (should be ~linear)
            if len(batch_sizes) >= 2 and batch_sizes[0] > 0:
                # Compare largest batch size to smallest
                throughput_scaling = (throughputs[-1] / throughputs[0]) / (batch_sizes[-1] / batch_sizes[0])
                
                # Calculate scaling efficiency at each step
                scaling_efficiencies = []
                for i in range(1, len(batch_sizes)):
                    if throughputs[i-1] > 0 and batch_sizes[i] > batch_sizes[i-1]:
                        step_scaling = (throughputs[i] / throughputs[i-1]) / (batch_sizes[i] / batch_sizes[i-1])
                        scaling_efficiencies.append(step_scaling)
                
                # Average step scaling efficiency
                avg_step_scaling = np.mean(scaling_efficiencies) if scaling_efficiencies else 0.0
                
                # Check for declining scaling efficiency
                declining_efficiency = False
                if len(scaling_efficiencies) >= 2:
                    declining_efficiency = scaling_efficiencies[-1] < scaling_efficiencies[0] * 0.8
            else:
                throughput_scaling = 0.0
                avg_step_scaling = 0.0
                declining_efficiency = False
                
            # 2. Memory scaling
            if len(batch_sizes) >= 2 and batch_sizes[0] > 0:
                memory_scaling = (memories[-1] / memories[0]) / (batch_sizes[-1] / batch_sizes[0])
            else:
                memory_scaling = 1.0
                
            # 3. Memory pressure detection
            memory_increase_ratio = group['memory_peak_mb'].max() / max(1.0, group['memory_peak_mb'].min())
            if memory_increase_ratio > 2.0:
                memory_pressure = "high"
            elif memory_increase_ratio > 1.5:
                memory_pressure = "medium"
            else:
                memory_pressure = "low"
                
            # Determine primary bottleneck
            # First, check for memory capacity issues
            if memory_pressure == "high" and memory_scaling > 1.5:
                primary_bottleneck = "memory_capacity"
                bottleneck_confidence = 0.9
            # Check for memory bandwidth bottlenecks
            elif throughput_scaling < 0.6 and declining_efficiency:
                primary_bottleneck = "memory_bandwidth"
                bottleneck_confidence = 0.8
            # Check for synchronization overhead
            elif avg_step_scaling < 0.7 and not declining_efficiency:
                primary_bottleneck = "synchronization"
                bottleneck_confidence = 0.7
            # Check for compute bound
            elif throughput_scaling < 0.8 and avg_step_scaling < 0.8:
                primary_bottleneck = "compute_bound"
                bottleneck_confidence = 0.75
            # No clear bottleneck
            else:
                primary_bottleneck = "none"
                bottleneck_confidence = 0.6
            
            # Get modality if available
            modality = group['modality'].iloc[0] if 'modality' in group.columns else "unknown"
            
            # Get ground truth if available (from sample data)
            ground_truth = group['bottleneck_pattern'].iloc[0] if 'bottleneck_pattern' in group.columns else None
            
            # Secondary bottlenecks
            secondary_bottlenecks = []
            if primary_bottleneck != "memory_capacity" and memory_pressure == "medium":
                secondary_bottlenecks.append("memory_capacity")
            if primary_bottleneck != "memory_bandwidth" and throughput_scaling < 0.7:
                secondary_bottlenecks.append("memory_bandwidth")
            if primary_bottleneck != "synchronization" and avg_step_scaling < 0.8:
                secondary_bottlenecks.append("synchronization")
                
            # Add to bottlenecks list
            bottlenecks.append({
                'model_family': model,
                'hardware_type': hardware,
                'modality': modality,
                'primary_bottleneck': primary_bottleneck,
                'bottleneck_confidence': bottleneck_confidence,
                'secondary_bottlenecks': secondary_bottlenecks,
                'throughput_scaling': throughput_scaling,
                'avg_step_scaling': avg_step_scaling,
                'memory_scaling': memory_scaling,
                'memory_pressure': memory_pressure,
                'memory_increase_ratio': memory_increase_ratio,
                'min_batch_size': min(batch_sizes),
                'max_batch_size': max(batch_sizes),
                'max_throughput': max(throughputs),
                'min_latency_ms': min(latencies),
                'max_memory_mb': max(memories),
                'num_batch_sizes': len(batch_sizes),
                'ground_truth': ground_truth
            })
        
        # Convert to DataFrame
        bottlenecks_df = pd.DataFrame(bottlenecks)
        
        return bottlenecks_df
    
    def _generate_optimization_recommendations(self, bottlenecks_df):
        """Generate specific optimization recommendations based on bottleneck analysis."""
        if bottlenecks_df.empty:
            return pd.DataFrame()
            
        # Add recommendations
        recommendations = []
        
        for _, row in bottlenecks_df.iterrows():
            model = row['model_family']
            hardware = row['hardware_type']
            bottleneck = row['primary_bottleneck']
            modality = row['modality']
            
            # Get bottleneck info
            bottleneck_info = BOTTLENECK_TYPES.get(bottleneck, BOTTLENECK_TYPES["none"])
            
            # Select relevant recommendations
            primary_recs = bottleneck_info["recommendations"][:3]  # Top 3 recommendations
            
            # Add model-specific recommendations
            model_specific_recs = []
            
            # For text models
            if modality == "text":
                if bottleneck == "memory_bandwidth":
                    model_specific_recs.append("Implement attention optimizations like FlashAttention")
                elif bottleneck == "memory_capacity":
                    model_specific_recs.append("Consider layer-wise model sharding or activation checkpointing")
                    
            # For vision models
            elif modality == "vision":
                if bottleneck == "compute_bound":
                    model_specific_recs.append("Use optimized convolution implementations")
                elif bottleneck == "memory_bandwidth":
                    model_specific_recs.append("Apply patch merging optimizations")
                    
            # For audio models
            elif modality == "audio":
                if bottleneck == "compute_bound":
                    model_specific_recs.append("Consider time-domain processing optimizations")
                elif hardware == "webgpu":
                    model_specific_recs.append("Use compute shader optimization for improved WebGPU audio performance")
                    
            # For multimodal models
            elif modality == "multimodal":
                if bottleneck == "memory_capacity":
                    model_specific_recs.append("Use model pruning techniques on non-critical components")
                elif hardware.startswith("web"):
                    model_specific_recs.append("Enable parallel loading optimization for faster initialization")
            
            # Hardware-specific recommendations
            hardware_specific_recs = []
            
            if hardware == "cpu":
                if bottleneck == "compute_bound":
                    hardware_specific_recs.append("Ensure SIMD vectorization is enabled")
                    hardware_specific_recs.append("Consider using OpenVINO for better CPU performance")
            elif hardware == "cuda":
                if bottleneck == "memory_bandwidth":
                    hardware_specific_recs.append("Use Tensor Cores with mixed precision where available")
                    hardware_specific_recs.append("Enable CUDA Graph to reduce kernel launch overhead")
            elif hardware == "webgpu":
                if bottleneck in ["synchronization", "memory_bandwidth"]:
                    hardware_specific_recs.append("Enable shader precompilation for faster initialization")
                    hardware_specific_recs.append("Use compute shaders for intensive operations")
            
            # Combine recommendations
            all_recs = primary_recs + model_specific_recs + hardware_specific_recs
            
            # Add to recommendations
            recommendations.append({
                'model_family': model,
                'hardware_type': hardware,
                'bottleneck_type': bottleneck,
                'bottleneck_name': bottleneck_info["name"],
                'general_recommendations': primary_recs,
                'model_specific_recommendations': model_specific_recs,
                'hardware_specific_recommendations': hardware_specific_recs,
                'all_recommendations': all_recs
            })
        
        # Convert to DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        return recommendations_df
    
    def analyze_bottlenecks(self, model_filter=None, hardware_filter=None, output_format="html", output_path=None):
        """Analyze performance bottlenecks and generate a report."""
        logger.info("Analyzing performance bottlenecks...")
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"bottleneck_analysis_{timestamp}.{output_format}"
        
        # Fetch benchmark data
        data = self._fetch_benchmark_data(model_filter, hardware_filter)
        
        if data.empty:
            logger.error("No benchmark data available for analysis")
            return None
            
        # Analyze batch scaling to identify bottlenecks
        bottlenecks_df = self._analyze_batch_scaling(data)
        
        if bottlenecks_df.empty:
            logger.error("Insufficient data for bottleneck analysis")
            return None
            
        # Generate optimization recommendations
        recommendations_df = self._generate_optimization_recommendations(bottlenecks_df)
        
        # Generate report based on format
        if output_format == "html":
            self._generate_html_report(bottlenecks_df, recommendations_df, data, output_path)
        elif output_format in ["md", "markdown"]:
            self._generate_markdown_report(bottlenecks_df, recommendations_df, output_path)
        elif output_format == "json":
            self._generate_json_report(bottlenecks_df, recommendations_df, output_path)
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return None
            
        logger.info(f"Bottleneck analysis report generated: {output_path}")
        return output_path
    
    def _generate_html_report(self, bottlenecks_df, recommendations_df, raw_data, output_path):
        """Generate an HTML bottleneck analysis report."""
        try:
            # Check for simulation data
            using_simulated_data = 'is_simulated' in raw_data.columns and raw_data['is_simulated'].any()
            
            with open(output_path, 'w') as f:
                # Start HTML document
                f.write(f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Performance Bottleneck Analysis</title>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; color: #333; line-height: 1.6; }}
                        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                        h1, h2, h3, h4 {{ color: #1a5276; }}
                        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }}
                        th, td {{ border: 1px solid #ddd; padding: 12px 15px; text-align: left; }}
                        th {{ background-color: #3498db; color: white; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        tr:hover {{ background-color: #f1f1f1; }}
                        .summary-card {{ background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin-bottom: 20px; }}
                        .warning {{ background-color: #fcf8e3; border-left: 4px solid #f0ad4e; padding: 15px; margin-bottom: 20px; }}
                        .bottleneck-card {{ background-color: #f1f9f7; border-left: 4px solid #2ecc71; padding: 15px; margin-bottom: 20px; }}
                        .recommendation-list {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                        .note {{ font-style: italic; margin-top: 5px; color: #666; }}
                        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
                        .badge-high {{ background-color: #d9534f; color: white; }}
                        .badge-medium {{ background-color: #f0ad4e; color: white; }}
                        .badge-low {{ background-color: #5bc0de; color: white; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Performance Bottleneck Analysis</h1>
                        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                """)
                
                # Add summary section
                f.write(f"""
                        <div class="summary-card">
                            <h2>Executive Summary</h2>
                            <p>This report identifies performance bottlenecks in {len(bottlenecks_df)} model-hardware combinations based on benchmark data.</p>
                            <p>Each bottleneck is analyzed with scaling behavior metrics and includes optimization recommendations.</p>
                        </div>
                """)
                
                # Add simulation warning if needed
                if using_simulated_data:
                    f.write(f"""
                        <div class="warning">
                            <h3>⚠️ Simulation Notice</h3>
                            <p>This analysis includes simulated benchmark data. Real hardware performance may exhibit different bottlenecks.</p>
                        </div>
                    """)
                
                # Add bottleneck overview
                f.write(f"""
                        <h2>Bottleneck Overview</h2>
                        <table>
                            <tr>
                                <th>Model</th>
                                <th>Hardware</th>
                                <th>Primary Bottleneck</th>
                                <th>Confidence</th>
                                <th>Throughput Scaling</th>
                                <th>Memory Pressure</th>
                                <th>Batch Size Range</th>
                            </tr>
                """)
                
                # Add rows for bottleneck overview
                for _, row in bottlenecks_df.iterrows():
                    model = row['model_family']
                    hardware = row['hardware_type']
                    bottleneck = row['primary_bottleneck']
                    confidence = row['bottleneck_confidence']
                    throughput_scaling = row['throughput_scaling']
                    memory_pressure = row['memory_pressure']
                    min_batch = row['min_batch_size']
                    max_batch = row['max_batch_size']
                    
                    # Get bottleneck name
                    bottleneck_name = BOTTLENECK_TYPES.get(bottleneck, BOTTLENECK_TYPES["none"])["name"]
                    
                    # Determine confidence class
                    if confidence >= 0.8:
                        confidence_class = "badge-high"
                    elif confidence >= 0.6:
                        confidence_class = "badge-medium"
                    else:
                        confidence_class = "badge-low"
                    
                    # Determine memory pressure class
                    if memory_pressure == "high":
                        memory_class = "badge-high"
                    elif memory_pressure == "medium":
                        memory_class = "badge-medium"
                    else:
                        memory_class = "badge-low"
                    
                    f.write(f"""
                        <tr>
                            <td>{model}</td>
                            <td>{hardware}</td>
                            <td>{bottleneck_name}</td>
                            <td><span class="badge {confidence_class}">{confidence:.2f}</span></td>
                            <td>{throughput_scaling:.2f}</td>
                            <td><span class="badge {memory_class}">{memory_pressure}</span></td>
                            <td>{min_batch} - {max_batch}</td>
                        </tr>
                    """)
                
                f.write("</table>\n")
                
                # Add detailed bottleneck analysis
                f.write("<h2>Detailed Bottleneck Analysis</h2>\n")
                
                # Group by bottleneck type for organized presentation
                for bottleneck_id, bottleneck_info in BOTTLENECK_TYPES.items():
                    # Filter to this bottleneck type
                    filtered_df = bottlenecks_df[bottlenecks_df['primary_bottleneck'] == bottleneck_id]
                    
                    if len(filtered_df) == 0:
                        continue
                    
                    f.write(f"""
                        <div class="bottleneck-card">
                            <h3>{bottleneck_info['name']} ({len(filtered_df)} models)</h3>
                            <p>{bottleneck_info['description']}</p>
                            
                            <h4>Indicators:</h4>
                            <ul>
                    """)
                    
                    for indicator in bottleneck_info['indicators']:
                        f.write(f"<li>{indicator}</li>\n")
                    
                    f.write("</ul>\n")
                    
                    # Add affected models
                    f.write(f"""
                            <h4>Affected Model-Hardware Combinations:</h4>
                            <table>
                                <tr>
                                    <th>Model</th>
                                    <th>Hardware</th>
                                    <th>Modality</th>
                                    <th>Throughput Scaling</th>
                                    <th>Memory Pressure</th>
                                    <th>Max Throughput</th>
                                    <th>Min Latency (ms)</th>
                                </tr>
                    """)
                    
                    for _, row in filtered_df.iterrows():
                        model = row['model_family']
                        hardware = row['hardware_type']
                        modality = row['modality']
                        throughput_scaling = row['throughput_scaling']
                        memory_pressure = row['memory_pressure']
                        max_throughput = row['max_throughput']
                        min_latency = row['min_latency_ms']
                        
                        f.write(f"""
                            <tr>
                                <td>{model}</td>
                                <td>{hardware}</td>
                                <td>{modality}</td>
                                <td>{throughput_scaling:.2f}</td>
                                <td>{memory_pressure}</td>
                                <td>{max_throughput:.2f}</td>
                                <td>{min_latency:.2f}</td>
                            </tr>
                        """)
                    
                    f.write("</table>\n")
                    
                    # Add general recommendations
                    f.write(f"""
                            <h4>General Recommendations:</h4>
                            <div class="recommendation-list">
                                <ol>
                    """)
                    
                    for rec in bottleneck_info['recommendations']:
                        f.write(f"<li>{rec}</li>\n")
                    
                    f.write("""
                                </ol>
                            </div>
                        </div>
                    """)
                
                # Add model-specific recommendations
                f.write("<h2>Model-Specific Recommendations</h2>\n")
                
                # Add each model-hardware combination
                for _, row in recommendations_df.iterrows():
                    model = row['model_family']
                    hardware = row['hardware_type']
                    bottleneck_name = row['bottleneck_name']
                    general_recs = row['general_recommendations']
                    model_recs = row['model_specific_recommendations']
                    hardware_recs = row['hardware_specific_recommendations']
                    
                    f.write(f"""
                        <h3>{model} on {hardware}</h3>
                        <p><strong>Primary Bottleneck:</strong> {bottleneck_name}</p>
                        
                        <h4>Recommended Optimizations:</h4>
                        <ol>
                    """)
                    
                    # Add all recommendations
                    for rec in general_recs:
                        f.write(f"<li>{rec}</li>\n")
                    
                    f.write("</ol>\n")
                    
                    # Add model-specific recommendations if any
                    if model_recs:
                        f.write(f"""
                            <h4>Model-Specific Optimizations:</h4>
                            <ul>
                        """)
                        
                        for rec in model_recs:
                            f.write(f"<li>{rec}</li>\n")
                        
                        f.write("</ul>\n")
                    
                    # Add hardware-specific recommendations if any
                    if hardware_recs:
                        f.write(f"""
                            <h4>Hardware-Specific Optimizations:</h4>
                            <ul>
                        """)
                        
                        for rec in hardware_recs:
                            f.write(f"<li>{rec}</li>\n")
                        
                        f.write("</ul>\n")
                
                # Add methodology
                f.write(f"""
                    <h2>Methodology</h2>
                    <p>This bottleneck analysis was performed using the following methodology:</p>
                    <ul>
                        <li>Throughput scaling efficiency analysis across batch sizes</li>
                        <li>Memory usage analysis to detect memory pressure and capacity issues</li>
                        <li>Step scaling analysis to identify synchronization overhead</li>
                        <li>Bottleneck classification based on scaling behavior patterns</li>
                        <li>Confidence scoring based on the strength of detected bottleneck signals</li>
                    </ul>
                    
                    <p class="note">Note: Bottleneck identification is based on observed scaling behavior and may not capture all performance factors.</p>
                </body>
                </html>
                """)
            
            logger.info(f"HTML bottleneck analysis report generated: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return False
    
    def _generate_markdown_report(self, bottlenecks_df, recommendations_df, output_path):
        """Generate a markdown bottleneck analysis report."""
        try:
            with open(output_path, 'w') as f:
                # Header
                f.write("# Performance Bottleneck Analysis\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Executive summary
                f.write("## Executive Summary\n\n")
                f.write(f"This report identifies performance bottlenecks in {len(bottlenecks_df)} model-hardware combinations based on benchmark data.\n")
                f.write("Each bottleneck is analyzed with scaling behavior metrics and includes optimization recommendations.\n\n")
                
                # Check for simulation data
                if 'ground_truth' in bottlenecks_df.columns:
                    f.write("⚠️ **Simulation Notice**: This analysis includes simulated benchmark data. Real hardware performance may exhibit different bottlenecks.\n\n")
                
                # Bottleneck overview
                f.write("## Bottleneck Overview\n\n")
                f.write("| Model | Hardware | Primary Bottleneck | Confidence | Throughput Scaling | Memory Pressure | Batch Size Range |\n")
                f.write("|-------|----------|-------------------|------------|-------------------|----------------|------------------|\n")
                
                for _, row in bottlenecks_df.iterrows():
                    model = row['model_family']
                    hardware = row['hardware_type']
                    bottleneck = row['primary_bottleneck']
                    confidence = row['bottleneck_confidence']
                    throughput_scaling = row['throughput_scaling']
                    memory_pressure = row['memory_pressure']
                    min_batch = row['min_batch_size']
                    max_batch = row['max_batch_size']
                    
                    # Get bottleneck name
                    bottleneck_name = BOTTLENECK_TYPES.get(bottleneck, BOTTLENECK_TYPES["none"])["name"]
                    
                    f.write(f"| {model} | {hardware} | {bottleneck_name} | {confidence:.2f} | {throughput_scaling:.2f} | {memory_pressure} | {min_batch} - {max_batch} |\n")
                
                f.write("\n")
                
                # Detailed bottleneck analysis
                f.write("## Detailed Bottleneck Analysis\n\n")
                
                # Group by bottleneck type
                for bottleneck_id, bottleneck_info in BOTTLENECK_TYPES.items():
                    # Filter to this bottleneck type
                    filtered_df = bottlenecks_df[bottlenecks_df['primary_bottleneck'] == bottleneck_id]
                    
                    if len(filtered_df) == 0:
                        continue
                    
                    f.write(f"### {bottleneck_info['name']} ({len(filtered_df)} models)\n\n")
                    f.write(f"{bottleneck_info['description']}\n\n")
                    
                    f.write("**Indicators:**\n\n")
                    for indicator in bottleneck_info['indicators']:
                        f.write(f"- {indicator}\n")
                    
                    f.write("\n**Affected Model-Hardware Combinations:**\n\n")
                    f.write("| Model | Hardware | Modality | Throughput Scaling | Memory Pressure | Max Throughput | Min Latency (ms) |\n")
                    f.write("|-------|----------|----------|-------------------|----------------|---------------|------------------|\n")
                    
                    for _, row in filtered_df.iterrows():
                        model = row['model_family']
                        hardware = row['hardware_type']
                        modality = row['modality']
                        throughput_scaling = row['throughput_scaling']
                        memory_pressure = row['memory_pressure']
                        max_throughput = row['max_throughput']
                        min_latency = row['min_latency_ms']
                        
                        f.write(f"| {model} | {hardware} | {modality} | {throughput_scaling:.2f} | {memory_pressure} | {max_throughput:.2f} | {min_latency:.2f} |\n")
                    
                    f.write("\n**General Recommendations:**\n\n")
                    for i, rec in enumerate(bottleneck_info['recommendations']):
                        f.write(f"{i+1}. {rec}\n")
                    
                    f.write("\n")
                
                # Model-specific recommendations
                f.write("## Model-Specific Recommendations\n\n")
                
                for _, row in recommendations_df.iterrows():
                    model = row['model_family']
                    hardware = row['hardware_type']
                    bottleneck_name = row['bottleneck_name']
                    general_recs = row['general_recommendations']
                    model_recs = row['model_specific_recommendations']
                    hardware_recs = row['hardware_specific_recommendations']
                    
                    f.write(f"### {model} on {hardware}\n\n")
                    f.write(f"**Primary Bottleneck:** {bottleneck_name}\n\n")
                    
                    f.write("**Recommended Optimizations:**\n\n")
                    for i, rec in enumerate(general_recs):
                        f.write(f"{i+1}. {rec}\n")
                    
                    f.write("\n")
                    
                    # Add model-specific recommendations if any
                    if model_recs:
                        f.write("**Model-Specific Optimizations:**\n\n")
                        for rec in model_recs:
                            f.write(f"- {rec}\n")
                        f.write("\n")
                    
                    # Add hardware-specific recommendations if any
                    if hardware_recs:
                        f.write("**Hardware-Specific Optimizations:**\n\n")
                        for rec in hardware_recs:
                            f.write(f"- {rec}\n")
                        f.write("\n")
                
                # Methodology
                f.write("## Methodology\n\n")
                f.write("This bottleneck analysis was performed using the following methodology:\n\n")
                f.write("- Throughput scaling efficiency analysis across batch sizes\n")
                f.write("- Memory usage analysis to detect memory pressure and capacity issues\n")
                f.write("- Step scaling analysis to identify synchronization overhead\n")
                f.write("- Bottleneck classification based on scaling behavior patterns\n")
                f.write("- Confidence scoring based on the strength of detected bottleneck signals\n\n")
                
                f.write("*Note: Bottleneck identification is based on observed scaling behavior and may not capture all performance factors.*\n")
            
            logger.info(f"Markdown bottleneck analysis report generated: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate Markdown report: {e}")
            return False
    
    def _generate_json_report(self, bottlenecks_df, recommendations_df, output_path):
        """Generate a JSON bottleneck analysis report."""
        try:
            # Create result dictionary
            result = {
                "generated_at": datetime.now().isoformat(),
                "report_type": "bottleneck_analysis",
                "bottleneck_types": BOTTLENECK_TYPES,
                "hardware_descriptions": HARDWARE_DESCRIPTIONS,
                "bottlenecks": [],
                "recommendations": []
            }
            
            # Convert bottlenecks DataFrame to list of dictionaries
            for _, row in bottlenecks_df.iterrows():
                bottleneck_dict = {
                    "model_family": row["model_family"],
                    "hardware_type": row["hardware_type"],
                    "modality": row["modality"],
                    "primary_bottleneck": row["primary_bottleneck"],
                    "bottleneck_confidence": float(row["bottleneck_confidence"]),
                    "secondary_bottlenecks": row["secondary_bottlenecks"],
                    "throughput_scaling": float(row["throughput_scaling"]),
                    "avg_step_scaling": float(row["avg_step_scaling"]),
                    "memory_scaling": float(row["memory_scaling"]),
                    "memory_pressure": row["memory_pressure"],
                    "memory_increase_ratio": float(row["memory_increase_ratio"]),
                    "min_batch_size": int(row["min_batch_size"]),
                    "max_batch_size": int(row["max_batch_size"]),
                    "max_throughput": float(row["max_throughput"]),
                    "min_latency_ms": float(row["min_latency_ms"]),
                    "max_memory_mb": float(row["max_memory_mb"]),
                    "num_batch_sizes": int(row["num_batch_sizes"])
                }
                
                # Add ground truth if available
                if "ground_truth" in row and row["ground_truth"] is not None:
                    bottleneck_dict["ground_truth"] = row["ground_truth"]
                
                result["bottlenecks"].append(bottleneck_dict)
            
            # Convert recommendations DataFrame to list of dictionaries
            for _, row in recommendations_df.iterrows():
                result["recommendations"].append({
                    "model_family": row["model_family"],
                    "hardware_type": row["hardware_type"],
                    "bottleneck_type": row["bottleneck_type"],
                    "bottleneck_name": row["bottleneck_name"],
                    "general_recommendations": row["general_recommendations"],
                    "model_specific_recommendations": row["model_specific_recommendations"],
                    "hardware_specific_recommendations": row["hardware_specific_recommendations"]
                })
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"JSON bottleneck analysis report generated: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            return False

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Performance Bottleneck Analyzer")
    
    # Main command groups
    parser.add_argument("--analyze", action="store_true", help="Analyze performance bottlenecks")
    
    # Filtering options
    parser.add_argument("--model", action="append", help="Filter by model family (can specify multiple)")
    parser.add_argument("--hardware", action="append", help="Filter by hardware type (can specify multiple)")
    
    # Configuration options
    parser.add_argument("--db-path", help="Path to benchmark database")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--format", choices=["html", "md", "markdown", "json"], default="html", help="Output format")
    
    args = parser.parse_args()
    
    # Create bottleneck analyzer
    analyzer = BottleneckAnalyzer(db_path=args.db_path)
    
    if args.analyze:
        analyzer.analyze_bottlenecks(
            model_filter=args.model,
            hardware_filter=args.hardware,
            output_format=args.format,
            output_path=args.output
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()