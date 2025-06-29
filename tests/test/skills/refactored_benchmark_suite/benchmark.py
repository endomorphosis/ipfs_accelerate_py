#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main benchmark orchestration module for the refactored HuggingFace model benchmark suite.

This module provides the core functionality for running benchmarks on HuggingFace models
across different hardware platforms, collecting various performance metrics, and managing
the benchmark lifecycle.
"""

import os
import json
import time
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

# Handle optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Local imports
from utils.logging import setup_logger
from metrics import (
    LatencyMetric, 
    ThroughputMetric, 
    MemoryMetric, 
    FLOPsMetric,
    PowerMetric,
    BandwidthMetric,
    get_available_metrics
)
from hardware import (
    get_available_hardware,
    get_hardware_info,
    initialize_hardware
)

# Configure logger
logger = setup_logger("benchmark")

@dataclass
class BenchmarkConfig:
    """Configuration for a single model benchmark."""
    
    model_id: str
    task: Optional[str] = None
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    sequence_lengths: List[int] = field(default_factory=lambda: [16, 32, 64])
    hardware: List[str] = field(default_factory=lambda: ["cpu"])
    metrics: List[str] = field(default_factory=lambda: ["latency", "throughput", "memory"])
    warmup_iterations: int = 5
    test_iterations: int = 20
    save_results: bool = True
    output_dir: str = "benchmark_results"
    
    # Hardware optimization options
    use_4bit: bool = False  # Use 4-bit quantization for large models (requires bitsandbytes)
    use_8bit: bool = False  # Use 8-bit quantization for large models (requires bitsandbytes)
    flash_attention: bool = False  # Use Flash Attention for transformer models if available
    torch_compile: bool = False  # Use torch.compile for PyTorch 2.0+ optimizations
    
    def __post_init__(self):
        # Validate hardware platforms
        available_hw = get_available_hardware()
        for hw in self.hardware:
            if hw not in available_hw:
                logger.warning(f"Hardware '{hw}' is not available on this system. Available: {available_hw}")
                self.hardware.remove(hw)
        
        if not self.hardware:
            logger.warning("No valid hardware platforms specified. Defaulting to CPU.")
            self.hardware = ["cpu"]
        
        # Validate metrics
        available_metrics = get_available_metrics()
        for metric in self.metrics:
            if metric not in available_metrics:
                logger.warning(f"Metric '{metric}' is not available. Available: {available_metrics}")
                self.metrics.remove(metric)
                
        if not self.metrics:
            logger.warning("No valid metrics specified. Defaulting to latency.")
            self.metrics = ["latency"]
        
        # Create output directory if it doesn't exist
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """Create a BenchmarkConfig from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "BenchmarkConfig":
        """Load a BenchmarkConfig from a JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the BenchmarkConfig to a dictionary."""
        return asdict(self)
    
    def to_json(self, json_path: str) -> None:
        """Save the BenchmarkConfig to a JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    model_id: str
    hardware: str
    batch_size: int
    sequence_length: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    input_shape: Dict[str, List[int]] = field(default_factory=dict)
    output_shape: Dict[str, List[int]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the BenchmarkResult to a dictionary."""
        return asdict(self)


class BenchmarkResults:
    """Collection of benchmark results with export and visualization capabilities."""
    
    def __init__(self, results: List[BenchmarkResult], config: BenchmarkConfig):
        self.results = results
        self.config = config
        self.timestamp = datetime.now().isoformat()
    
    def export_to_json(self, output_path: Optional[str] = None) -> str:
        """Export results to a JSON file."""
        if output_path is None:
            # Create default filename if none provided
            safe_model_id = self.config.model_id.replace('/', '__')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.config.output_dir, 
                f"benchmark_{safe_model_id}_{timestamp}.json"
            )
        
        # Create output dictionary
        output = {
            "timestamp": self.timestamp,
            "model": self.config.model_id,
            "hardware_tested": list(set(result.hardware for result in self.results)),
            "batch_sizes": list(set(result.batch_size for result in self.results)),
            "available_hardware": get_hardware_info(),
            "gpu_theoretical_tflops": self._get_gpu_theoretical_tflops(),
            "hardware_efficiency": self._get_hardware_efficiency_metrics(),
            "results": {}
        }
        
        # Organize results by hardware
        for hw in output["hardware_tested"]:
            hw_results = [r for r in self.results if r.hardware == hw]
            if not hw_results:
                continue
                
            output["results"][hw] = {
                "model_name": self.config.model_id,
                "hardware": hw,
                "device": hw,  # For compatibility with existing tools
                "results": [r.to_dict() for r in hw_results]
            }
        
        # Write to file
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Exported benchmark results to {output_path}")
        return output_path
    
    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """Export results to a CSV file."""
        try:
            import pandas as pd
            
            # Convert results to DataFrame
            rows = []
            for result in self.results:
                row = {
                    "model_id": result.model_id,
                    "hardware": result.hardware,
                    "batch_size": result.batch_size,
                    "sequence_length": result.sequence_length,
                    "timestamp": result.timestamp
                }
                
                # Add metrics
                for metric_name, metric_value in result.metrics.items():
                    row[metric_name] = metric_value
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Create default filename if none provided
            if output_path is None:
                safe_model_id = self.config.model_id.replace('/', '__')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(
                    self.config.output_dir, 
                    f"benchmark_{safe_model_id}_{timestamp}.csv"
                )
            
            # Write to file
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported benchmark results to {output_path}")
            return output_path
            
        except ImportError:
            logger.error("pandas is required for CSV export. Install with 'pip install pandas'")
            return ""
    
    def export_to_markdown(self, output_path: Optional[str] = None) -> str:
        """Export results to a Markdown report."""
        if output_path is None:
            # Create default filename if none provided
            safe_model_id = self.config.model_id.replace('/', '__')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.config.output_dir, 
                f"benchmark_{safe_model_id}_{timestamp}.md"
            )
        
        # Create markdown content
        markdown = f"# Benchmark Results: {self.config.model_id}\n\n"
        markdown += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Hardware information
        markdown += "## Hardware Information\n\n"
        for hw, hw_info in get_hardware_info().items():
            if hw in self.config.hardware:
                markdown += f"- **{hw.upper()}**: {hw_info}\n"
        markdown += "\n"
        
        # Results by hardware
        for hw in self.config.hardware:
            hw_results = [r for r in self.results if r.hardware == hw]
            if not hw_results:
                continue
                
            markdown += f"## Results for {hw.upper()}\n\n"
            
            # Create table headers
            metrics = []
            for result in hw_results:
                metrics.extend(result.metrics.keys())
            metrics = sorted(set(metrics))
            
            headers = ["Batch Size", "Sequence Length"] + [m.capitalize() for m in metrics]
            markdown += "| " + " | ".join(headers) + " |\n"
            markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            # Group results by batch size and sequence length
            results_by_batch_seq = {}
            for result in hw_results:
                key = (result.batch_size, result.sequence_length)
                results_by_batch_seq[key] = result
            
            # Add rows
            for key in sorted(results_by_batch_seq.keys()):
                batch_size, seq_len = key
                result = results_by_batch_seq[key]
                
                row = [str(batch_size), str(seq_len)]
                for metric in metrics:
                    value = result.metrics.get(metric, "N/A")
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                
                markdown += "| " + " | ".join(row) + " |\n"
            
            markdown += "\n"
        
        # Add CPU vs GPU speedup if available
        speedup = self.get_cpu_gpu_speedup()
        if speedup is not None:
            markdown += f"## Performance Insights\n\n"
            markdown += f"**CPU to GPU Speedup: {speedup:.1f}x**\n\n"
            markdown += f"The model runs {speedup:.1f} times faster on GPU compared to CPU under the same conditions.\n\n"
        
        # Add input/output shapes
        markdown += "## Model Information\n\n"
        
        # Get a sample result for shapes
        sample_result = self.results[0] if self.results else None
        if sample_result and sample_result.input_shape:
            markdown += "### Input Shapes\n\n"
            markdown += "```\n"
            for key, shape in sample_result.input_shape.items():
                markdown += f"{key}: {shape}\n"
            markdown += "```\n\n"
        
        if sample_result and sample_result.output_shape:
            markdown += "### Output Shapes\n\n"
            markdown += "```\n"
            for key, shape in sample_result.output_shape.items():
                markdown += f"{key}: {shape}\n"
            markdown += "```\n\n"
        
        # Write to file
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(markdown)
        
        logger.info(f"Exported benchmark results to {output_path}")
        return output_path
    
    def publish_to_hub(self, token: Optional[str] = None) -> bool:
        """Publish results to HuggingFace Model Hub."""
        try:
            from exporters.hf_hub_exporter import ModelCardExporter
            
            exporter = ModelCardExporter(token=token)
            success = exporter.publish_results(self)
            
            if success:
                logger.info(f"Successfully published benchmark results to HuggingFace Hub for {self.config.model_id}")
            else:
                logger.error(f"Failed to publish benchmark results to HuggingFace Hub for {self.config.model_id}")
            
            return success
            
        except ImportError:
            logger.error("huggingface_hub is required for Hub export. Install with 'pip install huggingface_hub'")
            return False
    
    def plot_latency_comparison(self, output_path: Optional[str] = None, include_percentiles: bool = True):
        """
        Plot latency comparison across hardware platforms.
        
        Args:
            output_path: Path to save the plot
            include_percentiles: Whether to include p90, p95, p99 percentiles
        
        Returns:
            Path to the saved plot
        """
        try:
            from visualizers.plots import plot_latency_comparison
            
            return plot_latency_comparison(self, output_path, include_percentiles)
            
        except ImportError:
            logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
            return None
    
    def plot_throughput_scaling(self, output_path: Optional[str] = None):
        """Plot throughput scaling with batch size."""
        try:
            from visualizers.plots import plot_throughput_scaling
            
            return plot_throughput_scaling(self, output_path)
            
        except ImportError:
            logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
            return None
    
    def plot_memory_usage(self, output_path: Optional[str] = None, detailed: bool = True):
        """
        Plot memory usage across hardware platforms.
        
        Args:
            output_path: Path to save the plot
            detailed: Whether to include detailed memory metrics
        
        Returns:
            Path to the saved plot
        """
        try:
            from visualizers.plots import plot_memory_usage
            
            return plot_memory_usage(self, output_path, detailed)
            
        except ImportError:
            logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
            return None
            
    def plot_flops_comparison(self, output_path: Optional[str] = None, detailed: bool = True):
        """
        Plot FLOPs comparison across hardware platforms.
        
        Args:
            output_path: Path to save the plot
            detailed: Whether to include detailed FLOPs breakdown
        
        Returns:
            Path to the saved plot
        """
        try:
            from visualizers.plots import plot_flops_comparison
            
            return plot_flops_comparison(self, output_path, detailed)
            
        except ImportError:
            logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
            return None
            
    def plot_power_efficiency(self, output_path: Optional[str] = None):
        """
        Plot power efficiency metrics across hardware platforms.
        
        Args:
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        try:
            from visualizers.plots import plot_power_efficiency
            
            return plot_power_efficiency(self, output_path)
            
        except ImportError:
            logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
            return None
            
    def plot_bandwidth_utilization(self, output_path: Optional[str] = None):
        """
        Plot memory bandwidth utilization metrics and roofline model.
        
        Args:
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        try:
            from visualizers.plots import plot_bandwidth_utilization
            
            return plot_bandwidth_utilization(self, output_path)
            
        except ImportError:
            logger.error("matplotlib is required for plotting. Install with 'pip install matplotlib'")
            return None
            
    def _get_gpu_theoretical_tflops(self) -> Dict[str, float]:
        """
        Get theoretical TFLOPS capabilities for available GPUs.
        
        Returns:
            Dictionary with GPU device names and their theoretical TFLOPS
        """
        tflops_dict = {}
        
        # Check for CUDA GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpu_name = props.name
                    # FMA = 2 ops per cycle per CUDA core
                    gpu_flops = props.multi_processor_count * props.max_threads_per_multi_processor * 2
                    gpu_clock = props.clock_rate / 1e3  # Convert to MHz
                    theoretical_tflops = gpu_flops * gpu_clock / 1e6  # Convert to TFLOPS
                    
                    tflops_dict[f"cuda:{i}"] = theoretical_tflops
                except Exception:
                    # Skip if we can't get device properties
                    pass
        
        return tflops_dict
    
    def _get_hardware_efficiency_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate hardware efficiency metrics for different hardware platforms.
        
        Returns:
            Dictionary with hardware efficiency metrics
        """
        efficiency_dict = {}
        
        # Group results by hardware
        for hw in set(result.hardware for result in self.results):
            hw_results = [r for r in self.results if r.hardware == hw]
            if not hw_results:
                continue
            
            # For each hardware, collect efficiency metrics
            hw_metrics = {}
            for result in hw_results:
                if "hardware_efficiency" in result.metrics:
                    hw_metrics["efficiency"] = result.metrics["hardware_efficiency"]
                if "tensor_core_eligible" in result.metrics:
                    hw_metrics["tensor_core_eligible"] = result.metrics["tensor_core_eligible"]
                if "model_type" in result.metrics:
                    hw_metrics["model_type"] = result.metrics["model_type"]
            
            if hw_metrics:
                efficiency_dict[hw] = hw_metrics
        
        return efficiency_dict
    
    def get_cpu_gpu_speedup(self) -> Optional[float]:
        """
        Calculate the CPU to GPU speedup ratio.
        
        Returns:
            Float speedup ratio or None if either CPU or GPU results are missing
        """
        cpu_latency = None
        gpu_latency = None
        
        # Find CPU and GPU results with batch size 1 and minimum sequence length
        cpu_results = [r for r in self.results if r.hardware == "cpu"]
        gpu_results = [r for r in self.results if r.hardware == "cuda"]
        
        if not cpu_results or not gpu_results:
            return None
        
        # Sort by batch size and sequence length
        cpu_results.sort(key=lambda r: (r.batch_size, r.sequence_length))
        gpu_results.sort(key=lambda r: (r.batch_size, r.sequence_length))
        
        # Get the results with batch size 1 if possible
        cpu_result = next((r for r in cpu_results if r.batch_size == 1), cpu_results[0])
        gpu_result = next((r for r in gpu_results if r.batch_size == 1), gpu_results[0])
        
        # Extract latency
        if "latency_ms" in cpu_result.metrics and "latency_ms" in gpu_result.metrics:
            cpu_latency = cpu_result.metrics["latency_ms"]
            gpu_latency = gpu_result.metrics["latency_ms"]
        
        # Calculate speedup
        if cpu_latency and gpu_latency and gpu_latency > 0:
            return cpu_latency / gpu_latency
        
        return None


class ModelBenchmark:
    """
    Benchmark runner for HuggingFace models.
    
    Runs performance benchmarks on HuggingFace models across different hardware platforms,
    collecting metrics such as latency, throughput, and memory usage.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """
        Initialize a model benchmark.
        
        Args:
            model_id: HuggingFace model ID
            **kwargs: Additional configuration parameters
        """
        # Check for dependencies
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for benchmarking. Install with 'pip install transformers'")
        
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for benchmarking. Install with 'pip install torch'")
        
        # Create configuration
        config_dict = {"model_id": model_id}
        config_dict.update(kwargs)
        self.config = BenchmarkConfig(**config_dict)
        
        # Initialize task if not provided
        if self.config.task is None:
            self._initialize_task()
        
        logger.info(f"Initialized benchmark for model {model_id} with task {self.config.task}")
    
    def _initialize_task(self):
        """Auto-detect the model task if not provided."""
        try:
            config = AutoConfig.from_pretrained(self.config.model_id)
            if hasattr(config, "architectures") and config.architectures:
                arch = config.architectures[0]
                
                # Mapping of architectures to tasks
                task_mapping = {
                    "ForCausalLM": "text-generation",
                    "ForMaskedLM": "fill-mask",
                    "ForSequenceClassification": "text-classification",
                    "ForQuestionAnswering": "question-answering",
                    "ForTokenClassification": "token-classification",
                    "ForSeq2SeqLM": "text2text-generation",
                    "ForImageClassification": "image-classification",
                    "ForObjectDetection": "object-detection",
                    "ForAudioClassification": "audio-classification",
                    "ForSpeechSeq2Seq": "automatic-speech-recognition"
                }
                
                for arch_suffix, task in task_mapping.items():
                    if arch_suffix in arch:
                        self.config.task = task
                        logger.info(f"Auto-detected task: {task} for model {self.config.model_id}")
                        return
            
            # Default to text-generation if we can't determine the task
            self.config.task = "text-generation"
            logger.warning(f"Could not auto-detect task for model {self.config.model_id}. Defaulting to {self.config.task}")
            
        except Exception as e:
            logger.warning(f"Error detecting task for model {self.config.model_id}: {e}")
            self.config.task = "text-generation"
            logger.warning(f"Defaulting to {self.config.task}")
    
    def _prepare_model_and_inputs(self, hardware: str, batch_size: int, sequence_length: int) -> Tuple:
        """
        Prepare the model and inputs for benchmarking.
        
        Args:
            hardware: Hardware platform to run on
            batch_size: Batch size for inputs
            sequence_length: Sequence length for inputs
            
        Returns:
            Tuple of (model, inputs, input_shape)
        """
        # Initialize hardware
        device = initialize_hardware(hardware)
        
        # Get optimization options from config
        use_4bit = getattr(self.config, "use_4bit", False)
        use_8bit = getattr(self.config, "use_8bit", False)
        
        # Load model adapter first to leverage model-specific input preparation
        try:
            from models import get_model_adapter
            model_adapter = get_model_adapter(self.config.model_id, self.config.task)
            
            # Load model with optimization options
            model = model_adapter.load_model(
                device=device, 
                use_4bit=use_4bit,
                use_8bit=use_8bit
            )
            
            # Prepare model-specific inputs
            inputs = model_adapter.prepare_inputs(batch_size, sequence_length)
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        except Exception as e:
            logger.warning(f"Error preparing model or inputs with adapter: {e}")
            logger.info("Falling back to basic input preparation")
            
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
                
                # Generate input text based on task
                if self.config.task in ["text-generation", "text-classification", "token-classification"]:
                    text = "The quick brown fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
                    text = text[:sequence_length]
                    inputs = tokenizer([text] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)
                
                elif self.config.task == "fill-mask":
                    if tokenizer.mask_token is None:
                        text = "The quick brown fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
                        text = text[:sequence_length]
                        inputs = tokenizer([text] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)
                    else:
                        text = f"The quick {tokenizer.mask_token} fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
                        text = text[:sequence_length]
                        inputs = tokenizer([text] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)
                
                elif self.config.task == "question-answering":
                    question = "What color is the fox?"
                    context = "The quick brown fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
                    context = context[:sequence_length]
                    inputs = tokenizer([question] * batch_size, [context] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)
                
                elif self.config.task == "text2text-generation":
                    text = "translate English to French: The quick brown fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
                    text = text[:sequence_length]
                    inputs = tokenizer([text] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)
                
                else:
                    # Default text input
                    text = "The quick brown fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
                    text = text[:sequence_length]
                    inputs = tokenizer([text] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=sequence_length)
                    
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Load model through adapter without quantization optimizations
                from models import get_model_adapter
                model_adapter = get_model_adapter(self.config.model_id, self.config.task)
                model = model_adapter.load_model(device=device)
                
            except Exception as e:
                logger.error(f"Error with fallback loading: {e}")
                # Ultimate fallback - random inputs with a dummy model
                logger.warning("Using dummy model and random inputs as a last resort")
                inputs = {"input_ids": torch.randint(0, 1000, (batch_size, sequence_length), dtype=torch.long).to(device)}
                inputs["attention_mask"] = torch.ones((batch_size, sequence_length), dtype=torch.long).to(device)
                
                # Create a simple model for testing
                model = nn.Sequential(
                    nn.Embedding(1000, 768),
                    nn.Linear(768, 768),
                    nn.LayerNorm(768)
                ).to(device)
        
        # Record input shapes
        input_shape = {k: list(v.shape) for k, v in inputs.items()}
        
        # Ensure model is in evaluation mode
        model.eval()
        
        return model, inputs, input_shape
    
    def _benchmark_on_hardware(self, hardware: str, batch_size: int, sequence_length: int) -> BenchmarkResult:
        """
        Run benchmarks on a specific hardware platform with given batch size and sequence length.
        
        Args:
            hardware: Hardware platform to run on
            batch_size: Batch size for inputs
            sequence_length: Sequence length for inputs
            
        Returns:
            BenchmarkResult with metrics
        """
        logger.info(f"Running benchmark on {hardware} with batch_size={batch_size}, sequence_length={sequence_length}")
        
        try:
            # Prepare model and inputs
            model, inputs, input_shape = self._prepare_model_and_inputs(hardware, batch_size, sequence_length)
            
            # Initialize metrics with hardware awareness
            metrics = {}
            metric_instances = []
            
            # Import factories
            from metrics.memory import MemoryMetricFactory
            from metrics.timing import TimingMetricFactory
            from metrics.flops import FLOPsMetricFactory
            from metrics.power import PowerMetricFactory
            from metrics.bandwidth import BandwidthMetricFactory
            
            # Create hardware-aware metrics
            flops_metric = None  # Store reference for bandwidth metric
            model_parameters = 0
            
            for metric_name in self.config.metrics:
                if metric_name == "latency":
                    metric_instances.append(TimingMetricFactory.create_latency_metric(device))
                elif metric_name == "throughput":
                    metric_instances.append(TimingMetricFactory.create_throughput_metric(device, batch_size=batch_size))
                elif metric_name == "memory":
                    metric_instances.append(MemoryMetricFactory.create(device))
                elif metric_name == "flops":
                    flops_metric = FLOPsMetricFactory.create(device)
                    flops_metric.set_model_and_inputs(model, inputs)
                    metric_instances.append(flops_metric)
                    # Calculate model size for bandwidth metric
                    model_parameters = sum(p.numel() * p.element_size() for p in model.parameters())
                elif metric_name == "power":
                    power_metric = PowerMetricFactory.create(device)
                    metric_instances.append(power_metric)
                elif metric_name == "bandwidth":
                    bandwidth_metric = BandwidthMetricFactory.create(device)
                    metric_instances.append(bandwidth_metric)
            
            # Start metrics collection
            for metric in metric_instances:
                metric.start()
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(self.config.warmup_iterations):
                    _ = model(**inputs)
            
            # Benchmark runs
            outputs = []
            with torch.no_grad():
                for i in range(self.config.test_iterations):
                    # For latency metrics, record the step start
                    for metric in metric_instances:
                        if isinstance(metric, LatencyMetric):
                            metric.record_step()
                    
                    # Run the model inference
                    output = model(**inputs)
                    outputs.append(output)
                    
                    # Update throughput metrics for each iteration
                    for metric in metric_instances:
                        if isinstance(metric, ThroughputMetric):
                            metric.update()
                        # For memory metrics, record at various points
                        elif isinstance(metric, MemoryMetric) and i % 5 == 0:  # Record every 5 iterations
                            metric.record_memory()
            
            # Stop metrics collection
            flops_count = 0
            throughput_value = 0
            
            # First get FLOPs and throughput for power metrics
            for metric in metric_instances:
                if isinstance(metric, FLOPsMetric):
                    metric.stop()
                    metrics_dict = metric.get_metrics()
                    flops_count = metrics_dict.get("flops", 0)
                elif isinstance(metric, ThroughputMetric):
                    metric.stop()
                    metrics_dict = metric.get_metrics()
                    throughput_value = metrics_dict.get("throughput_items_per_sec", 0)
            
            # Update metrics with operations count, throughput, and memory transfers
            for metric in metric_instances:
                if hasattr(metric, 'set_operations_count'):
                    metric.set_operations_count(flops_count)
                if hasattr(metric, 'set_throughput'):
                    metric.set_throughput(throughput_value)
                if hasattr(metric, 'set_memory_transfers') and model_parameters > 0:
                    # Estimate memory transfers based on model size, batch size, and inference count
                    if hasattr(metric, 'estimate_memory_transfers'):
                        metric.estimate_memory_transfers(
                            model_parameters, 
                            batch_size, 
                            self.config.test_iterations
                        )
                if hasattr(metric, 'set_compute_operations'):
                    metric.set_compute_operations(flops_count)
                
                # Stop any remaining metrics
                if not (isinstance(metric, FLOPsMetric) or isinstance(metric, ThroughputMetric)):
                    metric.stop()
                
                # Get metrics with different detail levels based on metric type
                if hasattr(metric, 'get_detailed_metrics'):
                    metrics.update(metric.get_detailed_metrics())
                else:
                    metrics.update(metric.get_metrics())
                    
                # Get roofline data if available
                if hasattr(metric, 'get_roofline_data'):
                    metrics.update({"roofline_data": metric.get_roofline_data()})
            
            # Record output shapes from the last output
            output_shape = {}
            if outputs:
                last_output = outputs[-1]
                if hasattr(last_output, "keys"):
                    # Handle dictionary-like outputs
                    for key, value in last_output.items():
                        if hasattr(value, "shape"):
                            output_shape[key] = list(value.shape)
                else:
                    # Handle tensor or tuple outputs
                    if hasattr(last_output, "shape"):
                        output_shape["output"] = list(last_output.shape)
                    elif isinstance(last_output, tuple):
                        for i, item in enumerate(last_output):
                            if hasattr(item, "shape"):
                                output_shape[f"output_{i}"] = list(item.shape)
            
            # Create result
            result = BenchmarkResult(
                model_id=self.config.model_id,
                hardware=hardware,
                batch_size=batch_size,
                sequence_length=sequence_length,
                metrics=metrics,
                input_shape=input_shape,
                output_shape=output_shape
            )
            
            logger.info(f"Benchmark complete on {hardware}: {metrics}")
            return result
            
        except Exception as e:
            logger.error(f"Error running benchmark on {hardware}: {e}")
            # Return a result with error information
            return BenchmarkResult(
                model_id=self.config.model_id,
                hardware=hardware,
                batch_size=batch_size,
                sequence_length=sequence_length,
                metrics={"error": str(e)},
                input_shape={},
                output_shape={}
            )
    
    def run(self) -> BenchmarkResults:
        """
        Run benchmarks according to the configuration.
        
        Returns:
            BenchmarkResults with all benchmark results
        """
        # Prepare parallel execution
        all_tasks = []
        for hw in self.config.hardware:
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    all_tasks.append((hw, batch_size, seq_len))
        
        results = []
        
        # Run benchmarks in parallel threads (one per hardware-batch-seq combination)
        with ThreadPoolExecutor(max_workers=min(len(all_tasks), 4)) as executor:
            future_to_task = {
                executor.submit(self._benchmark_on_hardware, hw, bs, sl): (hw, bs, sl)
                for hw, bs, sl in all_tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    hw, bs, sl = task
                    logger.error(f"Task {hw}-{bs}-{sl} generated an exception: {e}")
        
        # Create benchmark results
        benchmark_results = BenchmarkResults(results, self.config)
        
        # Save results if configured
        if self.config.save_results:
            benchmark_results.export_to_json()
        
        return benchmark_results


class BenchmarkSuite:
    """
    Run a suite of benchmarks on multiple models.
    """
    
    def __init__(self, models: List[str], **kwargs):
        """
        Initialize a benchmark suite.
        
        Args:
            models: List of HuggingFace model IDs to benchmark
            **kwargs: Additional configuration parameters applied to all models
        """
        self.models = models
        self.common_config = kwargs
    
    def run(self) -> Dict[str, BenchmarkResults]:
        """
        Run benchmarks for all models in the suite.
        
        Returns:
            Dictionary mapping model IDs to their benchmark results
        """
        results = {}
        
        for model_id in self.models:
            logger.info(f"Running benchmarks for model {model_id}")
            try:
                benchmark = ModelBenchmark(model_id, **self.common_config)
                model_results = benchmark.run()
                results[model_id] = model_results
            except Exception as e:
                logger.error(f"Error benchmarking model {model_id}: {e}")
        
        return results
    
    @classmethod
    def from_predefined_suite(cls, suite_name: str, **kwargs) -> "BenchmarkSuite":
        """
        Create a benchmark suite from a predefined set of models.
        
        Args:
            suite_name: Name of the predefined suite
            **kwargs: Additional configuration parameters
        
        Returns:
            BenchmarkSuite instance
        """
        # Predefined suites
        suites = {
            "text-classification": [
                "bert-base-uncased",
                "roberta-base",
                "distilbert-base-uncased",
                "albert-base-v2"
            ],
            "text-generation": [
                "gpt2",
                "facebook/opt-125m",
                "EleutherAI/gpt-neo-125m",
                "bigscience/bloom-560m"
            ],
            "text2text-generation": [
                "t5-small",
                "facebook/bart-base",
                "google/flan-t5-small"
            ],
            "image-classification": [
                "google/vit-base-patch16-224",
                "facebook/convnext-tiny-224",
                "microsoft/resnet-50"
            ],
            "popular-models": [
                "bert-base-uncased",
                "gpt2",
                "t5-small",
                "facebook/bart-base",
                "google/vit-base-patch16-224"
            ]
        }
        
        if suite_name not in suites:
            raise ValueError(f"Unknown suite: {suite_name}. Available suites: {list(suites.keys())}")
        
        return cls(suites[suite_name], **kwargs)


if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="HuggingFace Model Benchmark Suite")
    parser.add_argument("--model", type=str, nargs="+", help="HuggingFace model ID(s) to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4], help="Batch sizes to benchmark")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[16, 32, 64], help="Sequence lengths to benchmark")
    parser.add_argument("--hardware", type=str, nargs="+", default=["cpu"], help="Hardware platforms to benchmark on")
    parser.add_argument("--metrics", type=str, nargs="+", default=["latency", "throughput", "memory"], help="Metrics to collect")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save results")
    parser.add_argument("--export-formats", type=str, nargs="+", default=["json"], help="Export formats (json, csv, markdown)")
    parser.add_argument("--suite", type=str, help="Run a predefined benchmark suite")
    parser.add_argument("--publish-to-hub", action="store_true", help="Publish results to HuggingFace Hub")
    parser.add_argument("--token", type=str, help="HuggingFace Hub token for publishing")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not args.suite:
        parser.error("Either --model or --suite must be specified")
    
    # Run benchmarks
    if args.suite:
        suite = BenchmarkSuite.from_predefined_suite(
            args.suite,
            batch_sizes=args.batch_sizes,
            sequence_lengths=args.sequence_lengths,
            hardware=args.hardware,
            metrics=args.metrics,
            output_dir=args.output_dir
        )
        suite_results = suite.run()
        
        # Export results
        for model_id, results in suite_results.items():
            for export_format in args.export_formats:
                if export_format == "json":
                    results.export_to_json()
                elif export_format == "csv":
                    results.export_to_csv()
                elif export_format == "markdown":
                    results.export_to_markdown()
            
            if args.publish_to_hub:
                results.publish_to_hub(token=args.token)
        
    else:
        for model_id in args.model:
            benchmark = ModelBenchmark(
                model_id=model_id,
                batch_sizes=args.batch_sizes,
                sequence_lengths=args.sequence_lengths,
                hardware=args.hardware,
                metrics=args.metrics,
                output_dir=args.output_dir
            )
            
            results = benchmark.run()
            
            # Export results in requested formats
            for export_format in args.export_formats:
                if export_format == "json":
                    results.export_to_json()
                elif export_format == "csv":
                    results.export_to_csv()
                elif export_format == "markdown":
                    results.export_to_markdown()
            
            if args.publish_to_hub:
                results.publish_to_hub(token=args.token)