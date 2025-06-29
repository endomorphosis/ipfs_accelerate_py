#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace Hub exporter for benchmark results.

This module provides functionality for publishing benchmark results to HuggingFace Model Hub
model cards, creating standardized performance badges and detailed benchmark information.
"""

import os
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("benchmark.exporters.hf_hub")

# Hardware platform info for badges
HARDWARE_DISPLAY_NAMES = {
    "cpu": "CPU",
    "cuda": "CUDA",
    "mps": "MPS",
    "openvino": "OpenVINO",
    "webnn": "WebNN",
    "webgpu": "WebGPU",
}

HARDWARE_COLORS = {
    "cpu": "blue",
    "cuda": "orange",
    "mps": "green",
    "openvino": "red",
    "webnn": "purple",
    "webgpu": "brown",
}

# Badge templates
BENCHMARK_BADGE_TEMPLATE = "https://img.shields.io/badge/{hardware}%20Inference-{time_ms}ms-{color}"


class ModelCardExporter:
    """
    Exporter for publishing benchmark results to HuggingFace Model Hub model cards.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the model card exporter.
        
        Args:
            token: HuggingFace API token
        """
        self.token = token or os.environ.get("HF_TOKEN")
        
        # Check for required packages
        try:
            from huggingface_hub import (
                HfApi, 
                ModelCard, 
                ModelCardData, 
                CardData,
                login
            )
            self.hf_hub_available = True
            self.hf_api = HfApi()
            
            # Login if token provided
            if self.token:
                login(token=self.token)
                
        except ImportError:
            self.hf_hub_available = False
            logger.warning("huggingface_hub not available. Install with 'pip install huggingface_hub'")
    
    def format_benchmark_data(self, benchmark_results) -> Dict[str, Any]:
        """
        Format benchmark data for inclusion in model card.
        
        Args:
            benchmark_results: BenchmarkResults object
            
        Returns:
            Dict with formatted benchmark data and markdown table
        """
        if not benchmark_results or not benchmark_results.results:
            return None
        
        # Extract model ID
        model_id = benchmark_results.config.model_id
        
        # Create markdown table
        markdown_table = "## Performance Benchmarks\n\n"
        markdown_table += "| Hardware | Inference Time (ms) | Memory Usage (MB) | Batch Size | Sequence Length |\n"
        markdown_table += "|----------|---------------------|-------------------|------------|----------------|\n"
        
        # Add badge markdown
        badges_markdown = ""
        
        # Metadata dictionary
        metadata = {
            "benchmarks": []
        }
        
        # Group results by hardware
        hardware_results = {}
        for result in benchmark_results.results:
            if result.hardware not in hardware_results:
                hardware_results[result.hardware] = []
            hardware_results[result.hardware].append(result)
        
        # Sort hardware platforms
        sorted_hardware = sorted(hardware_results.keys())
        
        # Add rows to table and metadata
        for hardware in sorted_hardware:
            hw_display = HARDWARE_DISPLAY_NAMES.get(hardware, hardware)
            
            # Use the result with batch size 1 and smallest sequence length for the badge
            results = sorted(hardware_results[hardware], key=lambda r: (r.batch_size, r.sequence_length))
            representative_result = next((r for r in results if r.batch_size == 1), results[0])
            
            # Extract latency and memory metrics
            if "latency_ms" in representative_result.metrics:
                inference_ms = representative_result.metrics["latency_ms"]
            else:
                inference_ms = None
            
            if "memory_peak_mb" in representative_result.metrics:
                memory_mb = representative_result.metrics["memory_peak_mb"]
            elif "memory_used_mb" in representative_result.metrics:
                memory_mb = representative_result.metrics["memory_used_mb"]
            else:
                memory_mb = None
            
            # Format values
            inference_str = f"{inference_ms:.1f}" if inference_ms is not None else "N/A"
            memory_str = f"{memory_mb:.1f}" if memory_mb is not None else "N/A"
            batch_str = str(representative_result.batch_size)
            seq_str = str(representative_result.sequence_length)
            
            # Add to table
            markdown_table += f"| {hw_display} | {inference_str} | {memory_str} | {batch_str} | {seq_str} |\n"
            
            # Add to metadata
            metadata["benchmarks"].append({
                "hardware": hardware,
                "inference_time_ms": float(f"{inference_ms:.1f}") if inference_ms is not None else None,
                "memory_usage_mb": float(f"{memory_mb:.1f}") if memory_mb is not None else None,
                "batch_size": representative_result.batch_size,
                "sequence_length": representative_result.sequence_length,
                "timestamp": representative_result.timestamp
            })
            
            # Create badge for this hardware
            if inference_ms is not None:
                color = HARDWARE_COLORS.get(hardware, "blue")
                badge_url = BENCHMARK_BADGE_TEMPLATE.format(
                    hardware=hw_display,
                    time_ms=f"{inference_ms:.1f}",
                    color=color
                )
                badges_markdown += f"![{hw_display} Benchmark]({badge_url}) "
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        markdown_table += f"\n*Benchmarks last updated: {timestamp}*\n\n"
        markdown_table += "*Measured with IPFS Accelerate Testing Framework*\n\n"
        
        # Add badges section
        markdown_table = badges_markdown + "\n\n" + markdown_table
        
        # Calculate or retrieve CPU vs GPU speedup if available
        cpu_time = None
        gpu_time = None
        
        # First try to get it from BenchmarkResults if available
        speedup = None
        if hasattr(benchmark_results, "get_cpu_gpu_speedup"):
            speedup = benchmark_results.get_cpu_gpu_speedup()
        
        # Otherwise calculate it manually
        if speedup is None:
            for benchmark in metadata["benchmarks"]:
                if benchmark["hardware"] == "cpu" and benchmark["inference_time_ms"] is not None:
                    cpu_time = benchmark["inference_time_ms"]
                elif benchmark["hardware"] == "cuda" and benchmark["inference_time_ms"] is not None:
                    gpu_time = benchmark["inference_time_ms"]
            
            if cpu_time is not None and gpu_time is not None and gpu_time > 0:
                speedup = cpu_time / gpu_time
        
        # Add to markdown and metadata if available
        if speedup is not None:
            markdown_table += f"\n**CPU to GPU Speedup: {speedup:.1f}x**\n"
            markdown_table += f"The model runs {speedup:.1f} times faster on GPU compared to CPU under the same conditions.\n"
            metadata["cpu_gpu_speedup"] = float(f"{speedup:.1f}")
        
        # Add methodology
        markdown_table += """
## Benchmark Methodology

These benchmarks were collected using the IPFS Accelerate Testing Framework:

- **Inference Time**: Average time for a single forward pass
- **Memory Usage**: Peak memory consumption during inference
- **Batch Size**: Number of inputs processed simultaneously
- **Sequence Length**: Length of input sequences in tokens

Measurements performed in a controlled environment with warm-up passes to ensure stability.
"""
        
        return {
            "markdown": markdown_table,
            "metadata": metadata
        }
    
    def update_model_card(self, model_id: str, benchmark_data: Dict[str, Any], commit_message: str = "Update performance benchmarks") -> bool:
        """
        Update the model card for a specific model with benchmark information.
        
        Args:
            model_id: The Hugging Face model ID (e.g., "bert-base-uncased")
            benchmark_data: Dict with markdown and metadata for benchmarks
            commit_message: Custom commit message for the update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.hf_hub_available:
            logger.error("Hugging Face Hub not available, cannot update model card")
            return False
        
        if not benchmark_data:
            logger.error(f"No benchmark data provided for model {model_id}")
            return False
        
        try:
            from huggingface_hub import ModelCard, ModelCardData
            
            logger.info(f"Updating model card for {model_id}")
            
            # Try to get existing model card
            try:
                model_card = ModelCard.load(model_id)
                logger.info(f"Loaded existing model card for {model_id}")
            except Exception as e:
                logger.warning(f"Could not load existing model card for {model_id}: {e}")
                # Create new model card
                model_card = ModelCard(ModelCardData())
            
            # Add benchmarks to model card metadata
            if model_card.data:
                # Add benchmarks to existing data
                model_card.data.to_dict()["benchmark"] = benchmark_data["metadata"]
            else:
                # Create new data with benchmarks
                card_data = {"benchmark": benchmark_data["metadata"]}
                model_card.data = ModelCardData.from_dict(card_data)
            
            # Add benchmark section to model card content
            benchmark_section = benchmark_data["markdown"]
            
            if model_card.text:
                # Try to update existing benchmark section
                if "## Performance Benchmarks" in model_card.text:
                    # Replace existing section
                    parts = model_card.text.split("## Performance Benchmarks")
                    before = parts[0]
                    
                    # Find the next section
                    after_parts = parts[1].split("\n## ")
                    if len(after_parts) > 1:
                        after = "\n## " + "\n## ".join(after_parts[1:])
                    else:
                        after = ""
                    
                    model_card.text = before + benchmark_section + after
                else:
                    # Add new section at the end
                    model_card.text += "\n\n" + benchmark_section
            else:
                # Create new model card text
                model_card.text = f"# {model_id}\n\n{benchmark_section}"
            
            # Save card to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                model_card.save(f.name)
                temp_path = f.name
            
            # Push to hub
            self.hf_api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="README.md",
                repo_id=model_id,
                repo_type="model",
                commit_message=commit_message
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            logger.info(f"Successfully updated model card for {model_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating model card for {model_id}: {e}")
            return False
    
    def publish_results(self, benchmark_results, commit_message: str = "Update performance benchmarks from IPFS Accelerate Testing Framework") -> bool:
        """
        Publish benchmark results to HuggingFace Model Hub.
        
        Args:
            benchmark_results: BenchmarkResults object
            commit_message: Custom commit message for the update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.hf_hub_available:
            logger.error("Hugging Face Hub not available, cannot publish results")
            return False
        
        # Format benchmark data
        benchmark_data = self.format_benchmark_data(benchmark_results)
        
        if not benchmark_data:
            logger.error("Failed to format benchmark data")
            return False
        
        # Get model ID
        model_id = benchmark_results.config.model_id
        
        # Update model card
        return self.update_model_card(
            model_id=model_id,
            benchmark_data=benchmark_data,
            commit_message=commit_message
        )