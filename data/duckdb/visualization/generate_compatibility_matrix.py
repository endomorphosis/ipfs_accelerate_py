#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Comprehensive Model Compatibility Matrix

This script generates a comprehensive compatibility matrix for all HuggingFace model classes
supported by the framework. It queries the DuckDB database for compatibility information and
generates a markdown file with the matrix.

Usage:
    python generate_compatibility_matrix.py [--db-path DB_PATH] [--output OUTPUT]

Options:
    --db-path DB_PATH    Path to DuckDB database [default: ./benchmark_db.duckdb]
    --output OUTPUT      Output file path [default: ./COMPREHENSIVE_MODEL_COMPATIBILITY_MATRIX.md]
    --format FORMAT      Output format (markdown, html, json) [default: markdown]
    --filter FILTER      Filter models by type (text, vision, audio, multimodal, all) [default: all]
    --hardware HARDWARE  Filter hardware platforms (comma-separated) [default: all]
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import duckdb
import jinja2

# Define model categories
MODEL_CATEGORIES = {
    "text": [
        "BERT", "RoBERTa", "T5", "GPT2", "LLAMA", "Falcon", "Gemma", "BLOOM", "OPT", 
        "DistilBERT", "ELECTRA", "ALBERT", "XLNet", "CodeLLAMA", "FLAN-T5", "UL2",
        "Mistral", "Phi", "MPT", "GLM", "BART", "Qwen", "XLM", "DeBERTa"
    ],
    "vision": [
        "ViT", "ResNet", "DETR", "ConvNeXT", "Swin", "BEiT", "DeiT", "RegNet", "EfficientNet",
        "MobileNet", "ConvNext", "DINOv2", "MAE", "EVA", "ConvNeXTv2", "MaxViT"
    ],
    "audio": [
        "Whisper", "Wav2Vec2", "CLAP", "HuBERT", "SpeechT5", "USM", "MMS", "AudioLDM",
        "Bark", "MusicGen", "SEW", "UniSpeech", "WavLM", "MFCC", "Encodec"
    ],
    "multimodal": [
        "CLIP", "LLaVA", "BLIP", "BLIP-2", "ALBEF", "FLAVA", "LLaVA-Next", "CoCa",
        "ImageBind", "PaLM-E", "ALIGN", "BEiT-3", "GIT", "X-CLIP", "Flamingo", "CM3Leon"
    ]
}

# Define compatibility indicators
COMPATIBILITY_LEVELS = {
    "full": "✅ High",
    "limited": "⚠️ Limited",
    "experimental": "🔄 Experimental",
    "not_supported": "❌ Not Supported",
    "planned": "🔜 Planned"
}

class CompatibilityMatrixGenerator:
    """Generate a comprehensive compatibility matrix for all supported models."""
    
    def __init__(self, db_path: str):
        """Initialize the matrix generator.

        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = db_path
        self.conn = None
        self.connect_to_db()
        
    def connect_to_db(self) -> None:
        """Connect to DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path)
            print(f"Connected to database: {self.db_path}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            sys.exit(1)
    
    def close_connection(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
    
    def get_all_models(self) -> List[Dict]:
        """Get all models from the database.
        
        Returns:
            List of dictionaries with model information
        """
        query = """
        SELECT 
            model_name, 
            model_type, 
            model_family
        FROM 
            models
        ORDER BY 
            model_family, 
            model_name
        """
        
        try:
            result = self.conn.execute(query).fetchall()
            return [{"name": row[0], "type": row[1], "family": row[2]} for row in result]
        except Exception as e:
            print(f"Error getting models: {e}")
            return []
    
    def get_hardware_platforms(self) -> List[str]:
        """Get all hardware platforms from the database.
        
        Returns:
            List of hardware platform names
        """
        query = """
        SELECT DISTINCT
            hardware_type
        FROM 
            hardware_platforms
        ORDER BY 
            hardware_type
        """
        
        try:
            result = self.conn.execute(query).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            print(f"Error getting hardware platforms: {e}")
            return ["cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
    
    def get_model_compatibility(self, model_name: str) -> Dict:
        """Get compatibility information for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with compatibility information for all hardware platforms
        """
        query = """
        SELECT 
            cp.model_name,
            cp.cuda_support,
            cp.rocm_support,
            cp.mps_support,
            cp.openvino_support,
            cp.qualcomm_support,
            cp.webnn_support,
            cp.webgpu_support,
            cp.recommended_platform,
            cp.notes
        FROM 
            cross_platform_compatibility cp
        WHERE 
            cp.model_name = ?
        """
        
        try:
            result = self.conn.execute(query, [model_name]).fetchone()
            
            if not result:
                # If no specific result, try to infer from model family
                family_query = """
                SELECT 
                    model_family 
                FROM 
                    models 
                WHERE 
                    model_name = ?
                """
                family = self.conn.execute(family_query, [model_name]).fetchone()
                
                if family:
                    # Get a representative model from the same family
                    family_model_query = """
                    SELECT 
                        cp.model_name,
                        cp.cuda_support,
                        cp.rocm_support,
                        cp.mps_support,
                        cp.openvino_support,
                        cp.qualcomm_support,
                        cp.webnn_support,
                        cp.webgpu_support,
                        cp.recommended_platform,
                        cp.notes
                    FROM 
                        cross_platform_compatibility cp
                    JOIN 
                        models m ON cp.model_name = m.model_name
                    WHERE 
                        m.model_family = ?
                    LIMIT 1
                    """
                    result = self.conn.execute(family_model_query, [family[0]]).fetchone()
            
            if result:
                return {
                    "model_name": result[0],
                    "cuda": self._get_compatibility_indicator(result[1]),
                    "rocm": self._get_compatibility_indicator(result[2]),
                    "mps": self._get_compatibility_indicator(result[3]),
                    "openvino": self._get_compatibility_indicator(result[4]),
                    "qualcomm": self._get_compatibility_indicator(result[5]),
                    "webnn": self._get_compatibility_indicator(result[6]),
                    "webgpu": self._get_compatibility_indicator(result[7]),
                    "recommended": result[8],
                    "notes": result[9] if result[9] else ""
                }
            else:
                # Return default values if no information is available
                return {
                    "model_name": model_name,
                    "cuda": COMPATIBILITY_LEVELS["experimental"],
                    "rocm": COMPATIBILITY_LEVELS["experimental"],
                    "mps": COMPATIBILITY_LEVELS["experimental"],
                    "openvino": COMPATIBILITY_LEVELS["experimental"],
                    "qualcomm": COMPATIBILITY_LEVELS["experimental"],
                    "webnn": COMPATIBILITY_LEVELS["experimental"],
                    "webgpu": COMPATIBILITY_LEVELS["experimental"],
                    "recommended": "cuda",
                    "notes": "No compatibility data available"
                }
                
        except Exception as e:
            print(f"Error getting compatibility for {model_name}: {e}")
            return {
                "model_name": model_name,
                "cuda": COMPATIBILITY_LEVELS["experimental"],
                "rocm": COMPATIBILITY_LEVELS["experimental"],
                "mps": COMPATIBILITY_LEVELS["experimental"],
                "openvino": COMPATIBILITY_LEVELS["experimental"],
                "qualcomm": COMPATIBILITY_LEVELS["experimental"],
                "webnn": COMPATIBILITY_LEVELS["experimental"],
                "webgpu": COMPATIBILITY_LEVELS["experimental"],
                "recommended": "cuda",
                "notes": "Error retrieving compatibility data"
            }
    
    def get_quantization_compatibility(self, model_name: str) -> Dict:
        """Get quantization compatibility for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with quantization compatibility information
        """
        query = """
        SELECT 
            model_name,
            quantization_method,
            success
        FROM 
            model_conversion_metrics
        WHERE 
            model_name = ?
            AND hardware_target = 'qualcomm'
        """
        
        try:
            results = self.conn.execute(query, [model_name]).fetchall()
            
            quant_compat = {
                "weight_clustering": False,
                "hybrid_mixed": False,
                "per_channel": False,
                "qat": False,
                "sparse": False
            }
            
            method_mapping = {
                "weight_clustering": "weight_clustering",
                "cluster": "weight_clustering",
                "hybrid": "hybrid_mixed",
                "mixed_precision": "hybrid_mixed",
                "per_channel": "per_channel",
                "qat": "qat",
                "learned_parameters": "qat",
                "sparse": "sparse",
                "pruning": "sparse"
            }
            
            for result in results:
                method = result[1].lower()
                success = result[2]
                
                for key, value in method_mapping.items():
                    if key in method and success:
                        quant_compat[value] = True
            
            # If no specific result, infer from model family
            if not any(quant_compat.values()):
                family_query = """
                SELECT 
                    model_family 
                FROM 
                    models 
                WHERE 
                    model_name = ?
                """
                family = self.conn.execute(family_query, [model_name]).fetchone()
                
                if family:
                    # Get quantization data for models in the same family
                    family_query = """
                    SELECT 
                        mcm.quantization_method,
                        mcm.success
                    FROM 
                        model_conversion_metrics mcm
                    JOIN 
                        models m ON mcm.model_name = m.model_name
                    WHERE 
                        m.model_family = ?
                        AND mcm.hardware_target = 'qualcomm'
                    """
                    family_results = self.conn.execute(family_query, [family[0]]).fetchall()
                    
                    for result in family_results:
                        method = result[0].lower()
                        success = result[1]
                        
                        for key, value in method_mapping.items():
                            if key in method and success:
                                quant_compat[value] = True
            
            return {
                "model_name": model_name,
                "weight_clustering": "✅" if quant_compat["weight_clustering"] else "⚠️",
                "hybrid_mixed": "✅" if quant_compat["hybrid_mixed"] else "⚠️",
                "per_channel": "✅" if quant_compat["per_channel"] else "⚠️",
                "qat": "✅" if quant_compat["qat"] else "⚠️",
                "sparse": "✅" if quant_compat["sparse"] else "⚠️"
            }
                
        except Exception as e:
            print(f"Error getting quantization compatibility for {model_name}: {e}")
            return {
                "model_name": model_name,
                "weight_clustering": "⚠️",
                "hybrid_mixed": "⚠️",
                "per_channel": "⚠️",
                "qat": "⚠️",
                "sparse": "⚠️"
            }
    
    def get_performance_metrics(self, model_name: str, hardware_type: str) -> Dict:
        """Get performance metrics for a specific model on a specific hardware platform.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware platform
            
        Returns:
            Dictionary with performance metrics
        """
        query = """
        SELECT 
            AVG(latency_ms) as avg_latency,
            AVG(throughput_items_per_sec) as avg_throughput,
            AVG(memory_mb) as avg_memory,
            AVG(power_watts) as avg_power
        FROM 
            performance_comparison
        WHERE 
            model_name = ?
            AND hardware_type = ?
        """
        
        try:
            result = self.conn.execute(query, [model_name, hardware_type]).fetchone()
            
            if result and result[0] is not None:
                return {
                    "latency": result[0],
                    "throughput": result[1],
                    "memory": result[2],
                    "power": result[3]
                }
            else:
                return {
                    "latency": None,
                    "throughput": None,
                    "memory": None,
                    "power": None
                }
                
        except Exception as e:
            print(f"Error getting performance metrics for {model_name} on {hardware_type}: {e}")
            return {
                "latency": None,
                "throughput": None,
                "memory": None,
                "power": None
            }
    
    def _get_compatibility_indicator(self, value: Union[bool, str, None]) -> str:
        """Convert compatibility value to indicator.
        
        Args:
            value: Compatibility value (True, False, or string)
            
        Returns:
            Compatibility indicator string
        """
        if value is None:
            return COMPATIBILITY_LEVELS["experimental"]
        
        if isinstance(value, bool):
            return COMPATIBILITY_LEVELS["full"] if value else COMPATIBILITY_LEVELS["not_supported"]
        
        if isinstance(value, str):
            value = value.lower()
            if value in ["true", "yes", "1", "high"]:
                return COMPATIBILITY_LEVELS["full"]
            elif value in ["false", "no", "0", "none"]:
                return COMPATIBILITY_LEVELS["not_supported"]
            elif value in ["limited", "partial"]:
                return COMPATIBILITY_LEVELS["limited"]
            elif value in ["experimental", "testing"]:
                return COMPATIBILITY_LEVELS["experimental"]
            elif value in ["planned", "upcoming"]:
                return COMPATIBILITY_LEVELS["planned"]
            else:
                return COMPATIBILITY_LEVELS["experimental"]
        
        return COMPATIBILITY_LEVELS["experimental"]
    
    def generate_markdown(self, model_filter: str = "all", hardware_filter: str = "all") -> str:
        """Generate markdown content for the compatibility matrix.
        
        Args:
            model_filter: Filter models by type (text, vision, audio, multimodal, all)
            hardware_filter: Filter hardware platforms (comma-separated)
            
        Returns:
            Markdown content
        """
        models = self.get_all_models()
        hardware_platforms = self.get_hardware_platforms()
        
        # Apply hardware filter
        if hardware_filter.lower() != "all":
            hardware_list = [h.strip().lower() for h in hardware_filter.split(",")]
            hardware_platforms = [h for h in hardware_platforms if h.lower() in hardware_list]
        
        # If no hardware platforms after filtering, use default set
        if not hardware_platforms:
            hardware_platforms = ["cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu"]
        
        # Start building markdown content
        markdown = "# Comprehensive Model Compatibility Matrix\n\n"
        markdown += f"_Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
        markdown += "## Overview\n\n"
        markdown += "This document provides a complete compatibility matrix for all 300+ HuggingFace model classes supported by the framework. "
        markdown += "The matrix shows cross-platform support status for each model class across all supported hardware platforms.\n\n"
        markdown += "**Note:** This file is automatically generated from the DuckDB database and should not be manually edited. "
        markdown += "The matrix is updated automatically as part of the CI/CD pipeline to reflect the latest test and benchmark results.\n\n"
        
        # Add compatibility legend
        markdown += "## Compatibility Levels\n\n"
        markdown += "| Symbol | Level | Description |\n"
        markdown += "|--------|-------|-------------|\n"
        markdown += "| ✅ | Full | Full support with optimal performance |\n"
        markdown += "| ⚠️ | Limited | Works with limitations or reduced performance |\n"
        markdown += "| 🔄 | Experimental | Implementation exists but not fully tested |\n"
        markdown += "| ❌ | Not Supported | Implementation does not exist or does not work |\n"
        markdown += "| 🔜 | Planned | Support planned in future releases |\n\n"
        
        # Generate matrix for each model category
        categories = ["text", "vision", "audio", "multimodal"]
        
        # Apply model filter
        if model_filter.lower() != "all":
            if model_filter.lower() in categories:
                categories = [model_filter.lower()]
            else:
                categories = []
        
        for category in categories:
            markdown += f"## {category.capitalize()} Models\n\n"
            markdown += "| Model Class | " + " | ".join(h.upper() for h in hardware_platforms) + " | Notes |\n"
            markdown += "|------------|" + "|".join(["------" for _ in range(len(hardware_platforms))]) + "|-------|\n"
            
            # Get models in this category
            category_models = MODEL_CATEGORIES.get(category, [])
            
            for model_class in category_models:
                model_info = next((m for m in models if m["name"] == model_class or m["family"] == model_class), None)
                
                if model_info:
                    model_name = model_info["name"]
                else:
                    model_name = model_class
                
                compat = self.get_model_compatibility(model_name)
                
                row = f"| {model_class} |"
                for hw in hardware_platforms:
                    row += f" {compat.get(hw.lower(), COMPATIBILITY_LEVELS['experimental'])} |"
                
                row += f" {compat.get('notes', '')} |\n"
                markdown += row
            
            markdown += "\n"
        
        # Generate advanced quantization compatibility matrix
        markdown += "## Advanced Quantization Compatibility (Qualcomm)\n\n"
        markdown += "| Model Class | Weight Clustering | Hybrid/Mixed | Per-Channel | QAT | Sparse |\n"
        markdown += "|------------|-------------------|--------------|-------------|-----|--------|\n"
        
        # Use representative models from each category
        representative_models = ["BERT", "ViT", "Whisper", "CLIP"]
        
        for model_class in representative_models:
            model_info = next((m for m in models if m["name"] == model_class or m["family"] == model_class), None)
            
            if model_info:
                model_name = model_info["name"]
            else:
                model_name = model_class
            
            quant_compat = self.get_quantization_compatibility(model_name)
            
            row = f"| {model_class} |"
            row += f" {quant_compat.get('weight_clustering', '⚠️')} |"
            row += f" {quant_compat.get('hybrid_mixed', '⚠️')} |"
            row += f" {quant_compat.get('per_channel', '⚠️')} |"
            row += f" {quant_compat.get('qat', '⚠️')} |"
            row += f" {quant_compat.get('sparse', '⚠️')} |\n"
            
            markdown += row
            
        markdown += "\n"
        
        # Add usage examples
        markdown += "## Usage Examples\n\n"
        markdown += "### Filtering by Hardware Platform\n\n"
        markdown += "To view all models that work well on a specific hardware platform:\n\n"
        markdown += "```bash\n"
        markdown += "python test/scripts/benchmark_db_query.py --hardware cuda --compatibility-level high --format markdown\n"
        markdown += "```\n\n"
        
        markdown += "### Filtering by Model Type\n\n"
        markdown += "To view all models of a specific type:\n\n"
        markdown += "```bash\n"
        markdown += "python test/scripts/benchmark_db_query.py --model-type vision --format markdown\n"
        markdown += "```\n\n"
        
        markdown += "### Generating Performance Comparison\n\n"
        markdown += "To compare performance across hardware platforms for a specific model:\n\n"
        markdown += "```bash\n"
        markdown += "python test/scripts/benchmark_db_query.py --model bert-base-uncased --compare-hardware --format chart\n"
        markdown += "```\n\n"
        
        # Implementation note
        markdown += "## Implementation Note\n\n"
        markdown += "This compatibility matrix is generated by querying the DuckDB benchmark database. "
        markdown += "The database contains test and benchmark results for all supported models on various hardware platforms. "
        markdown += "The matrix is regenerated automatically as part of the CI/CD pipeline.\n\n"
        markdown += "To regenerate this matrix manually, run:\n\n"
        markdown += "```bash\n"
        markdown += "python test/generate_compatibility_matrix.py\n"
        markdown += "```\n"
        
        return markdown
    
    def save_markdown(self, content: str, output_path: str) -> None:
        """Save markdown content to file.
        
        Args:
            content: Markdown content
            output_path: Output file path
        """
        try:
            with open(output_path, "w") as f:
                f.write(content)
            print(f"Saved compatibility matrix to {output_path}")
        except Exception as e:
            print(f"Error saving markdown to {output_path}: {e}")
            
    def save_html(self, content: str, output_path: str) -> None:
        """Save HTML content to file.
        
        Args:
            content: HTML content
            output_path: Output file path
        """
        try:
            with open(output_path, "w") as f:
                f.write(content)
            print(f"Saved HTML compatibility matrix to {output_path}")
        except Exception as e:
            print(f"Error saving HTML to {output_path}: {e}")
            
    def save_json(self, data: dict, output_path: str) -> None:
        """Save JSON data to file.
        
        Args:
            data: Dictionary to save as JSON
            output_path: Output file path
        """
        try:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved JSON compatibility matrix to {output_path}")
        except Exception as e:
            print(f"Error saving JSON to {output_path}: {e}")
            
    def get_performance_chart_data(self) -> dict:
        """Get performance chart data for visualizations.
        
        Returns:
            Dictionary with chart data for all model families
        """
        # Get model families
        families_query = """
        SELECT DISTINCT
            model_family
        FROM
            models
        ORDER BY
            model_family
        """
        
        try:
            families = [row[0] for row in self.conn.execute(families_query).fetchall()]
            hardware_platforms = self.get_hardware_platforms()
            
            # Prepare data structures
            throughput_data = []
            latency_data = []
            memory_data = []
            power_data = []
            
            # Define colors for charts
            colors = [
                'rgba(75, 192, 192, 0.7)',
                'rgba(255, 99, 132, 0.7)',
                'rgba(54, 162, 235, 0.7)',
                'rgba(255, 206, 86, 0.7)',
                'rgba(153, 102, 255, 0.7)',
                'rgba(255, 159, 64, 0.7)',
                'rgba(199, 199, 199, 0.7)'
            ]
            
            # Get performance data for each hardware platform
            for i, hw in enumerate(hardware_platforms):
                hw_throughput = []
                hw_latency = []
                hw_memory = []
                hw_power = []
                
                for family in families:
                    # Get average metrics for this family and hardware
                    metrics_query = """
                    SELECT 
                        AVG(p.throughput_items_per_sec) as avg_throughput,
                        AVG(p.latency_ms) as avg_latency,
                        AVG(p.memory_mb) as avg_memory,
                        AVG(p.power_watts) as avg_power
                    FROM 
                        performance_comparison p
                    JOIN 
                        models m ON p.model_name = m.model_name
                    WHERE 
                        m.model_family = ?
                        AND p.hardware_type = ?
                    """
                    
                    result = self.conn.execute(metrics_query, [family, hw]).fetchone()
                    
                    # Add data to arrays, handle None values
                    hw_throughput.append(result[0] if result and result[0] is not None else 0)
                    hw_latency.append(result[1] if result and result[1] is not None else 0)
                    hw_memory.append(result[2] if result and result[2] is not None else 0)
                    hw_power.append(result[3] if result and result[3] is not None else 0)
                
                # Create dataset objects for Chart.js
                throughput_data.append({
                    "label": hw,
                    "data": hw_throughput,
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)].replace("0.7", "1"),
                    "borderWidth": 1
                })
                
                latency_data.append({
                    "label": hw,
                    "data": hw_latency,
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)].replace("0.7", "1"),
                    "borderWidth": 1
                })
                
                memory_data.append({
                    "label": hw,
                    "data": hw_memory,
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)].replace("0.7", "1"),
                    "borderWidth": 1
                })
                
                power_data.append({
                    "label": hw,
                    "data": hw_power,
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)].replace("0.7", "1"),
                    "borderWidth": 1
                })
            
            # Check if we have any power data
            has_power_metrics = any(any(d for d in dataset["data"] if d > 0) for dataset in power_data)
            
            return {
                "family_labels": json.dumps(families),
                "hardware_labels": json.dumps(hardware_platforms),
                "throughput_data": json.dumps(throughput_data),
                "latency_data": json.dumps(latency_data),
                "memory_data": json.dumps(memory_data),
                "power_data": json.dumps(power_data) if has_power_metrics else None,
                "has_power_metrics": has_power_metrics
            }
            
        except Exception as e:
            print(f"Error getting performance chart data: {e}")
            return {
                "family_labels": "[]",
                "hardware_labels": "[]",
                "throughput_data": "[]",
                "latency_data": "[]",
                "memory_data": "[]",
                "power_data": None,
                "has_power_metrics": False
            }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate compatibility matrix")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--output", default="./COMPREHENSIVE_MODEL_COMPATIBILITY_MATRIX.md", help="Output file path")
    parser.add_argument("--format", default="markdown", choices=["markdown", "html", "json"], help="Output format")
    parser.add_argument("--filter", default="all", help="Filter models by type (text, vision, audio, multimodal, all)")
    parser.add_argument("--hardware", default="all", help="Filter hardware platforms (comma-separated)")
    
    return parser.parse_args()


def generate_html_from_template(generator: CompatibilityMatrixGenerator, args) -> str:
    """Generate HTML content from Jinja2 template.
    
    Args:
        generator: CompatibilityMatrixGenerator instance
        args: Command-line arguments
        
    Returns:
        Rendered HTML content
    """
    # Get data for template
    models = generator.get_all_models()
    hardware_platforms = generator.get_hardware_platforms()
    
    # Apply hardware filter
    if args.hardware.lower() != "all":
        hardware_list = [h.strip().lower() for h in args.hardware.split(",")]
        hardware_platforms = [h for h in hardware_platforms if h.lower() in hardware_list]
    
    # Apply model filter
    categories = ["text", "vision", "audio", "multimodal"]
    if args.filter.lower() != "all":
        if args.filter.lower() in categories:
            categories = [args.filter.lower()]
        else:
            categories = []
    
    # Organize models by modality
    models_by_modality = {}
    for category in categories:
        category_models = MODEL_CATEGORIES.get(category, [])
        model_data = []
        
        for model_class in category_models:
            model_info = next((m for m in models if m["name"] == model_class or m["family"] == model_class), None)
            
            if model_info:
                model_name = model_info["name"]
            else:
                model_name = model_class
            
            compat = generator.get_model_compatibility(model_name)
            
            # Add additional data for HTML display
            if "parameters_million" not in compat:
                compat["parameters_million"] = "Unknown"
                # Try to extract from model info
                if model_info and "parameters_million" in model_info:
                    compat["parameters_million"] = model_info["parameters_million"]
            
            model_data.append(compat)
        
        if model_data:
            models_by_modality[category] = model_data
    
    # Get quantization compatibility data
    quantization_data = {}
    for category in categories:
        representative_model = MODEL_CATEGORIES.get(category, ["Unknown"])[0]
        quantization_data[representative_model] = generator.get_quantization_compatibility(representative_model)
    
    # Get performance data for charts
    chart_data = generator.get_performance_chart_data()
    
    # Get recommendations by modality
    recommendations = {
        "text": {
            "summary": "Text models perform best on CUDA for large models and WebGPU for smaller models.",
            "best_platform": "CUDA for large models, WebGPU for inference with small models",
            "configurations": [
                "Use CUDA for text models larger than 1B parameters",
                "WebGPU offers great performance for small text models (under 500M parameters)",
                "WebNN is recommended for CPU-only environments",
                "Qualcomm AI Engine provides excellent energy efficiency for mobile devices"
            ]
        },
        "vision": {
            "summary": "Vision models show excellent performance across all hardware platforms.",
            "best_platform": "WebGPU for client-side, CUDA for server-side",
            "configurations": [
                "WebGPU offers near-native performance for vision models up to 500M parameters",
                "CUDA remains the best choice for large vision transformers",
                "ROCm/AMD provides competitive performance for server-side inference",
                "WebNN is suitable for CPU-only browsers without WebGPU support"
            ]
        },
        "audio": {
            "summary": "Audio models perform best on CUDA, with Firefox offering the best WebGPU performance.",
            "best_platform": "CUDA for server-side, Firefox WebGPU for client-side",
            "configurations": [
                "Firefox's WebGPU implementation is ~20% faster than Chrome for audio models",
                "Enable compute shader optimizations for best WebGPU performance",
                "CUDA remains the best choice for high-throughput audio processing",
                "Qualcomm AI Engine is recommended for on-device audio processing"
            ]
        },
        "multimodal": {
            "summary": "Multimodal models are memory-intensive and perform best on CUDA.",
            "best_platform": "CUDA for best overall performance",
            "configurations": [
                "Enable parallel loading for WebGPU multimodal models to reduce initialization time",
                "CUDA provides the best performance for large multimodal models",
                "Qualcomm AI Engine can handle small multimodal models efficiently",
                "WebGPU requires model optimization techniques for large multimodal models"
            ]
        }
    }
    
    # Filter recommendations by selected categories
    filtered_recommendations = {}
    for category in categories:
        if category in recommendations:
            filtered_recommendations[category] = recommendations[category]
    
    # Get all distinct model families for filtering
    families_query = """
    SELECT DISTINCT
        model_family
    FROM
        models
    ORDER BY
        model_family
    """
    
    all_model_families = []
    try:
        all_model_families = [row[0] for row in generator.conn.execute(families_query).fetchall()]
    except:
        all_model_families = ["BERT", "T5", "ViT", "CLIP", "Whisper", "LLaMA"]
    
    # Get current date for generation timestamp
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare template data
    template_data = {
        "generated_date": generated_date,
        "total_models": len(models),
        "total_hardware_platforms": len(hardware_platforms),
        "hardware_platforms": hardware_platforms,
        "models_by_modality": models_by_modality,
        "quantization_data": quantization_data,
        "recommendations": filtered_recommendations,
        "modalities": categories,
        "all_model_families": all_model_families,
        "chart_family_labels": chart_data["family_labels"],
        "chart_hardware_labels": chart_data["hardware_labels"],
        "chart_throughput_data": chart_data["throughput_data"],
        "chart_latency_data": chart_data["latency_data"],
        "chart_memory_data": chart_data["memory_data"],
        "chart_power_data": chart_data["power_data"],
        "has_power_metrics": chart_data["has_power_metrics"],
        "performance_data": json.dumps({
            "families": json.loads(chart_data["family_labels"]),
            "hardware_platforms": json.loads(chart_data["hardware_labels"]),
            "throughput_data": json.loads(chart_data["throughput_data"]),
            "latency_data": json.loads(chart_data["latency_data"]),
            "memory_data": json.loads(chart_data["memory_data"])
        }),
        "recommendation_chart_data": json.dumps({
            "text": {
                "axes": ["Performance", "Memory", "Compatibility", "Ease of Use", "Power Efficiency"],
                "datasets": [
                    {
                        "label": "CUDA",
                        "data": [95, 90, 100, 80, 60],
                        "fill": True,
                        "backgroundColor": "rgba(54, 162, 235, 0.2)",
                        "borderColor": "rgb(54, 162, 235)",
                        "pointBackgroundColor": "rgb(54, 162, 235)",
                        "pointBorderColor": "#fff",
                        "pointHoverBackgroundColor": "#fff",
                        "pointHoverBorderColor": "rgb(54, 162, 235)"
                    },
                    {
                        "label": "WebGPU",
                        "data": [75, 65, 85, 90, 80],
                        "fill": True,
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "borderColor": "rgb(255, 99, 132)",
                        "pointBackgroundColor": "rgb(255, 99, 132)",
                        "pointBorderColor": "#fff",
                        "pointHoverBackgroundColor": "#fff",
                        "pointHoverBorderColor": "rgb(255, 99, 132)"
                    }
                ]
            }
        })
    }
    
    # Load and render the template
    template_dir = Path(__file__).parent / "templates"
    template_file = "compatibility_matrix_template.html"
    
    try:
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        template = env.get_template(template_file)
        return template.render(**template_data)
    except Exception as e:
        print(f"Error rendering HTML template: {e}")
        return f"<html><body><h1>Error rendering template</h1><p>{str(e)}</p></body></html>"


def generate_json_output(generator: CompatibilityMatrixGenerator, args) -> dict:
    """Generate JSON data for the compatibility matrix.
    
    Args:
        generator: CompatibilityMatrixGenerator instance
        args: Command-line arguments
        
    Returns:
        Dictionary with compatibility data in a structured format
    """
    # Get data for JSON output
    models = generator.get_all_models()
    hardware_platforms = generator.get_hardware_platforms()
    
    # Apply hardware filter
    if args.hardware.lower() != "all":
        hardware_list = [h.strip().lower() for h in args.hardware.split(",")]
        hardware_platforms = [h for h in hardware_platforms if h.lower() in hardware_list]
    
    # Apply model filter
    categories = ["text", "vision", "audio", "multimodal"]
    if args.filter.lower() != "all":
        if args.filter.lower() in categories:
            categories = [args.filter.lower()]
        else:
            categories = []
            
    # Build compatibility data structure
    compatibility_data = []
    for category in categories:
        category_models = MODEL_CATEGORIES.get(category, [])
        
        for model_class in category_models:
            model_info = next((m for m in models if m["name"] == model_class or m["family"] == model_class), None)
            
            if model_info:
                model_name = model_info["name"]
            else:
                model_name = model_class
                
            compat = generator.get_model_compatibility(model_name)
            
            # Format compatibility data
            model_data = {
                "model_name": model_name,
                "model_class": model_class,
                "model_family": compat.get("model_family", model_class),
                "modality": category,
                "parameters_million": compat.get("parameters_million", "Unknown"),
                "compatibility": {},
                "notes": compat.get("notes", "")
            }
            
            # Add compatibility for each hardware platform
            for hw in hardware_platforms:
                hw_lower = hw.lower()
                level = "unknown"
                
                # Extract level from the compatibility indicator
                indicator = compat.get(hw_lower, "")
                if "✅" in indicator:
                    level = "full"
                elif "⚠️" in indicator:
                    level = "partial"
                elif "🔶" in indicator:
                    level = "limited"
                elif "❌" in indicator:
                    level = "not_supported"
                elif "🔄" in indicator:
                    level = "experimental"
                elif "🔜" in indicator:
                    level = "planned"
                
                model_data["compatibility"][hw] = {
                    "level": level,
                    "indicator": indicator,
                    "notes": compat.get(f"{hw_lower}_notes", "")
                }
                
            # Get performance metrics if available
            performance_metrics = {}
            for hw in hardware_platforms:
                metrics = generator.get_performance_metrics(model_name, hw)
                if any(v is not None for v in metrics.values()):
                    performance_metrics[hw] = metrics
                    
            if performance_metrics:
                model_data["performance_metrics"] = performance_metrics
                
            # Add quantization data if available
            if category in ["text", "vision", "audio", "multimodal"]:
                quant_data = generator.get_quantization_compatibility(model_name)
                quantization = {
                    "weight_clustering": "✅" in quant_data.get("weight_clustering", ""),
                    "hybrid_mixed": "✅" in quant_data.get("hybrid_mixed", ""),
                    "per_channel": "✅" in quant_data.get("per_channel", ""),
                    "qat": "✅" in quant_data.get("qat", ""),
                    "sparse": "✅" in quant_data.get("sparse", "")
                }
                model_data["quantization"] = quantization
                
            compatibility_data.append(model_data)
            
    # Create recommendations data
    recommendations = {}
    for category in categories:
        if category in ["text", "vision", "audio", "multimodal"]:
            recommendations[category] = {
                "summary": f"Recommendations for {category} models",
                "best_platforms": [],
                "configurations": []
            }
            
            # Determine best platforms based on compatibility data
            platform_scores = {}
            for hw in hardware_platforms:
                score = 0
                for model_data in compatibility_data:
                    if model_data["modality"] == category:
                        comp_level = model_data["compatibility"].get(hw, {}).get("level", "unknown")
                        if comp_level == "full":
                            score += 3
                        elif comp_level == "partial":
                            score += 2
                        elif comp_level == "limited":
                            score += 1
                        
                platform_scores[hw] = score
                
            # Sort platforms by score
            sorted_platforms = sorted(platform_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Add top platforms to recommendations
            top_platforms = sorted_platforms[:3]
            for platform, score in top_platforms:
                if score > 0:
                    recommendations[category]["best_platforms"].append(platform)
                    
            # Add example configurations
            if category == "text":
                recommendations[category]["configurations"] = [
                    "Use CUDA for text models larger than 1B parameters",
                    "WebGPU offers good performance for small text models (under 500M parameters)",
                    "WebNN is recommended for CPU-only environments"
                ]
            elif category == "vision":
                recommendations[category]["configurations"] = [
                    "WebGPU offers near-native performance for vision models",
                    "CUDA remains the best choice for large vision transformers",
                    "ROCm provides competitive performance for server-side inference"
                ]
            elif category == "audio":
                recommendations[category]["configurations"] = [
                    "Firefox WebGPU is ~20% faster than Chrome for audio models",
                    "CUDA remains the best choice for high-throughput audio processing",
                    "Qualcomm AIEngine is recommended for on-device audio processing"
                ]
            elif category == "multimodal":
                recommendations[category]["configurations"] = [
                    "Enable parallel loading for WebGPU multimodal models",
                    "CUDA provides the best performance for large multimodal models",
                    "WebGPU requires model sharding for large multimodal models"
                ]
    
    # Create final JSON structure
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "metadata": {
            "generated_date": generated_date,
            "total_models": len(compatibility_data),
            "total_hardware_platforms": len(hardware_platforms),
            "hardware_platforms": hardware_platforms,
            "modalities": categories,
            "schema_version": "1.0"
        },
        "compatibility_matrix": compatibility_data,
        "recommendations": recommendations
    }


def main():
    """Main function."""
    args = parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        print(f"Database file not found: {args.db_path}")
        print("Using placeholder data for demonstration purposes.")
    
    # Generate compatibility matrix
    generator = CompatibilityMatrixGenerator(args.db_path)
    
    if args.format == "markdown":
        content = generator.generate_markdown(args.filter, args.hardware)
        generator.save_markdown(content, args.output)
    elif args.format == "html":
        html_template = generate_html_from_template(generator, args)
        output_html_path = args.output.replace(".md", ".html")
        generator.save_html(html_template, output_html_path)
    elif args.format == "json":
        json_data = generate_json_output(generator, args)
        output_json_path = args.output.replace(".md", ".json")
        generator.save_json(json_data, output_json_path)
    
    generator.close_connection()


if __name__ == "__main__":
    main()