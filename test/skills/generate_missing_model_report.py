#!/usr/bin/env python3
"""
Generate Missing Model Report

This script analyzes HuggingFace model test coverage by identifying missing model implementations
and generating a prioritized implementation plan.

Usage:
    python generate_missing_model_report.py [--directory TESTS_DIR] [--report REPORT_FILE]
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
import datetime
from typing import Dict, List, Set, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import architecture and task mappings from the standardization script if available
try:
    from standardize_task_configurations import ARCHITECTURE_TYPES
except ImportError:
    # Define them here as fallback
    ARCHITECTURE_TYPES = {
        "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert"],
        "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
        "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
        "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "resnet"],
        "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
        "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
        "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
    }

# List of all main HuggingFace model families we want to support
# These are grouped by architecture type and sorted by priority/popularity
# Populate with most important/commonly used models first
ALL_MODELS = {
    "encoder-only": [
        "bert", "roberta", "distilbert", "deberta", "albert", "xlm-roberta", "electra", 
        "camembert", "flaubert", "ernie", "luke", "mpnet", "mobilebert", "nezha", "squeezebert",
        "roformer", "layoutlm", "splinter", "convbert", "funnel", "ibert", "nystromformer",
        "deberta-v2", "xlm", "xlnet", "xlm-roberta-xl", "megatron-bert", "data2vec-text",
        "esm", "xmod", "mra"
    ],
    
    "decoder-only": [
        "gpt2", "llama", "mistral", "falcon", "phi", "bloom", "opt", "gpt-neo", 
        "gpt-neox", "gpt-j", "mixtral", "mpt", "gemma", "qwen2", "qwen3", 
        "stablelm", "codegen", "persimmon", "openai-gpt", "codellama", "olmo",
        "command-r", "gemma2", "gemma3", "mamba", "rwkv", "starcoder2", "olmoe",
        "phi3", "phi4", "nemotron", "mistral-next", "recurrent-gemma", "llama-3"
    ],
    
    "encoder-decoder": [
        "t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan-t5",
        "pegasus-x", "m2m-100", "switch-transformers", "seamless-m4t", "mt5", "umt5"
    ],
    
    "vision": [
        "vit", "swin", "deit", "beit", "convnext", "resnet", "poolformer", "dinov2",
        "regnet", "efficientnet", "mobilenet-v1", "mobilenet-v2", "segformer", 
        "levit", "cvt", "mlp-mixer", "bit", "imagegpt", "detr", "mask2former", "dino",
        "dinat", "swinv2", "van", "vitdet", "dpt", "mobilevit", "yolos", "conditional-detr",
        "convnextv2", "beit3", "depth-anything"
    ],
    
    "vision-text": [
        "clip", "blip", "vision-text-dual-encoder", "vision-encoder-decoder", "chinese-clip",
        "clipseg", "blip-2", "xclip", "instructblip", "git", "pix2struct"
    ],
    
    "speech": [
        "whisper", "wav2vec2", "hubert", "speecht5", "bark", "encodec", "musicgen",
        "speech-to-text", "speech-to-text-2", "unispeech", "wav2vec2-conformer", "wavlm"
    ],
    
    "multimodal": [
        "llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava",
        "llava-next", "fuyu", "kosmos-2", "idefics", "flava", "qwen2-vl", "qwen3-vl",
        "siglip", "imagebind", "llava-next-video", "idefics2", "idefics3", "mllama"
    ]
}

# Model importance (priority) categories
PRIORITY_CATEGORIES = {
    "critical": {
        "encoder-only": ["bert", "roberta", "distilbert", "deberta", "xlm-roberta", "albert"],
        "decoder-only": ["gpt2", "llama", "mistral", "falcon", "phi", "bloom", "opt", "gpt-j"],
        "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "flan-t5"],
        "vision": ["vit", "swin", "deit", "beit", "convnext", "resnet"],
        "vision-text": ["clip", "blip", "vision-text-dual-encoder"],
        "speech": ["whisper", "wav2vec2", "hubert", "speecht5"],
        "multimodal": ["llava", "pix2struct", "paligemma"]
    },
    "high": {
        "encoder-only": ["electra", "camembert", "mpnet", "ernie", "luke", "roformer"],
        "decoder-only": ["mixtral", "mpt", "gemma", "qwen2", "qwen3", "gpt-neo", "gpt-neox", "codellama"],
        "encoder-decoder": ["longt5", "led", "marian", "mt5", "pegasus-x"],
        "vision": ["dinov2", "poolformer", "regnet", "efficientnet", "mobilenet-v2", "segformer"],
        "vision-text": ["blip-2", "git", "chinese-clip", "clipseg"],
        "speech": ["bark", "encodec", "musicgen", "wavlm"],
        "multimodal": ["video-llava", "llava-next", "fuyu", "kosmos-2", "idefics", "flava"]
    },
    "medium": {
        # All other models not in critical or high
    }
}

class MissingModelAnalyzer:
    """Analyzes model test coverage and identifies missing model implementations."""
    
    def __init__(self, directory: str, report_file: Optional[str] = None):
        """Initialize the analyzer.
        
        Args:
            directory: Directory containing test files
            report_file: Path to write report (optional)
        """
        self.directory = Path(directory)
        self.report_file = report_file
        
        # Statistics
        self.stats = {
            "total_implemented": 0,
            "total_missing": 0,
            "by_architecture": {},
            "by_priority": {
                "critical": {"implemented": 0, "missing": 0},
                "high": {"implemented": 0, "missing": 0},
                "medium": {"implemented": 0, "missing": 0}
            }
        }
        
        # Results
        self.implemented_models = set()
        self.missing_models = {}
        
        # Initialize architecture stats
        for arch in ARCHITECTURE_TYPES:
            self.stats["by_architecture"][arch] = {
                "implemented": 0,
                "missing": 0,
                "total": len(ALL_MODELS.get(arch, []))
            }
        # Add unknown architecture for models that don't match known types
        self.stats["by_architecture"]["unknown"] = {
            "implemented": 0,
            "missing": 0,
            "total": 0
        }
    
    def run(self):
        """Run the analysis."""
        # Find all existing test files - both test_hf_*.py and plain test_*.py
        test_files = list(self.directory.glob("test_hf_*.py")) + list(self.directory.glob("test_*.py"))
        # Remove duplicates if any
        test_files = list(set(test_files))
        logger.info(f"Found {len(test_files)} test files to analyze")
        
        # Extract implemented models
        for file_path in test_files:
            model_name = self._extract_model_name(file_path)
            self.implemented_models.add(model_name)
            
            # Determine architecture
            architecture = self._get_model_architecture(model_name)
            
            # Update statistics
            self.stats["total_implemented"] += 1
            self.stats["by_architecture"][architecture]["implemented"] += 1
            
            # Update priority stats
            priority = self._get_model_priority(model_name, architecture)
            self.stats["by_priority"][priority]["implemented"] += 1
        
        # Find missing models
        for architecture, models in ALL_MODELS.items():
            missing_in_arch = []
            
            for model in models:
                if model not in self.implemented_models:
                    priority = self._get_model_priority(model, architecture)
                    missing_in_arch.append({
                        "name": model,
                        "priority": priority
                    })
                    
                    # Update statistics
                    self.stats["total_missing"] += 1
                    self.stats["by_architecture"][architecture]["missing"] += 1
                    self.stats["by_priority"][priority]["missing"] += 1
            
            if missing_in_arch:
                self.missing_models[architecture] = missing_in_arch
        
        # Sort missing models by priority
        for architecture in self.missing_models:
            self.missing_models[architecture].sort(
                key=lambda m: 0 if m["priority"] == "critical" else (1 if m["priority"] == "high" else 2)
            )
        
        # Generate report
        self._generate_report()
        
        return {
            "implemented": list(self.implemented_models),
            "missing": self.missing_models,
            "stats": self.stats
        }
    
    def _extract_model_name(self, file_path: Path) -> str:
        """Extract the model name from a test file path."""
        stem = file_path.stem
        if stem.startswith("test_hf_"):
            return stem.replace("test_hf_", "")
        elif stem.startswith("test_"):
            return stem.replace("test_", "")
        return stem
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Determine the model architecture from the model name."""
        if not model_name:
            return "unknown"
            
        # Normalize the model name to handle variations
        model_name = self._standardize_model_name(model_name)
        
        # Check for model type in architecture mappings
        for architecture, models in ARCHITECTURE_TYPES.items():
            if any(model_name == model or model_name.startswith(f"{model}-") or 
                   model_name.startswith(f"{model}_") for model in models):
                return architecture
            
        # If not found, check for substring matches (less precise)
        for architecture, models in ARCHITECTURE_TYPES.items():
            for model in models:
                if model in model_name:
                    return architecture
        
        # Default to unknown if no match found
        return "unknown"
        
    def _standardize_model_name(self, name):
        """Standardize model names to handle both hyphenated and underscore versions."""
        if not name:
            return name
            
        # Hyphenated to underscore mappings
        HYPHENATED_MODELS = {
            'gpt-j': 'gpt_j',
            'gpt-neo': 'gpt_neo',
            'gpt-neox': 'gpt_neox',
            'flan-t5': 'flan_t5',
            'xlm-roberta': 'xlm_roberta',
            'vision-text-dual-encoder': 'vision_text_dual_encoder',
            'speech-to-text': 'speech_to_text',
            'speech-to-text-2': 'speech_to_text_2',
            'data2vec-text': 'data2vec_text',
            'data2vec-audio': 'data2vec_audio',
            'data2vec-vision': 'data2vec_vision',
            'wav2vec2-conformer': 'wav2vec2_conformer',
            'transfo-xl': 'transfo_xl',
            'mlp-mixer': 'mlp_mixer'
        }
        
        # Check if the name exists in the mapping
        if name in HYPHENATED_MODELS:
            return HYPHENATED_MODELS[name]
        
        # Check if this is an underscore version of a hyphenated name
        for hyphenated, underscore in HYPHENATED_MODELS.items():
            if name == underscore:
                return name
        
        # Default to original name
        return name
    
    def _get_model_priority(self, model_name: str, architecture: str) -> str:
        """Determine the priority of a model."""
        model_name_lower = model_name.lower()
        
        # Check critical models
        if model_name_lower in PRIORITY_CATEGORIES["critical"].get(architecture, []):
            return "critical"
        
        # Check high priority models
        if model_name_lower in PRIORITY_CATEGORIES["high"].get(architecture, []):
            return "high"
        
        # Default to medium
        return "medium"
    
    def _generate_report(self):
        """Generate a missing model report."""
        if not self.report_file:
            return
        
        # Create report directory if needed
        report_dir = os.path.dirname(self.report_file)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        
        total_models = self.stats["total_implemented"] + self.stats["total_missing"]
        
        # Generate markdown report
        report = [
            "# HuggingFace Model Coverage Report",
            "",
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total models tracked:** {total_models}",
            f"- **Implemented models:** {self.stats['total_implemented']} ({self.stats['total_implemented']/total_models*100:.1f}%)",
            f"- **Missing models:** {self.stats['total_missing']} ({self.stats['total_missing']/total_models*100:.1f}%)",
            "",
            "## Coverage by Architecture",
            ""
        ]
        
        # Add architecture-specific stats
        for arch, stats in sorted(self.stats["by_architecture"].items()):
            arch_total = stats["implemented"] + stats["missing"]
            coverage_pct = stats["implemented"] / arch_total * 100 if arch_total > 0 else 0
            report.extend([
                f"### {arch.capitalize()} Models",
                "",
                f"- **Total {arch} models:** {arch_total}",
                f"- **Implemented:** {stats['implemented']} ({coverage_pct:.1f}%)",
                f"- **Missing:** {stats['missing']}",
                ""
            ])
        
        # Add coverage by priority
        report.extend([
            "## Coverage by Priority",
            ""
        ])
        
        for priority in ["critical", "high", "medium"]:
            priority_stats = self.stats["by_priority"][priority]
            priority_total = priority_stats["implemented"] + priority_stats["missing"]
            coverage_pct = priority_stats["implemented"] / priority_total * 100 if priority_total > 0 else 0
            
            report.extend([
                f"### {priority.capitalize()} Priority Models",
                "",
                f"- **Total {priority} models:** {priority_total}",
                f"- **Implemented:** {priority_stats['implemented']} ({coverage_pct:.1f}%)",
                f"- **Missing:** {priority_stats['missing']}",
                ""
            ])
        
        # Add implementation roadmap
        report.extend([
            "## Implementation Roadmap",
            "",
            "### Critical Priority Models",
            "",
            "These models should be implemented first due to their importance and widespread use:"
        ])
        
        critical_models = []
        for arch, models in self.missing_models.items():
            for model in models:
                if model["priority"] == "critical":
                    critical_models.append((arch, model["name"]))
        
        if critical_models:
            report.append("")
            for arch, model in sorted(critical_models):
                report.append(f"- `{model}` ({arch})")
        else:
            report.append("")
            report.append("*All critical priority models have been implemented!*")
        
        # Add high priority models
        report.extend([
            "",
            "### High Priority Models",
            "",
            "These models should be implemented next:"
        ])
        
        high_models = []
        for arch, models in self.missing_models.items():
            for model in models:
                if model["priority"] == "high":
                    high_models.append((arch, model["name"]))
        
        if high_models:
            report.append("")
            for arch, model in sorted(high_models):
                report.append(f"- `{model}` ({arch})")
        else:
            report.append("")
            report.append("*All high priority models have been implemented!*")
        
        # Add medium priority models by architecture
        report.extend([
            "",
            "### Medium Priority Models",
            "",
            "These models can be implemented after critical and high priority models:"
        ])
        
        for arch in sorted(self.missing_models.keys()):
            medium_models = [model["name"] for model in self.missing_models[arch] if model["priority"] == "medium"]
            
            if medium_models:
                report.extend([
                    "",
                    f"#### {arch.capitalize()} Models",
                    ""
                ])
                
                for model in sorted(medium_models):
                    report.append(f"- `{model}`")
        
        # Add implemented models list
        report.extend([
            "",
            "## Implemented Models",
            "",
            "These models already have test implementations:"
        ])
        
        # Group implemented models by architecture
        implemented_by_arch = {}
        for model in sorted(self.implemented_models):
            arch = self._get_model_architecture(model)
            if arch not in implemented_by_arch:
                implemented_by_arch[arch] = []
            implemented_by_arch[arch].append(model)
        
        for arch in sorted(implemented_by_arch.keys()):
            report.extend([
                "",
                f"### {arch.capitalize()} Models",
                ""
            ])
            
            for model in sorted(implemented_by_arch[arch]):
                report.append(f"- `{model}`")
        
        # Write report to file
        with open(self.report_file, 'w') as f:
            f.write("\n".join(report))
        
        logger.info(f"Missing model report written to {self.report_file}")

def main():
    """Main entry point for the analyzer."""
    parser = argparse.ArgumentParser(description="Analyze HuggingFace model test coverage")
    parser.add_argument("--directory", "-d", type=str, default="fixed_tests",
                        help="Directory containing test files")
    parser.add_argument("--report", "-r", type=str, default="reports/missing_models.md",
                        help="Path to write report")
    
    args = parser.parse_args()
    
    # Resolve directory path
    directory = args.directory
    if not os.path.isabs(directory):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    
    # Create and run the analyzer
    analyzer = MissingModelAnalyzer(directory, args.report)
    results = analyzer.run()
    
    # Print a brief summary
    total_models = analyzer.stats["total_implemented"] + analyzer.stats["total_missing"]
    critical_missing = analyzer.stats["by_priority"]["critical"]["missing"]
    high_missing = analyzer.stats["by_priority"]["high"]["missing"]
    
    print("\nMODEL COVERAGE SUMMARY")
    print("="*50)
    print(f"Total models tracked: {total_models}")
    print(f"Implemented models: {analyzer.stats['total_implemented']} ({analyzer.stats['total_implemented']/total_models*100:.1f}%)")
    print(f"Missing models: {analyzer.stats['total_missing']} ({analyzer.stats['total_missing']/total_models*100:.1f}%)")
    print("\nPriority breakdown:")
    print(f"- Critical models missing: {critical_missing}")
    print(f"- High priority models missing: {high_missing}")
    
    if args.report:
        print(f"\nDetailed report written to: {args.report}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())