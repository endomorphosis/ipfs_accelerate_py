#!/usr/bin/env python3
"""
Generate Missing Model Report

This script identifies HuggingFace model classes without corresponding test files
and generates a report prioritizing which models should be implemented next.

Usage:
    python generate_missing_model_report.py [--output-report REPORT_FILE]
"""

import os
import sys
import json
import argparse
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model type categorization
MODEL_CATEGORIES = {
    "encoder-only": [
        "bert", "roberta", "albert", "distilbert", "electra", "deberta", "camembert", 
        "xlm-roberta", "xlnet", "mpnet", "ernie", "canine", "roformer", "convbert"
    ],
    "decoder-only": [
        "gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon",
        "phi", "mixtral", "mpt", "opt", "gemma", "stablelm", "pythia", "xglm", "codellama"
    ],
    "encoder-decoder": [
        "t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan-t5",
        "prophetnet", "bigbird-pegasus", "fsmt", "m2m-100", "plbart"
    ],
    "vision": [
        "vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "resnet",
        "levit", "efficientnet", "mobilenet", "regnet", "dino", "segformer", "detr",
        "mask2former", "yolos"
    ],
    "vision-text": [
        "vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip",
        "chinese-clip", "alt-clip", "blip-2", "siglip", "fuyu"
    ],
    "speech": [
        "wav2vec2", "hubert", "whisper", "bark", "speecht5", "unispeech", "wavlm",
        "sew", "encodec", "musicgen", "audio-spectrogram-transformer", "clap"
    ],
    "multimodal": [
        "llava", "git", "paligemma", "imagebind", "flamingo", "blip", "idefics",
        "video-llava", "flava", "owlvit", "groupvit", "kosmos-2"
    ]
}

# Priority levels for models
MODEL_PRIORITIES = {
    "high": [
        "bert", "gpt2", "t5", "vit", "roberta", "llama", "bart", "swin", "clip", 
        "whisper", "wav2vec2", "blip", "distilbert", "mistral", "phi", "falcon"
    ],
    "medium": [
        "albert", "deberta", "electra", "gpt-j", "gpt-neo", "pegasus", "mbart", 
        "deit", "convnext", "hubert", "llava", "flan-t5", "opt", "gemma", "beit"
    ],
    "low": [
        "camembert", "xlm-roberta", "bloom", "mixtral", "mpt", "led", "longt5", 
        "poolformer", "dinov2", "detr", "mask2former", "unispeech", "encodec"
    ]
}

class MissingModelAnalyzer:
    """Analyzer for identifying missing model tests."""
    
    def __init__(self, test_directory: str, output_report: str):
        """Initialize the analyzer.
        
        Args:
            test_directory: Directory containing model test files
            output_report: Path to output report file
        """
        self.test_directory = Path(test_directory)
        self.output_report = Path(output_report)
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "missing_models": [],
            "stats": {
                "total_known_models": 0,
                "implemented_models": 0,
                "missing_models": 0,
                "by_category": {},
                "by_priority": {
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "unknown": 0
                }
            }
        }
    
    def get_implemented_models(self) -> Set[str]:
        """Get the set of models with implemented tests."""
        implemented = set()
        
        # Look for test files of the form test_hf_*.py
        test_files = self.test_directory.glob("test_hf_*.py")
        
        for file_path in test_files:
            # Extract model name from file name
            # test_hf_bert.py -> bert
            model_name = file_path.stem.replace("test_hf_", "")
            implemented.add(model_name)
        
        return implemented
    
    def get_known_models(self) -> Dict[str, Dict[str, Any]]:
        """Get the set of all known model types with their categories and priorities."""
        known_models = {}
        
        # Compile full list from MODEL_CATEGORIES
        for category, models in MODEL_CATEGORIES.items():
            for model in models:
                if model not in known_models:
                    # Determine priority
                    if model in MODEL_PRIORITIES["high"]:
                        priority = "high"
                    elif model in MODEL_PRIORITIES["medium"]:
                        priority = "medium"
                    elif model in MODEL_PRIORITIES["low"]:
                        priority = "low"
                    else:
                        priority = "unknown"
                    
                    known_models[model] = {
                        "category": category,
                        "priority": priority
                    }
        
        return known_models
    
    def analyze(self) -> Dict:
        """Analyze implemented vs. missing models and generate report."""
        implemented_models = self.get_implemented_models()
        known_models = self.get_known_models()
        
        # Initialize category stats
        for category in MODEL_CATEGORIES.keys():
            self.results["stats"]["by_category"][category] = {
                "total": 0,
                "implemented": 0,
                "missing": 0
            }
        
        # Track total and implemented models
        self.results["stats"]["total_known_models"] = len(known_models)
        self.results["stats"]["implemented_models"] = len(implemented_models)
        
        # Find missing models
        missing_models = []
        
        for model_name, model_info in known_models.items():
            category = model_info["category"]
            priority = model_info["priority"]
            
            # Update category counts
            self.results["stats"]["by_category"][category]["total"] += 1
            
            if model_name in implemented_models:
                self.results["stats"]["by_category"][category]["implemented"] += 1
            else:
                self.results["stats"]["by_category"][category]["missing"] += 1
                self.results["stats"]["by_priority"][priority] += 1
                
                missing_models.append({
                    "name": model_name,
                    "category": category,
                    "priority": priority
                })
        
        # Sort missing models by priority (high first) and then by name
        missing_models.sort(
            key=lambda x: (
                0 if x["priority"] == "high" else 
                1 if x["priority"] == "medium" else 
                2 if x["priority"] == "low" else 3,
                x["name"]
            )
        )
        
        self.results["missing_models"] = missing_models
        self.results["stats"]["missing_models"] = len(missing_models)
        
        # Generate and save the report
        self.generate_report()
        
        return self.results
    
    def generate_report(self):
        """Generate and save the missing models report."""
        # Save JSON report
        json_report_path = self.output_report.with_suffix('.json')
        with open(json_report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate markdown report
        md_report = self._generate_markdown_report()
        with open(self.output_report, 'w') as f:
            f.write(md_report)
        
        logger.info(f"Missing models report saved to {self.output_report}")
        logger.info(f"JSON report saved to {json_report_path}")
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown report of missing models."""
        total = self.results["stats"]["total_known_models"]
        implemented = self.results["stats"]["implemented_models"]
        missing = self.results["stats"]["missing_models"]
        
        # Calculate implementation percentage
        implementation_percent = round(implemented / total * 100, 2) if total > 0 else 0
        
        report = [
            "# HuggingFace Model Implementation Report",
            f"\nGenerated on: {self.results['timestamp']}",
            
            "\n## Summary",
            f"- **Total known models**: {total}",
            f"- **Implemented models**: {implemented} ({implementation_percent}%)",
            f"- **Missing models**: {missing} ({round(100 - implementation_percent, 2)}%)",
            
            "\n## Implementation Status by Category",
            "| Category | Total | Implemented | Missing | Implementation % |",
            "|----------|-------|-------------|---------|------------------|",
        ]
        
        # Add category statistics
        for category, stats in sorted(self.results["stats"]["by_category"].items()):
            cat_total = stats["total"]
            cat_implemented = stats["implemented"]
            cat_missing = stats["missing"]
            cat_percent = round(cat_implemented / cat_total * 100, 2) if cat_total > 0 else 0
            
            report.append(f"| {category} | {cat_total} | {cat_implemented} | {cat_missing} | {cat_percent}% |")
        
        # Add missing models by priority
        report.extend([
            "\n## Missing Models by Priority",
            "\n### High Priority",
            "| Model Name | Category |",
            "|------------|----------|",
        ])
        
        # Add high priority missing models
        high_priority = [m for m in self.results["missing_models"] if m["priority"] == "high"]
        if high_priority:
            for model in high_priority:
                report.append(f"| {model['name']} | {model['category']} |")
        else:
            report.append("| *None* | - |")
        
        report.extend([
            "\n### Medium Priority",
            "| Model Name | Category |",
            "|------------|----------|",
        ])
        
        # Add medium priority missing models
        medium_priority = [m for m in self.results["missing_models"] if m["priority"] == "medium"]
        if medium_priority:
            for model in medium_priority:
                report.append(f"| {model['name']} | {model['category']} |")
        else:
            report.append("| *None* | - |")
        
        report.extend([
            "\n### Low Priority",
            "| Model Name | Category |",
            "|------------|----------|",
        ])
        
        # Add low priority missing models
        low_priority = [m for m in self.results["missing_models"] if m["priority"] == "low"]
        if low_priority:
            for model in low_priority:
                report.append(f"| {model['name']} | {model['category']} |")
        else:
            report.append("| *None* | - |")
        
        # Add next steps section
        report.extend([
            "\n## Next Steps",
            "\n1. **Implement High Priority Models First**",
            "   - Focus on models with high user demand and community interest",
            "   - These models represent core architectures used in production environments",
            
            "\n2. **Batch Implementation by Category**",
            "   - Implement models in batches by category",
            "   - This allows reusing templates and patterns within categories",
            
            "\n3. **Validation and Testing**",
            "   - Validate each implemented model with syntax checks",
            "   - Run functional tests with small model variants",
            "   - Integrate with the distributed testing framework",
            
            "\n4. **Documentation and Coverage Tracking**",
            "   - Update coverage tracking tools after each implementation",
            "   - Maintain documentation for each implemented model",
            "   - Publish regular updates to the implementation roadmap"
        ])
        
        return "\n".join(report)

def main():
    """Main entry point for the missing model analyzer."""
    parser = argparse.ArgumentParser(description="Generate report of missing model tests")
    parser.add_argument("--test-directory", "-d", type=str, default="fixed_tests",
                      help="Directory containing existing test files")
    parser.add_argument("--output-report", "-o", type=str, default="reports/missing_models.md",
                      help="Path to output report file")
    
    args = parser.parse_args()
    
    # Resolve paths
    test_directory = args.test_directory
    if not os.path.isabs(test_directory):
        test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_directory)
    
    output_report = args.output_report
    if not os.path.isabs(output_report):
        output_report = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_report)
    
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    
    # Create analyzer and run analysis
    analyzer = MissingModelAnalyzer(test_directory, output_report)
    results = analyzer.analyze()
    
    # Print summary
    logger.info(f"Analysis complete.")
    logger.info(f"Total known models: {results['stats']['total_known_models']}")
    logger.info(f"Implemented models: {results['stats']['implemented_models']}")
    logger.info(f"Missing models: {results['stats']['missing_models']}")
    
    # Log missing high priority models
    high_priority_missing = [m for m in results["missing_models"] if m["priority"] == "high"]
    if high_priority_missing:
        logger.info(f"High priority missing models: {len(high_priority_missing)}")
        for model in high_priority_missing[:5]:  # Show first 5
            logger.info(f"  - {model['name']} ({model['category']})")
        if len(high_priority_missing) > 5:
            logger.info(f"  ... and {len(high_priority_missing) - 5} more")

if __name__ == "__main__":
    main()