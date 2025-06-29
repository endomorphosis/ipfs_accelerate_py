#!/usr/bin/env python3
"""
Generate tests for high-priority HuggingFace models.

This script uses the enhanced_generator to generate test files for the 
high-priority models defined in the HF_MODEL_COVERAGE_ROADMAP.md file.
It implements the models in order of priority to achieve full coverage.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the enhanced generator
try:
    from enhanced_generator import (
        generate_test, generate_all_tests, MODEL_REGISTRY,
        ARCHITECTURE_TYPES, validate_generated_file
    )
    HAS_ENHANCED_GENERATOR = True
except ImportError:
    logger.error("Cannot import enhanced_generator.py. Please ensure it exists and is importable.")
    HAS_ENHANCED_GENERATOR = False
    sys.exit(1)

# Define model priority categories from the roadmap
HIGH_PRIORITY_MODELS = [
    # Decoder-only Models
    "gpt2", "mistral", "falcon", "mixtral", "phi", "codellama", "qwen", 
    "gpt-neo", "gpt-j", "llama", "mamba",
    
    # Encoder-decoder Models
    "t5", "flan-t5", "longt5", "bart", "pegasus", "mbart", "led",
    
    # Encoder-only Models
    "bert", "deberta", "deberta-v2", "roberta", "distilbert", "albert", "xlm-roberta", "electra", 
    
    # Vision Models
    "vit", "deit", "beit", "swin", "convnext", "dinov2", "mobilenet-v2", "detr", "yolos",
    
    # Speech Models
    "whisper", "wav2vec2", "hubert", "bark", "speecht5",
    
    # Vision-text Models
    "clip", "blip", "blip-2", "chinese-clip", "clipseg",
    
    # Multimodal Models
    "llava", "git", "paligemma", "video-llava", "fuyu", "kosmos-2", "llava-next"
]

MEDIUM_PRIORITY_MODELS = [
    # Encoder-only Models
    "camembert", "flaubert", "ernie", "rembert", "luke", "mpnet", "canine", "layoutlm",
    
    # Decoder-only Models
    "olmo", "qwen2", "qwen3", "gemma", "pythia", "stable-lm", "xglm", "gpt-neox",
    
    # Encoder-decoder Models
    "marian", "mt5", "umt5", "pegasus-x", "plbart", "m2m-100", "nllb", 
    
    # Vision Models
    "convnextv2", "efficientnet", "levit", "mobilevit", "poolformer", "resnet", "swinv2", "cvt",
    
    # Specialty Models
    "imagebind", "groupvit", "perceiver", "mask2former", "segformer"
]

def get_implemented_models() -> List[str]:
    """Return list of model types already defined in MODEL_REGISTRY."""
    return list(MODEL_REGISTRY.keys())

def get_missing_models() -> List[str]:
    """Return list of high-priority models not yet implemented."""
    implemented = get_implemented_models()
    return [model for model in HIGH_PRIORITY_MODELS if model not in implemented]

def generate_priority_model_tests(output_dir: str, priority: str = "high") -> Dict[str, Any]:
    """
    Generate test files for priority models.
    
    Args:
        output_dir (str): Directory to save generated test files
        priority (str): Priority level - "high", "medium", or "all"
        
    Returns:
        dict: Summary of generation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which models to generate
    if priority == "high":
        models_to_generate = HIGH_PRIORITY_MODELS
    elif priority == "medium":
        models_to_generate = MEDIUM_PRIORITY_MODELS
    elif priority == "all":
        models_to_generate = HIGH_PRIORITY_MODELS + MEDIUM_PRIORITY_MODELS
    else:
        models_to_generate = HIGH_PRIORITY_MODELS  # Default to high priority
    
    # Filter to only include implemented models
    implemented = get_implemented_models()
    models_to_generate = [model for model in models_to_generate if model in implemented]
    
    # Generate tests for each model
    results = {}
    success_count = 0
    valid_count = 0
    
    for model_type in models_to_generate:
        try:
            logger.info(f"Generating test for {model_type}...")
            result = generate_test(model_type, output_dir)
            results[model_type] = result
            
            if result.get("success", False):
                success_count += 1
                
            if result.get("is_valid", False):
                valid_count += 1
                
        except Exception as e:
            logger.error(f"Error generating test for {model_type}: {e}")
            results[model_type] = {
                "success": False,
                "error": str(e),
                "model_type": model_type
            }
    
    # Create summary
    summary = {
        "total": len(models_to_generate),
        "successful": success_count,
        "valid_syntax": valid_count,
        "results": results
    }
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "priority_models_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Generated {success_count} of {len(models_to_generate)} test files")
    logger.info(f"Files with valid syntax: {valid_count} of {len(models_to_generate)}")
    logger.info(f"Results saved to {summary_path}")
    
    return summary

def get_missing_medium_priority_models() -> List[str]:
    """Return list of medium-priority models not yet implemented."""
    implemented = get_implemented_models()
    return [model for model in MEDIUM_PRIORITY_MODELS if model not in implemented]

def generate_missing_model_report(output_dir: str, include_medium: bool = True) -> None:
    """
    Generate a report of high-priority and optionally medium-priority models not yet implemented.
    
    Args:
        output_dir (str): Directory to save the report
        include_medium (bool): Whether to include medium-priority models in the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    missing_high_models = get_missing_models()
    missing_medium_models = get_missing_medium_priority_models() if include_medium else []
    implemented_models = get_implemented_models()
    
    # Create report content
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Model Implementation Status Report

Generated on: {current_time}

## High-Priority Models Status

- **Total High-Priority Models:** {len(HIGH_PRIORITY_MODELS)}
- **Implemented Models:** {len(set(HIGH_PRIORITY_MODELS).intersection(implemented_models))} ({len(set(HIGH_PRIORITY_MODELS).intersection(implemented_models)) / len(HIGH_PRIORITY_MODELS) * 100:.1f}%)
- **Missing Models:** {len(missing_high_models)} ({len(missing_high_models) / len(HIGH_PRIORITY_MODELS) * 100:.1f}%)

### Missing High-Priority Models

"""
    
    if missing_high_models:
        for model in missing_high_models:
            report += f"- [ ] {model}\n"
    else:
        report += "**✅ All high-priority models have been implemented!**\n"
    
    if include_medium:
        report += f"""
## Medium-Priority Models Status

- **Total Medium-Priority Models:** {len(MEDIUM_PRIORITY_MODELS)}
- **Implemented Models:** {len(set(MEDIUM_PRIORITY_MODELS).intersection(implemented_models))} ({len(set(MEDIUM_PRIORITY_MODELS).intersection(implemented_models)) / len(MEDIUM_PRIORITY_MODELS) * 100:.1f}%)
- **Missing Models:** {len(missing_medium_models)} ({len(missing_medium_models) / len(MEDIUM_PRIORITY_MODELS) * 100:.1f}%)

### Missing Medium-Priority Models

"""
        if missing_medium_models:
            for model in missing_medium_models:
                report += f"- [ ] {model}\n"
        else:
            report += "**✅ All medium-priority models have been implemented!**\n"
    
    # Add implementation instructions
    report += """
## Implementation Instructions

To implement these models:

1. Add the model to the `MODEL_REGISTRY` in `enhanced_generator.py`
2. Ensure the model is correctly classified in `ARCHITECTURE_TYPES`
3. Run `generate_priority_models.py` to generate the test files
4. Validate the generated files
5. Update this report

"""
    
    # Save report
    report_path = os.path.join(output_dir, "model_implementation_status.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Model implementation status report saved to {report_path}")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate tests for high-priority HuggingFace models")
    parser.add_argument("--output-dir", type=str, default="priority_model_tests",
                       help="Output directory for generated tests")
    parser.add_argument("--priority", type=str, choices=["high", "medium", "all"], default="high",
                       help="Priority level of models to generate tests for")
    parser.add_argument("--report", action="store_true", 
                       help="Generate a report of missing high-priority models")
    parser.add_argument("--report-dir", type=str, default="reports",
                       help="Directory to save the missing models report")
    
    args = parser.parse_args()
    
    if args.report:
        generate_missing_model_report(args.report_dir)
        return 0
    
    # Generate tests for priority models
    summary = generate_priority_model_tests(args.output_dir, args.priority)
    
    return 0 if summary["successful"] == summary["total"] else 1

if __name__ == "__main__":
    sys.exit(main())