#!/usr/bin/env python3
"""
Generate inference code for all 300+ HuggingFace model types.

This script generates the inference code for all model types with endpoint handlers
for CPU, CUDA, OpenVINO, MPS, QNN, and ROCm.
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import generator modules
from scripts.generators.reference_model_generator import ReferenceModelGenerator
from scripts.generators.architecture_detector import get_architecture_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_all_models_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Complete list of HuggingFace model types to generate
ALL_MODELS = [
    # CRITICAL PRIORITY
    "bert", "roberta", "gpt2", "t5", "llama", "mistral", 
    "vit", "clip", "whisper",
    
    # HIGH PRIORITY - Encoder-only models
    "albert", "deberta", "distilbert", "electra", "ernie", "mpnet", "xlm-roberta",
    
    # HIGH PRIORITY - Decoder-only models
    "bloom", "codellama", "falcon", "gemma", "gpt-j", "gpt-neo", "gpt-neox", "opt", "phi",
    
    # HIGH PRIORITY - Encoder-decoder models
    "bart", "mbart", "pegasus", "longt5", "mt5", "flan-t5",
    
    # HIGH PRIORITY - Vision models
    "beit", "convnext", "deit", "mobilenet-v2", "swin",
    
    # HIGH PRIORITY - Vision-text models
    "blip", "blip-2", "chinese-clip", "git", "llava",
    
    # HIGH PRIORITY - Speech models
    "hubert", "wav2vec2", "bark",
    
    # MEDIUM PRIORITY - Encoder-only models
    "camembert", "canine", "esm", "flaubert", "layoutlm", "luke", "rembert", "roformer", "splinter",
    
    # MEDIUM PRIORITY - Decoder-only models
    "biogpt", "ctrl", "gptj", "mpt", "persimmon", "qwen", "rwkv", "stablelm",
    
    # MEDIUM PRIORITY - Encoder-decoder models
    "bigbird", "fsmt", "led", "longt5", "m2m-100", "pegasus-x", "prophetnet", "switch-transformers",
    
    # MEDIUM PRIORITY - Vision models
    "convnextv2", "data2vec-vision", "detr", "dinov2", "efficientnet", "mobilevit", "segformer", "yolos",
    
    # MEDIUM PRIORITY - Vision-text models
    "clipseg", "donut", "flava", "idefics", "kosmos-2", "owlvit", "paligemma", "vilt", "xclip",
    
    # MEDIUM PRIORITY - Speech models
    "data2vec-audio", "encodec", "seamless-m4t", "speecht5", "sew", "unispeech", "whisper-tiny",
    
    # LOW PRIORITY - Remaining models
    "align", "audio-spectrogram-transformer", "autoformer", "barthez", "bartpho",
    "beit3", "bertweet", "big_bird", "bigbird_pegasus", "biogpt", "bit", "blenderbot", 
    "blenderbot-small", "bridgetower", "bros", "clap", "clvp", "cm3", "codegen", 
    "conditional-detr", "convbert", "cpm", "cvt", "data2vec", "decision-transformer", 
    "deta", "dialogpt", "dinat", "dino", "distilroberta", "dpr", "dpt", "efficientformer", 
    "fnet", "focalnet", "funnel", "gptsan-japanese", "herbert", "ibert", "jukebox", 
    "layoutlmv2", "layoutlmv3", "levit", "lilt", "longformer", "lxmert", "marian", 
    "markuplm", "mask2former", "maskformer", "mbart50", "mega", "megatron-bert", 
    "mlp-mixer", "mobilebert", "musicgen", "nezha", "nllb", "nllb-moe", "nougat", 
    "nystromformer", "owlv2", "patchtst", "perceiver", "pix2struct", "plbart", 
    "poolformer", "pop2piano", "pvt", "pvt-v2", "qdqbert", "reformer", "regnet", 
    "resnet", "retribert", "roberta-prelayernorm", "roc-bert", "sam", "sew-d", 
    "speech-encoder-decoder", "speech-to-text", "speech-to-text-2", "squeezebert", 
    "swin2sr", "swinv2", "table-transformer", "tapas", "time-series-transformer", 
    "timesformer", "trajectory-transformer", "transfo-xl", "trocr", "tvlt", "tvp", 
    "udop", "univnet", "upernet", "van", "videomae", "vision-encoder-decoder", 
    "vision-text-dual-encoder", "visual-bert", "vit-mae", "vit-msn", "vitdet", 
    "vitmatte", "vits", "vivit", "wav2vec2-bert", "wav2vec2-conformer", "wavlm", 
    "xglm", "xlm", "xlm-prophetnet", "xlnet", "xmod", "yoso"
]

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate inference code for 300+ HuggingFace models with hardware-specific handlers"
    )
    
    # Model selection options
    model_group = parser.add_argument_group("Model Selection")
    model_selection = model_group.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--model", "-m", type=str,
        help="Specific model to generate (e.g., 'bert', 'gpt2')"
    )
    model_selection.add_argument(
        "--priority", "-p", type=str, default="critical",
        choices=["critical", "high", "medium", "low", "all"],
        help="Generate models with the specified priority level"
    )
    model_selection.add_argument(
        "--from-file", "-f", type=str,
        help="Read model names from the specified file (one per line)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", "-o", type=str, default="../../test/skillset",
        help="Directory to write generated skillsets to"
    )
    output_group.add_argument(
        "--force", action="store_true",
        help="Force overwrite of existing files"
    )
    output_group.add_argument(
        "--no-verify", action="store_true",
        help="Skip verification of generated files"
    )
    output_group.add_argument(
        "--summary-file", "-s", type=str, default="generation_summary.json",
        help="Write generation summary to the specified JSON file"
    )
    
    # Template options
    template_group = parser.add_argument_group("Template Options")
    template_group.add_argument(
        "--template-dir", "-t", type=str,
        help="Directory containing template files"
    )
    
    return parser.parse_args()

def get_models_from_file(file_path: str) -> List[str]:
    """
    Read model names from a file.
    
    Args:
        file_path: Path to file containing model names (one per line).
        
    Returns:
        List of model names.
    """
    try:
        with open(file_path, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Read {len(models)} models from {file_path}")
        return models
    except Exception as e:
        logger.error(f"Error reading model list from {file_path}: {e}")
        return []

def get_models_by_priority(priority: str) -> List[str]:
    """
    Get models based on priority level.
    
    Args:
        priority: Priority level ("critical", "high", "medium", "low", "all").
        
    Returns:
        List of model names.
    """
    if priority == "all":
        return ALL_MODELS
    
    # Critical models
    if priority == "critical":
        return ["bert", "roberta", "gpt2", "t5", "llama", "mistral", "vit", "clip", "whisper"]
    
    # High priority includes critical
    if priority == "high":
        high_priority = ["bert", "roberta", "gpt2", "t5", "llama", "mistral", "vit", "clip", "whisper",
                         "albert", "deberta", "distilbert", "electra", "ernie", "mpnet", "xlm-roberta",
                         "bloom", "codellama", "falcon", "gemma", "gpt-j", "gpt-neo", "gpt-neox", "opt", "phi",
                         "bart", "mbart", "pegasus", "longt5", "mt5", "flan-t5",
                         "beit", "convnext", "deit", "mobilenet-v2", "swin",
                         "blip", "blip-2", "chinese-clip", "git", "llava",
                         "hubert", "wav2vec2", "bark"]
        return high_priority
    
    # Medium priority includes high and critical
    if priority == "medium":
        medium_priority = ["bert", "roberta", "gpt2", "t5", "llama", "mistral", "vit", "clip", "whisper",
                           "albert", "deberta", "distilbert", "electra", "ernie", "mpnet", "xlm-roberta",
                           "bloom", "codellama", "falcon", "gemma", "gpt-j", "gpt-neo", "gpt-neox", "opt", "phi",
                           "bart", "mbart", "pegasus", "longt5", "mt5", "flan-t5",
                           "beit", "convnext", "deit", "mobilenet-v2", "swin",
                           "blip", "blip-2", "chinese-clip", "git", "llava",
                           "hubert", "wav2vec2", "bark"]
        # Add medium priority models
        medium_priority.extend([
            "camembert", "canine", "esm", "flaubert", "layoutlm", "luke", "rembert", "roformer", "splinter",
            "biogpt", "ctrl", "gptj", "mpt", "persimmon", "qwen", "rwkv", "stablelm",
            "bigbird", "fsmt", "led", "longt5", "m2m-100", "pegasus-x", "prophetnet", "switch-transformers",
            "convnextv2", "data2vec-vision", "detr", "dinov2", "efficientnet", "mobilevit", "segformer", "yolos",
            "clipseg", "donut", "flava", "idefics", "kosmos-2", "owlvit", "paligemma", "vilt", "xclip",
            "data2vec-audio", "encodec", "seamless-m4t", "speecht5", "sew", "unispeech", "whisper-tiny"
        ])
        return list(set(medium_priority))  # Remove duplicates
    
    # Low priority includes all models
    if priority == "low":
        return ALL_MODELS
    
    # Default to critical
    logger.warning(f"Unknown priority '{priority}', defaulting to critical")
    return ["bert", "roberta", "gpt2", "t5", "llama", "mistral", "vit", "clip", "whisper"]

def generate_models(args):
    """
    Generate skillset code for selected models.
    
    Args:
        args: Parsed command-line arguments.
    """
    # Create generator
    generator = ReferenceModelGenerator(args.template_dir, args.output_dir)
    
    # Determine which models to generate
    models_to_generate = []
    
    if args.model:
        # Single model
        models_to_generate = [args.model]
        logger.info(f"Generating reference implementation for model: {args.model}")
    
    elif args.from_file:
        # Models from file
        models_to_generate = get_models_from_file(args.from_file)
        logger.info(f"Generating reference implementations for {len(models_to_generate)} models from file")
    
    else:
        # Generate by priority
        models_to_generate = get_models_by_priority(args.priority)
        logger.info(f"Generating reference implementations for {len(models_to_generate)} {args.priority} models")
    
    # Process each model
    total_generated = 0
    total_failed = 0
    results = {}
    
    for i, model in enumerate(models_to_generate):
        logger.info(f"Generating model {i+1}/{len(models_to_generate)}: {model}")
        
        try:
            success, files = generator.generate_reference_implementation(
                model, args.force, not args.no_verify
            )
            
            results[model] = {
                "success": success,
                "files": files
            }
            
            if success:
                total_generated += 1
            else:
                total_failed += 1
                
        except Exception as e:
            logger.error(f"Error generating {model}: {e}")
            results[model] = {
                "success": False,
                "error": str(e),
                "files": []
            }
            total_failed += 1
    
    # Print summary
    logger.info(f"Generation completed.")
    logger.info(f"Successfully generated: {total_generated}/{len(models_to_generate)}")
    logger.info(f"Failed: {total_failed}/{len(models_to_generate)}")
    
    # Write summary to file
    if args.summary_file:
        try:
            import json
            with open(os.path.join(args.output_dir, args.summary_file), 'w') as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_models": len(models_to_generate),
                    "successful": total_generated,
                    "failed": total_failed,
                    "results": results
                }, f, indent=2)
            logger.info(f"Summary written to {os.path.join(args.output_dir, args.summary_file)}")
        except Exception as e:
            logger.error(f"Error writing summary: {e}")
    
    return total_generated, total_failed, results

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print banner
    print("""
    ╭────────────────────────────────────────────────────────╮
    │                                                        │
    │   HuggingFace Inference Generator                      │
    │                                                        │
    │   Generates model implementations for 300+ models      │
    │   with handlers for CPU, CUDA, OpenVINO, MPS, QNN      │
    │                                                        │
    ╰────────────────────────────────────────────────────────╯
    """)
    
    # Generate models
    total_generated, total_failed, results = generate_models(args)
    
    # Exit code based on success
    if total_failed == 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())