#!/usr/bin/env python3
"""
Generate inference skillsets for all 300+ HuggingFace model types.

This script generates the inference code for all model types with endpoint handlers
for CPU, CUDA, OpenVINO, MPS, QNN, and ROCm.
"""

import os
import sys
import time
import json
import logging
import argparse
import concurrent.futures
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_simple_model import generate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_all_skillsets_{time.strftime('%Y%m%d_%H%M%S')}.log")
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
        description="Generate inference skillsets for 300+ HuggingFace models with hardware-specific handlers"
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
        "--output-dir", "-o", type=str, default="../ipfs_accelerate_py/worker/skillset",
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
    
    # Parallel processing
    parser.add_argument(
        "--parallel", action="store_true", 
        help="Generate models in parallel for faster processing"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum number of worker threads when using parallel generation"
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

def generate_model_wrapper(model_type, output_dir, force=False):
    """Wrapper for generate_model to handle exceptions for parallel processing."""
    try:
        file_path = os.path.join(output_dir, f"hf_{model_type}.py")
        
        # Skip if file exists and force is False
        if os.path.exists(file_path) and not force:
            logger.info(f"Skipping {model_type} (file exists)")
            return model_type, True, file_path, None
        
        # Generate the model
        file_path = generate_model(model_type, output_dir)
        
        # Verify syntax
        try:
            import py_compile
            py_compile.compile(file_path, doraise=True)
            return model_type, True, file_path, None
        except Exception as e:
            return model_type, False, file_path, f"Syntax error: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error generating {model_type}: {e}")
        return model_type, False, None, str(e)

def generate_models(args):
    """
    Generate skillset code for selected models.
    
    Args:
        args: Parsed command-line arguments.
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Generate the models
    results = {}
    
    if args.parallel and len(models_to_generate) > 1:
        # Parallel generation
        logger.info(f"Using parallel generation with {args.max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_model = {
                executor.submit(generate_model_wrapper, model, args.output_dir, args.force): model 
                for model in models_to_generate
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    model_type, success, file_path, error = future.result()
                    results[model_type] = {
                        "success": success,
                        "file": file_path,
                        "error": error
                    }
                    
                    if success:
                        logger.info(f"Successfully generated: {model}")
                    else:
                        logger.error(f"Failed to generate {model}: {error}")
                        
                except Exception as e:
                    logger.error(f"Exception for {model}: {e}")
                    results[model] = {
                        "success": False,
                        "file": None,
                        "error": str(e)
                    }
    else:
        # Sequential generation
        for model in models_to_generate:
            model_type, success, file_path, error = generate_model_wrapper(model, args.output_dir, args.force)
            results[model_type] = {
                "success": success,
                "file": file_path,
                "error": error
            }
            
            if success:
                logger.info(f"Successfully generated: {model}")
            else:
                logger.error(f"Failed to generate {model}: {error}")
    
    # Calculate statistics
    total_models = len(models_to_generate)
    successful = sum(1 for model in results.values() if model["success"])
    failed = total_models - successful
    
    logger.info(f"Generation completed.")
    logger.info(f"Successfully generated: {successful}/{total_models}")
    logger.info(f"Failed: {failed}/{total_models}")
    
    # Write summary to file
    if args.summary_file:
        try:
            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_models": total_models,
                "successful": successful,
                "failed": failed,
                "results": results
            }
            
            with open(os.path.join(args.output_dir, args.summary_file), 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Summary written to {os.path.join(args.output_dir, args.summary_file)}")
        except Exception as e:
            logger.error(f"Error writing summary: {e}")
    
    return successful, failed, results

def main():
    """Main entry point."""
    args = parse_args()
    
    # Print banner
    print("""
    ╭────────────────────────────────────────────────────────╮
    │                                                        │
    │   HuggingFace Inference Skillset Generator             │
    │                                                        │
    │   Generates model implementations for 300+ models      │
    │   with handlers for CPU, CUDA, OpenVINO, MPS, QNN      │
    │                                                        │
    ╰────────────────────────────────────────────────────────╯
    """)
    
    # Generate models
    successful, failed, results = generate_models(args)
    
    # Exit code based on success
    if failed == 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())