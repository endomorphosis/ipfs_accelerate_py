#!/usr/bin/env python3
"""
Generate inference skillsets for all Hugging Face Transformers classes.

This script generates Python implementations for all Hugging Face Transformers model classes,
with each class having endpoint handlers for CPU, CUDA, OpenVINO, MPS, QNN, and ROCm backends.
"""

import os
import sys
import logging
import json
import argparse
from typing import Dict, List, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generate_huggingface_skillset.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from templates.template_composer import TemplateComposer

# Import all architecture templates
from templates.encoder_only import EncoderOnlyArchitectureTemplate
from templates.decoder_only import DecoderOnlyArchitectureTemplate
from templates.encoder_decoder import EncoderDecoderArchitectureTemplate
from templates.vision import VisionArchitectureTemplate
from templates.vision_text import VisionTextArchitectureTemplate
from templates.speech import SpeechArchitectureTemplate
from templates.multimodal import MultimodalArchitectureTemplate
from templates.diffusion import DiffusionArchitectureTemplate
from templates.moe import MoEArchitectureTemplate
from templates.state_space import StateSpaceArchitectureTemplate
from templates.rag import RAGArchitectureTemplate

# Import all hardware templates
from templates.cpu_hardware import CPUHardwareTemplate
from templates.cuda_hardware import CudaHardwareTemplate
from templates.openvino_hardware import OpenvinoHardwareTemplate
from templates.apple_hardware import AppleHardwareTemplate
from templates.qualcomm_hardware import QualcommHardwareTemplate
from templates.rocm_hardware import RocmHardwareTemplate

# Import all pipeline templates
from templates.base_pipeline import TextPipelineTemplate
try:
    from templates.image_pipeline import ImagePipelineTemplate
except ImportError:
    # Create a simple mock
    from templates.base_pipeline import BasePipelineTemplate
    class ImagePipelineTemplate(BasePipelineTemplate):
        """Temporary mock for image pipeline template."""
        def __init__(self):
            super().__init__()
            self.pipeline_type = "image"
            self.input_type = "image"
            self.output_type = "image"
            
        def get_import_statements(self):
            return "# Image pipeline imports"
            
        def get_preprocessing_code(self, task_type):
            return "# Image preprocessing code"
            
        def get_postprocessing_code(self, task_type):
            return "# Image postprocessing code"
            
        def get_result_formatting_code(self, task_type):
            return "# Image result formatting code"
            
        def is_compatible_with_architecture(self, arch_type):
            return arch_type in ["vision"]

from templates.vision_text_pipeline import VisionTextPipelineTemplate
from templates.audio_pipeline import AudioPipelineTemplate
from templates.multimodal_pipeline import MultimodalPipelineTemplate
from templates.diffusion_pipeline import DiffusionPipelineTemplate
from templates.moe_pipeline import MoEPipelineTemplate
from templates.state_space_pipeline import StateSpacePipelineTemplate
from templates.rag_pipeline import RAGPipelineTemplate


def get_huggingface_models() -> List[Dict[str, str]]:
    """
    Get the list of Hugging Face models to generate.
    
    Returns:
        List of dictionaries with model name and architecture type.
    """
    # Define the model architecture mapping
    architecture_models = {
        "encoder-only": [
            "bert-base-uncased", "roberta-base", "albert-base-v2", "distilbert-base-uncased",
            "xlm-roberta-base", "electra-small-discriminator", "layoutlm-base-uncased",
            "deberta-base", "deberta-v2-xlarge", "mpnet-base", "ernie-2.0-base-en",
            "flaubert/flaubert_base_cased", "microsoft/deberta-base", "facebook/bart-base",
            "sentence-transformers/all-MiniLM-L6-v2", "joeddav/xlm-roberta-large-xnli"
        ],
        "decoder-only": [
            "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "bigscience/bloom-560m", 
            "facebook/opt-350m", "facebook/opt-1.3b", "EleutherAI/gpt-neo-125m",
            "EleutherAI/gpt-j-6b", "bigscience/bloom-1b7", "bigscience/bloomz-1b7",
            "meta-llama/Llama-2-7b-hf", "codellama/CodeLlama-7b-hf", "stabilityai/stablelm-base-alpha-7b",
            "mistralai/Mistral-7B-v0.1", "microsoft/phi-2", "meta-llama/Llama-2-7b-chat-hf"
        ],
        "encoder-decoder": [
            "t5-small", "t5-base", "facebook/bart-large", "facebook/bart-large-cnn",
            "google/pegasus-xsum", "google/flan-t5-small", "facebook/mbart-large-50",
            "google/mt5-small", "google/pegasus-cnn_dailymail", "microsoft/prophetnet-large-uncased",
            "facebook/mbart-large-cc25", "google/pegasus-large", "google/flan-t5-base",
            "allenai/led-base-16384", "google/mt5-base", "facebook/nllb-200-distilled-600M"
        ],
        "vision": [
            "google/vit-base-patch16-224", "microsoft/beit-base-patch16-224", 
            "facebook/deit-base-distilled-patch16-224", "facebook/convnext-base-224",
            "microsoft/swin-base-patch4-window7-224", "google/vit-base-patch32-384",
            "facebook/deit-small-patch16-224", "facebook/convnext-tiny-224",
            "microsoft/swin-tiny-patch4-window7-224", "microsoft/dit-base-finetuned-rvlcdip",
            "facebook/levit-128S", "facebook/regnet-y-040", "facebook/dino-vitb16",
            "microsoft/resnet-50", "facebook/dinov2-base", "facebook/dino-vits16"
        ],
        "vision-encoder-text-decoder": [
            "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14", 
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "Salesforce/blip-vqa-base",
            "Salesforce/blip-image-captioning-base", "Salesforce/blip-vqa-capfilt-large",
            "Salesforce/blip2-opt-2.7b", "Salesforce/blip2-opt-6.7b", "llava-hf/llava-1.5-7b-hf"
        ],
        "speech": [
            "openai/whisper-tiny", "openai/whisper-small", "openai/whisper-base", 
            "facebook/wav2vec2-base", "facebook/wav2vec2-large", "facebook/wav2vec2-large-960h",
            "facebook/hubert-base-ls960", "facebook/hubert-large-ll60k", "microsoft/speecht5_asr",
            "facebook/mms-1b-all", "facebook/encodec_24khz", "facebook/mms-300m-fleurs",
            "microsoft/wavlm-base", "facebook/data2vec-audio-base", "facebook/s2t-small-librispeech-asr"
        ],
        "multimodal": [
            "facebook/flava-full", "facebook/imagebind-huge", "llava-hf/llava-1.5-7b-hf",
            "microsoft/git-base", "facebook/musicgen-small", "facebook/dbrx-base",
            "facebook/musicgen-melody", "facebook/musicgen-large", "facebook/musicgen-stereo-small",
            "stabilityai/stablecode-completion-alpha-3b", "AdamG012/chat-imagebind-huge-vicuna-13b"
        ],
        "diffusion": [
            "runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1", 
            "CompVis/ldm-text2im-large-256", "stabilityai/stable-diffusion-xl-base-1.0",
            "kandinsky-community/kandinsky-2-1", "stabilityai/stable-diffusion-2-inpainting",
            "lambdalabs/sd-image-variations-diffusers", "CompVis/stable-diffusion-v1-4",
            "frankjoshua/icbinv_style_icons", "facebook/maskformer-swin-base-ade",
            "facebook/sam-vit-base", "facebook/sam-vit-huge"
        ],
        "mixture-of-experts": [
            "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "google/switch-base-8", "google/switch-base-16", "google/switch-base-32",
            "google/switch-large-128", "google/gshard-gpt2", "mistralai/Mixtral-8x22B-v0.1",
            "moe-large-2t-alpaca", "bigscience/bloom-moe"
        ],
        "state-space": [
            "state-spaces/mamba-130m", "state-spaces/mamba-370m", 
            "state-spaces/mamba-790m", "state-spaces/mamba-1.4b", "state-spaces/mamba-2.8b",
            "state-spaces/mamba2-2.8b", "RWKV/rwkv-4-430m-pile", "RWKV/rwkv-4-3b-pile",
            "RWKV/rwkv-4-7b-pile", "RWKV/rwkv-4-raven", "RWKV/rwkv-5-world"
        ],
        "rag": [
            "facebook/rag-token-base", "facebook/rag-token-nq", 
            "facebook/rag-sequence-base", "facebook/rag-sequence-nq",
            "colbert-ir/colbertv2.0", "rag-fusion-dense"
        ]
    }
    
    models = []
    for arch_type, model_list in architecture_models.items():
        for model_name in model_list:
            models.append({
                "name": model_name,
                "type": arch_type
            })
    
    return models


def generate_model_implementations(
    output_dir: str,
    models: List[Dict[str, str]],
    hardware_types: List[str] = None,
    batch_size: int = 10,
    max_workers: int = 4,
    force: bool = False
) -> Dict[str, Any]:
    """
    Generate model implementations for all specified models.
    
    Args:
        output_dir: Directory to save generated implementations
        models: List of models to generate
        hardware_types: List of hardware backends to generate handlers for
        batch_size: Number of models to process in each batch
        max_workers: Maximum number of worker threads
        force: Whether to overwrite existing files
        
    Returns:
        Dictionary with generation statistics
    """
    if not hardware_types:
        hardware_types = ["cpu", "cuda", "openvino", "mps", "qnn", "rocm"]
    
    # Initialize templates
    architecture_templates = {
        "encoder-only": EncoderOnlyArchitectureTemplate(),
        "decoder-only": DecoderOnlyArchitectureTemplate(),
        "encoder-decoder": EncoderDecoderArchitectureTemplate(),
        "vision": VisionArchitectureTemplate(),
        "vision-encoder-text-decoder": VisionTextArchitectureTemplate(),
        "speech": SpeechArchitectureTemplate(),
        "multimodal": MultimodalArchitectureTemplate(),
        "diffusion": DiffusionArchitectureTemplate(),
        "mixture-of-experts": MoEArchitectureTemplate(),
        "state-space": StateSpaceArchitectureTemplate(),
        "rag": RAGArchitectureTemplate()
    }
    
    hardware_templates = {
        "cpu": CPUHardwareTemplate(),
        "cuda": CudaHardwareTemplate(),
        "openvino": OpenvinoHardwareTemplate(),
        "mps": AppleHardwareTemplate(),
        "qnn": QualcommHardwareTemplate(),
        "rocm": RocmHardwareTemplate()
    }
    
    pipeline_templates = {
        "text": TextPipelineTemplate(),
        "image": ImagePipelineTemplate(),
        "vision-text": VisionTextPipelineTemplate(),
        "audio": AudioPipelineTemplate(),
        "multimodal": MultimodalPipelineTemplate(),
        "diffusion": DiffusionPipelineTemplate(),
        "moe": MoEPipelineTemplate(),
        "state-space": StateSpacePipelineTemplate(),
        "rag": RAGPipelineTemplate()
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create template composer
    composer = TemplateComposer(
        hardware_templates={k: v for k, v in hardware_templates.items() if k in hardware_types},
        architecture_templates=architecture_templates,
        pipeline_templates=pipeline_templates,
        output_dir=output_dir
    )
    
    # Process models in batches
    total_models = len(models)
    success_count = 0
    failure_count = 0
    results = []
    
    logger.info(f"Generating implementations for {total_models} models with "
                f"{len(hardware_types)} hardware backends...")
    
    # Process models in batches
    for batch_start in range(0, total_models, batch_size):
        batch_end = min(batch_start + batch_size, total_models)
        batch = models[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_models+batch_size-1)//batch_size} "
                    f"({batch_end-batch_start} models)")
        
        # Generate implementations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(
                    generate_single_model,
                    composer=composer,
                    model=model,
                    hardware_types=hardware_types,
                    force=force
                ): model for model in batch
            }
            
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["success"]:
                        success_count += 1
                        logger.info(f"✅ Successfully generated {model['type']} implementation for {model['name']}")
                    else:
                        failure_count += 1
                        logger.error(f"❌ Failed to generate {model['type']} implementation for {model['name']}: {result['error']}")
                        
                except Exception as e:
                    failure_count += 1
                    logger.error(f"❌ Exception generating implementation for {model['name']}: {str(e)}")
        
        # Log progress
        logger.info(f"Batch complete. Progress: {batch_end}/{total_models} models "
                    f"({success_count} successes, {failure_count} failures)")
    
    # Generate summary
    unique_architectures = set(model["type"] for model in models)
    processed_architectures = set(result["architecture"] for result in results if result["success"])
    
    summary = {
        "total_models": total_models,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": f"{(success_count / total_models) * 100:.1f}%",
        "architecture_coverage": f"{len(processed_architectures)}/{len(unique_architectures)}",
        "hardware_types": hardware_types,
        "results": results
    }
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "generation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Generation summary saved to {summary_file}")
    logger.info(f"Generated {success_count}/{total_models} models successfully "
                f"({(success_count/total_models)*100:.1f}%)")
    
    return summary


def generate_single_model(
    composer: TemplateComposer,
    model: Dict[str, str],
    hardware_types: List[str],
    force: bool = False
) -> Dict[str, Any]:
    """
    Generate implementation for a single model.
    
    Args:
        composer: Template composer instance
        model: Model information (name and type)
        hardware_types: List of hardware backends
        force: Whether to overwrite existing files
        
    Returns:
        Dictionary with generation result
    """
    model_name = model["name"]
    arch_type = model["type"]
    
    try:
        success, output_file = composer.generate_model_implementation(
            model_name=model_name,
            arch_type=arch_type,
            hardware_types=hardware_types,
            force=force
        )
        
        result = {
            "model": model_name,
            "architecture": arch_type,
            "success": success,
            "output_file": output_file if success else None
        }
        
        if success:
            # Verify file size
            file_size = os.path.getsize(output_file)
            result["file_size"] = file_size
            
            if file_size < 10000:
                result["warning"] = f"File is suspiciously small: {file_size} bytes"
        else:
            result["error"] = f"Failed to generate implementation for {model_name}"
            
        return result
    
    except Exception as e:
        return {
            "model": model_name,
            "architecture": arch_type,
            "success": False,
            "error": str(e)
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Hugging Face model implementations")
    
    parser.add_argument("--output-dir", type=str, default="huggingface_skillsets",
                        help="Directory to save generated implementations")
    
    parser.add_argument("--hardware", type=str, nargs="+", 
                        default=["cpu", "cuda", "openvino", "mps", "qnn", "rocm"],
                        help="Hardware backends to generate handlers for")
    
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of models to process in each batch")
    
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum number of worker threads")
    
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Get models to generate
    models = get_huggingface_models()
    logger.info(f"Loaded {len(models)} Hugging Face models across {len(set(m['type'] for m in models))} architectures")
    
    # Generate implementations
    summary = generate_model_implementations(
        output_dir=args.output_dir,
        models=models,
        hardware_types=args.hardware,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        force=args.force
    )
    
    # Generate markdown report
    report_file = os.path.join(args.output_dir, "GENERATION_REPORT.md")
    with open(report_file, 'w') as f:
        f.write("# Hugging Face Model Implementation Generation Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total models processed: {summary['total_models']}\n")
        f.write(f"- Successfully generated: {summary['success_count']} ({summary['success_rate']})\n")
        f.write(f"- Failed: {summary['failure_count']}\n")
        f.write(f"- Architecture coverage: {summary['architecture_coverage']}\n")
        f.write(f"- Hardware backends: {', '.join(summary['hardware_types'])}\n\n")
        
        f.write("## Architecture Statistics\n\n")
        f.write("| Architecture Type | Total Models | Generated | Success Rate |\n")
        f.write("|-------------------|-------------|-----------|-------------|\n")
        
        arch_stats = {}
        for model in models:
            arch = model["type"]
            if arch not in arch_stats:
                arch_stats[arch] = {"total": 0, "success": 0}
            arch_stats[arch]["total"] += 1
        
        for result in summary["results"]:
            if result["success"]:
                arch_stats[result["architecture"]]["success"] += 1
                
        for arch, stats in sorted(arch_stats.items()):
            success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            f.write(f"| {arch} | {stats['total']} | {stats['success']} | {success_rate:.1f}% |\n")
            
        f.write("\n## Hardware Support\n\n")
        f.write("The following hardware backends are supported for each model:\n\n")
        f.write(f"- {', '.join(summary['hardware_types'])}\n\n")
        
        if summary["failure_count"] > 0:
            f.write("## Failed Models\n\n")
            f.write("| Model | Architecture |\n")
            f.write("|-------|-------------|\n")
            
            for result in summary["results"]:
                if not result["success"]:
                    f.write(f"| {result['model']} | {result['architecture']} |\n")
    
    logger.info(f"Report saved to {report_file}")
    logger.info("Generation complete!")


if __name__ == "__main__":
    main()