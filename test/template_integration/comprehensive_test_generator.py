#!/usr/bin/env python3
"""
Comprehensive Test Generator for HuggingFace Transformers models.

This script provides a single entry point for generating test files for all
HuggingFace Transformers model classes, ensuring complete coverage of
from_pretrained and pipeline methods.
"""

import os
import sys
import argparse
import logging
import json
import importlib
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple, Union
import concurrent.futures

# Configure logging
log_filename = f"comprehensive_test_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

# Import template utilities
try:
    from template_integration.template_integration_workflow import generate_test_file
    from template_integration.generate_refactored_test import (
        determine_architecture, MODEL_ARCHITECTURE_MAPPING
    )
except ImportError as e:
    logger.error(f"Could not import required modules: {e}")
    sys.exit(1)

# Define model architecture categories
ARCHITECTURE_CATEGORIES = {
    "vision": [
        "ViT", "DeiT", "BEiT", "ConvNeXT", "DINOv2", "Swin", "ConvNext",
        "SegFormer", "DETR", "YOLOS", "Mask2Former", "SAM"
    ],
    "encoder_only": [
        "BERT", "RoBERTa", "ALBERT", "ELECTRA", "DistilBERT", "XLM", "XLMR",
        "FNet", "ERNIE", "RemBERT", "ProphetNet"
    ],
    "decoder_only": [
        "GPT2", "OPT", "GPTNeo", "GPTJ", "LLaMA", "Qwen", "TransfoXL"
    ],
    "encoder_decoder": [
        "T5", "BART", "FlanT5", "MT5", "mBART", "LED", "PEGASUS", "Marian"
    ],
    "speech": [
        "Whisper", "Wav2Vec2", "HuBERT", "SEW", "EnCodec", "Data2VecAudio",
        "CLAP", "MusicGen", "UniSpeech"
    ],
    "multimodal": [
        "CLIP", "BLIP", "BLIP2", "LLaVA", "FLAVA", "GIT", "IDEFICS", "ImageBind",
        "ViLT", "XCLIP", "PaliGemma"
    ]
}

# Mapping from model class to recommended models for testing
MODEL_CLASS_MAPPING = {
    # Vision models
    "ViTModel": "google/vit-base-patch16-224",
    "DeiTModel": "facebook/deit-base-patch16-224",
    "BeitModel": "microsoft/beit-base-patch16-224",
    "ConvNextModel": "facebook/convnext-base-224-22k",
    "Dinov2Model": "facebook/dinov2-base",
    "SwinModel": "microsoft/swin-base-patch4-window7-224",
    "SegformerModel": "nvidia/segformer-b0-finetuned-ade-512-512",
    "DetrModel": "facebook/detr-resnet-50",
    "YolosModel": "hustvl/yolos-small",
    "Mask2FormerModel": "facebook/mask2former-swin-base-coco-instance",
    "SamModel": "facebook/sam-vit-base",
    
    # Encoder-only models
    "BertModel": "bert-base-uncased",
    "RobertaModel": "roberta-base",
    "AlbertModel": "albert-base-v2",
    "ElectraModel": "google/electra-base-discriminator",
    "DistilBertModel": "distilbert-base-uncased",
    "XLMModel": "xlm-mlm-en-2048",
    "XLMRobertaModel": "xlm-roberta-base",
    "FNetModel": "google/fnet-base",
    "ErnieModel": "ernie-health-chinese",
    "RemBertModel": "google/rembert",
    "ProphetNetModel": "microsoft/prophetnet-large-uncased",
    
    # Decoder-only models
    "GPT2Model": "gpt2",
    "OPTModel": "facebook/opt-350m",
    "GPTNeoModel": "EleutherAI/gpt-neo-125m",
    "GPTJModel": "EleutherAI/gpt-j-6b",
    "LlamaModel": "meta-llama/Llama-2-7b-hf",
    "Qwen2Model": "Qwen/Qwen2-7B-Instruct",
    "TransfoXLModel": "transfo-xl-wt103",
    
    # Encoder-decoder models
    "T5Model": "t5-base",
    "BartModel": "facebook/bart-base",
    "MT5Model": "google/mt5-base",
    "MBartModel": "facebook/mbart-large-50",
    "LEDModel": "allenai/led-base-16384",
    "PegasusModel": "google/pegasus-xsum",
    "MarianModel": "Helsinki-NLP/opus-mt-en-de",
    
    # Speech models
    "WhisperModel": "openai/whisper-tiny",
    "Wav2Vec2Model": "facebook/wav2vec2-base-960h",
    "HubertModel": "facebook/hubert-base-ls960",
    "SEWModel": "asapp/sew-mid-100k",
    "EncodecModel": "facebook/encodec_24khz",
    "Data2VecAudioModel": "facebook/data2vec-audio-base-960h",
    "ClapModel": "laion/clap-htsat-unfused",
    "MusicgenModel": "facebook/musicgen-small",
    "UniSpeechModel": "microsoft/unispeech-sat-base",
    
    # Multimodal models
    "CLIPModel": "openai/clip-vit-base-patch32",
    "BlipModel": "Salesforce/blip-image-captioning-base",
    "BlipForConditionalGeneration": "Salesforce/blip-image-captioning-base",
    "BlipForQuestionAnswering": "Salesforce/blip-vqa-base",
    "BlipForImageTextRetrieval": "Salesforce/blip-itm-base-coco",
    "Blip2Model": "Salesforce/blip2-opt-2.7b",
    "LlavaModel": "llava-hf/llava-1.5-7b-hf",
    "FlavaModel": "facebook/flava-full",
    "GitModel": "microsoft/git-base",
    "IdeficsModel": "HuggingFaceM4/idefics-9b",
    "ImagebindModel": "facebook/imagebind-huge",
    "ViltModel": "dandelin/vilt-b32-mlm",
    "XClipModel": "microsoft/xclip-base-patch32",
    "PaliGemmaModel": "google/paligemma-3b"
}

# Task mapping for pipeline testing
PIPELINE_TASK_MAPPING = {
    # Vision tasks
    "ViTModel": "image-classification",
    "DeiTModel": "image-classification",
    "BeitModel": "image-classification",
    "ConvNextModel": "image-classification",
    "Dinov2Model": "image-classification",
    "SwinModel": "image-classification",
    "SegformerModel": "image-segmentation",
    "DetrModel": "object-detection",
    "YolosModel": "object-detection",
    "Mask2FormerModel": "image-segmentation",
    "SamModel": "image-segmentation",
    
    # Text tasks
    "BertModel": "fill-mask",
    "RobertaModel": "fill-mask",
    "AlbertModel": "fill-mask",
    "ElectraModel": "fill-mask",
    "DistilBertModel": "fill-mask",
    "XLMModel": "fill-mask",
    "XLMRobertaModel": "fill-mask",
    "FNetModel": "fill-mask",
    "ErnieModel": "fill-mask",
    "RemBertModel": "fill-mask",
    "ProphetNetModel": "text-generation",
    
    # Text generation tasks
    "GPT2Model": "text-generation",
    "OPTModel": "text-generation",
    "GPTNeoModel": "text-generation",
    "GPTJModel": "text-generation",
    "LlamaModel": "text-generation",
    "Qwen2Model": "text-generation",
    "TransfoXLModel": "text-generation",
    
    # Encoder-decoder tasks
    "T5Model": "text2text-generation",
    "BartModel": "summarization",
    "MT5Model": "text2text-generation",
    "MBartModel": "translation",
    "LEDModel": "summarization",
    "PegasusModel": "summarization",
    "MarianModel": "translation",
    
    # Speech tasks
    "WhisperModel": "automatic-speech-recognition",
    "Wav2Vec2Model": "automatic-speech-recognition",
    "HubertModel": "automatic-speech-recognition",
    "SEWModel": "automatic-speech-recognition",
    "EncodecModel": "audio-to-audio",
    "Data2VecAudioModel": "automatic-speech-recognition",
    "ClapModel": "audio-classification",
    "MusicgenModel": "text-to-audio",
    "UniSpeechModel": "automatic-speech-recognition",
    
    # Multimodal tasks
    "CLIPModel": "zero-shot-image-classification",
    "BlipModel": "image-to-text",
    "BlipForConditionalGeneration": "image-to-text",
    "BlipForQuestionAnswering": "visual-question-answering",
    "BlipForImageTextRetrieval": "image-to-text",
    "Blip2Model": "image-to-text",
    "LlavaModel": "image-to-text",
    "FlavaModel": "image-to-text",
    "GitModel": "image-to-text",
    "IdeficsModel": "image-to-text",
    "ImagebindModel": "image-classification",
    "ViltModel": "visual-question-answering",
    "XClipModel": "zero-shot-image-classification",
    "PaliGemmaModel": "image-to-text"
}

def discover_transformers_classes() -> Dict[str, Dict[str, Any]]:
    """
    Discover all HuggingFace Transformers model classes with from_pretrained support.
    
    Returns:
        Dictionary mapping class names to information about each class.
    """
    try:
        import transformers
    except ImportError:
        logger.error("Transformers library not installed. Cannot discover classes.")
        return {}
    
    transformers_classes = {}
    
    # Get all module attributes
    for attr_name in dir(transformers):
        # Check if it's a class that might be a model
        if attr_name.endswith("Model") or attr_name.endswith("ForSequenceClassification") or \
           attr_name.endswith("ForQuestionAnswering") or attr_name.endswith("ForMaskedLM") or \
           attr_name.endswith("ForCausalLM") or attr_name.endswith("ForTokenClassification") or \
           attr_name.endswith("ForImageClassification") or attr_name.endswith("ForConditionalGeneration"):
            
            try:
                # Get the class object
                cls = getattr(transformers, attr_name)
                
                # Verify it's a class
                if not inspect.isclass(cls):
                    continue
                
                # Check if it has from_pretrained method
                if hasattr(cls, "from_pretrained") and callable(getattr(cls, "from_pretrained")):
                    # Determine category
                    category = "other"
                    for cat, class_prefixes in ARCHITECTURE_CATEGORIES.items():
                        if any(attr_name.startswith(prefix) for prefix in class_prefixes):
                            category = cat
                            break
                    
                    # Store class info
                    transformers_classes[attr_name] = {
                        "category": category,
                        "has_from_pretrained": True,
                        "class_path": f"transformers.{attr_name}",
                        "recommended_model": MODEL_CLASS_MAPPING.get(attr_name, None),
                        "pipeline_task": PIPELINE_TASK_MAPPING.get(attr_name, None)
                    }
            
            except (AttributeError, ImportError):
                # Skip any classes that can't be loaded
                continue
    
    logger.info(f"Discovered {len(transformers_classes)} Transformers classes with from_pretrained support")
    return transformers_classes

def get_architecture_from_class(class_name: str) -> str:
    """Determine architecture type from class name."""
    for arch, class_prefixes in ARCHITECTURE_CATEGORIES.items():
        if any(class_name.startswith(prefix) for prefix in class_prefixes):
            return arch
    return "other"

def generate_test_for_class(
    class_name: str, 
    class_info: Dict[str, Any],
    output_dir: str,
    overwrite: bool = False
) -> bool:
    """
    Generate a test file for a specific Transformers class.
    
    Args:
        class_name: Name of the class
        class_info: Information about the class
        output_dir: Directory to save the test file
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if generation was successful, False otherwise
    """
    # Skip if no recommended model
    if not class_info.get("recommended_model"):
        logger.warning(f"No recommended model for {class_name}, skipping")
        return False
    
    # Determine model ID
    model_id = class_info["recommended_model"]
    
    # Determine architecture
    architecture = class_info["category"]
    
    # Determine output path
    # Convert class name to snake case for file name
    file_name = "test_" + "".join([
        "_" + c.lower() if c.isupper() else c.lower() 
        for c in class_name
    ]).lstrip("_").replace("_model", "") + ".py"
    
    subdir = f"models/{architecture}"
    output_path = os.path.join(output_dir, subdir, file_name)
    
    # Check if file already exists
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Test file for {class_name} already exists at {output_path}, skipping")
        return True
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate test file
    try:
        # Try using the template system first
        result = generate_test_file(model_id, architecture, debug=False)
        
        if result:
            logger.info(f"Successfully generated test file for {class_name} at {output_path}")
            return True
        else:
            # If template system fails, try direct generation with existing template
            # Find appropriate reference file for the architecture
            architecture_ref_files = {
                "vision": "test_vit_base_patch16_224.py",
                "encoder_only": "test_bert_base_uncased.py",
                "decoder_only": "test_gpt2.py",
                "encoder_decoder": "test_t5_base.py",
                "speech": "test_whisper_tiny.py",
                "multimodal": "test_clip_vit_base_patch32.py",
                "other": "test_bert_base_uncased.py"
            }
            
            ref_file = architecture_ref_files.get(architecture)
            if not ref_file:
                logger.error(f"No reference file for architecture {architecture}")
                return False
            
            # Try to locate reference file
            ref_path = None
            for root, _, files in os.walk(output_dir):
                if ref_file in files:
                    ref_path = os.path.join(root, ref_file)
                    break
            
            if not ref_path:
                logger.error(f"Reference file {ref_file} not found")
                return False
            
            # Use reference file as template
            with open(ref_path, 'r') as f:
                content = f.read()
            
            # Extract model name from class name
            model_short_name = "".join(
                [c for c in class_name if c.isupper()]
            ).lower()
            
            # Replace model specifics
            content = content.replace(os.path.basename(ref_path).replace('.py', ''), 
                                     file_name.replace('.py', ''))
            
            # Replace class name - convert to CamelCase
            class_name_parts = file_name.replace('test_', '').replace('.py', '').split('_')
            test_class_name = ''.join(part.capitalize() for part in class_name_parts)
            content = content.replace(f"Test{ref_file.replace('test_', '').replace('.py', '').capitalize()}", 
                                     f"Test{test_class_name}")
            
            # Replace model ID
            content = content.replace(f'self.model_id = "{model_id}"', 
                                     f'self.model_id = "{class_info["recommended_model"]}"')
            
            # Update pipeline task if available
            if class_info.get("pipeline_task"):
                content = content.replace(f'self.task = "image-classification"', 
                                        f'self.task = "{class_info["pipeline_task"]}"')
                content = content.replace(f'self.task = "text-generation"', 
                                        f'self.task = "{class_info["pipeline_task"]}"')
                content = content.replace(f'self.task = "fill-mask"', 
                                        f'self.task = "{class_info["pipeline_task"]}"')
            
            # Write to output file
            with open(output_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Generated test file for {class_name} at {output_path} using reference file")
            return True
            
    except Exception as e:
        logger.error(f"Error generating test file for {class_name}: {e}")
        return False

def batch_generate_tests(
    class_info_dict: Dict[str, Dict[str, Any]],
    output_dir: str,
    categories: Optional[List[str]] = None,
    class_filter: Optional[List[str]] = None,
    max_workers: int = 4,
    overwrite: bool = False,
    dry_run: bool = False
) -> Dict[str, bool]:
    """
    Generate test files for multiple Transformers classes.
    
    Args:
        class_info_dict: Dictionary of class information
        output_dir: Base output directory for test files
        categories: Optional list of categories to include
        class_filter: Optional list of class names to filter by
        max_workers: Maximum number of parallel workers
        overwrite: Whether to overwrite existing files
        dry_run: Just print what would be done, don't actually generate files
        
    Returns:
        Dictionary mapping class names to generation success status
    """
    results = {}
    
    # Filter by category if specified
    if categories:
        filtered_classes = {
            class_name: info for class_name, info in class_info_dict.items()
            if info["category"] in categories
        }
    else:
        filtered_classes = class_info_dict
    
    # Further filter by class name if specified
    if class_filter:
        filtered_classes = {
            class_name: info for class_name, info in filtered_classes.items()
            if any(class_name.startswith(prefix) for prefix in class_filter)
        }
    
    # Filter by classes that have recommended models
    classes_with_models = {
        class_name: info for class_name, info in filtered_classes.items()
        if info.get("recommended_model")
    }
    
    logger.info(f"Preparing to generate tests for {len(classes_with_models)} classes")
    
    if dry_run:
        logger.info("Dry run mode - showing what would be generated:")
        for class_name, info in classes_with_models.items():
            output_path = os.path.join(
                output_dir, 
                f"models/{info['category']}", 
                "test_" + "".join(["_" + c.lower() if c.isupper() else c.lower() for c in class_name]).lstrip("_").replace("_model", "") + ".py"
            )
            logger.info(f"Would generate test for {class_name} using model {info['recommended_model']} at {output_path}")
        return {class_name: True for class_name in classes_with_models}
    
    # Generate tests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_class = {
            executor.submit(
                generate_test_for_class, 
                class_name, 
                info, 
                output_dir,
                overwrite
            ): class_name 
            for class_name, info in classes_with_models.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_class):
            class_name = future_to_class[future]
            try:
                success = future.result()
                results[class_name] = success
                if success:
                    logger.info(f"Successfully generated test for {class_name}")
                else:
                    logger.error(f"Failed to generate test for {class_name}")
            except Exception as e:
                logger.error(f"Error generating test for {class_name}: {e}")
                results[class_name] = False
    
    # Summarize results
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    success_rate = (successful / total) * 100 if total > 0 else 0
    
    logger.info(f"Test generation complete: {successful}/{total} successful ({success_rate:.1f}%)")
    
    return results

def save_results(results: Dict[str, bool], output_file: str = None):
    """Save batch generation results to a file."""
    if output_file is None:
        output_file = f"hf_test_gen_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Add timestamp
    results_with_meta = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "successful": sum(1 for success in results.values() if success),
            "failed": sum(1 for success in results.values() if not success),
        },
        "results": results
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(results_with_meta, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return output_file

def generate_report(
    class_info_dict: Dict[str, Dict[str, Any]],
    results: Dict[str, bool],
    output_file: str = "hf_test_coverage_report.md"
):
    """Generate a test coverage report."""
    with open(output_file, 'w') as f:
        f.write("# HuggingFace Transformers Test Coverage Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        total_classes = len(class_info_dict)
        testable_classes = sum(1 for info in class_info_dict.values() if info.get("recommended_model"))
        successful_tests = sum(1 for success in results.values() if success)
        
        f.write("## Summary\n\n")
        f.write(f"- Total Transformer Classes: {total_classes}\n")
        f.write(f"- Classes with Recommended Models: {testable_classes}\n")
        f.write(f"- Successfully Generated Tests: {successful_tests}\n")
        f.write(f"- Coverage Rate: {(successful_tests / testable_classes) * 100:.1f}%\n\n")
        
        # Coverage by category
        f.write("## Coverage by Category\n\n")
        f.write("| Category | Total Classes | Testable Classes | Tests Generated | Coverage |\n")
        f.write("|----------|--------------|------------------|-----------------|----------|\n")
        
        for category in sorted(ARCHITECTURE_CATEGORIES.keys()):
            cat_classes = {name: info for name, info in class_info_dict.items() 
                          if info["category"] == category}
            cat_total = len(cat_classes)
            cat_testable = sum(1 for info in cat_classes.values() if info.get("recommended_model"))
            cat_success = sum(1 for name, success in results.items() 
                             if name in cat_classes and success)
            cat_coverage = (cat_success / cat_testable) * 100 if cat_testable > 0 else 0
            
            f.write(f"| {category} | {cat_total} | {cat_testable} | {cat_success} | {cat_coverage:.1f}% |\n")
        
        # List successful tests
        f.write("\n## Generated Test Files\n\n")
        
        # Group by category
        for category in sorted(ARCHITECTURE_CATEGORIES.keys()):
            f.write(f"### {category.capitalize()} Models\n\n")
            
            # Get successful tests for this category
            successful_tests = [
                name for name, success in results.items() 
                if success and class_info_dict[name]["category"] == category
            ]
            
            if successful_tests:
                for class_name in sorted(successful_tests):
                    model_id = class_info_dict[class_name].get("recommended_model", "N/A")
                    task = class_info_dict[class_name].get("pipeline_task", "N/A")
                    f.write(f"- **{class_name}**: {model_id} (Task: {task})\n")
            else:
                f.write("No tests generated for this category.\n")
            
            f.write("\n")
        
        # List failed tests
        f.write("## Failed Tests\n\n")
        failed_tests = [name for name, success in results.items() if not success]
        
        if failed_tests:
            for class_name in sorted(failed_tests):
                model_id = class_info_dict[class_name].get("recommended_model", "N/A")
                f.write(f"- **{class_name}**: {model_id}\n")
        else:
            f.write("No test generation failures.\n")
    
    logger.info(f"Coverage report generated at {output_file}")
    return output_file

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive test generator for HuggingFace models")
    
    # Discovery options
    discovery_group = parser.add_argument_group("Discovery Options")
    discovery_group.add_argument("--discover-only", action="store_true", 
                               help="Only discover classes, don't generate tests")
    discovery_group.add_argument("--discovery-output", type=str,
                               help="Save discovered classes to JSON file")
    
    # Generation options
    generation_group = parser.add_argument_group("Generation Options")
    generation_group.add_argument("--categories", type=str, nargs="+", 
                                choices=list(ARCHITECTURE_CATEGORIES.keys()),
                                help="Categories to generate tests for")
    generation_group.add_argument("--classes", type=str, nargs="+",
                                help="Specific class prefixes to generate tests for")
    generation_group.add_argument("--output-dir", type=str,
                                help="Output directory for test files")
    generation_group.add_argument("--max-workers", type=int, default=4,
                                help="Maximum number of parallel workers")
    generation_group.add_argument("--overwrite", action="store_true",
                                help="Overwrite existing test files")
    generation_group.add_argument("--results-file", type=str,
                                help="Save results to specified JSON file")
    generation_group.add_argument("--report-file", type=str, default="hf_test_coverage_report.md",
                                help="Generate coverage report to specified file")
    
    # Other options
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't generate files, just list what would be done")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Discover transformers classes
    logger.info("Discovering HuggingFace Transformers classes...")
    class_info_dict = discover_transformers_classes()
    
    # Save discovered classes if requested
    if args.discovery_output:
        with open(args.discovery_output, 'w') as f:
            json.dump(class_info_dict, f, indent=2)
        logger.info(f"Saved discovered classes to {args.discovery_output}")
    
    # Exit if discover-only
    if args.discover_only:
        logger.info(f"Discovered {len(class_info_dict)} classes. Exiting without generating tests.")
        return 0
    
    # Set output directory
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(script_dir), "refactored_test_suite")
    
    # Generate tests
    results = batch_generate_tests(
        class_info_dict,
        args.output_dir,
        categories=args.categories,
        class_filter=args.classes,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run
    )
    
    # Save results if not in dry run mode
    if not args.dry_run:
        save_results(results, args.results_file)
        
        # Generate report
        generate_report(class_info_dict, results, args.report_file)
    
    # Calculate success rate
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    if total > 0:
        success_rate = (successful / total) * 100
        logger.info(f"Test generation complete: {successful}/{total} successful ({success_rate:.1f}%)")
    
    # Exit with success if all generations succeeded or no tests were attempted
    return 0 if successful == total or total == 0 else 1

if __name__ == "__main__":
    sys.exit(main())