#!/usr/bin/env python3
"""
Simple Generator for HuggingFace Test Files

This script implements a simple generator for creating test files for HuggingFace models
based on existing templates. It doesn't attempt to import template files with Jinja syntax,
which can cause syntax errors.
"""

import os
import sys
import argparse
import logging
import json
import time
import datetime
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Architecture to model mapping
ARCHITECTURE_MAPPING = {
    "encoder-only": ["bert", "roberta", "distilbert", "albert", "electra", "camembert", 
                    "xlm-roberta", "deberta", "ernie", "rembert"],
    "decoder-only": ["gpt2", "gpt-2", "gptj", "gpt-j", "gpt-neo", "gpt-neox", "llama", 
                    "llama2", "mistral", "falcon", "phi", "gemma", "opt", "mpt"],
    "encoder-decoder": ["t5", "bart", "mbart", "pegasus", "mt5", "led", "prophetnet"],
    "vision": ["vit", "swin", "resnet", "deit", "beit", "segformer", "detr", "mask2former", 
                "yolos", "sam", "dinov2", "convnext"],
    "vision-text": ["clip", "blip", "flava", "git", "idefics", "paligemma", "imagebind", 
                    "llava", "fuyu"],
    "speech": ["whisper", "wav2vec2", "hubert", "sew", "unispeech", "clap", "musicgen", 
                "encodec"]
}

# Default model mapping (model type to default model ID)
DEFAULT_MODELS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "gpt2": "gpt2",
    "llama": "meta-llama/Llama-2-7b-hf",
    "t5": "t5-small",
    "bart": "facebook/bart-base",
    "vit": "google/vit-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
    "whisper": "openai/whisper-tiny"
}

def map_model_to_architecture(model_type: str) -> str:
    """Map a model type to its architecture.
    
    Args:
        model_type: The model type (bert, gpt2, t5, etc.)
        
    Returns:
        The architecture type (encoder-only, decoder-only, etc.)
    """
    # First, check for exact match in any architecture
    for architecture, models in ARCHITECTURE_MAPPING.items():
        if model_type in models:
            return architecture
    
    # If no exact match, check for partial matches
    model_type_lower = model_type.lower()
    for architecture, models in ARCHITECTURE_MAPPING.items():
        for model in models:
            if model_type_lower.startswith(model.lower()) or model.lower().startswith(model_type_lower):
                return architecture
    
    # If all else fails, return unknown
    logger.warning(f"Unknown architecture for model type: {model_type}")
    return "unknown"

def get_default_model(model_type: str) -> str:
    """Get the default model ID for a model type.
    
    Args:
        model_type: The model type (bert, gpt2, t5, etc.)
        
    Returns:
        The default model ID for the model type.
    """
    if model_type in DEFAULT_MODELS:
        return DEFAULT_MODELS[model_type]
    
    # Try to construct a default model ID
    return f"{model_type}-base"

def get_default_task(model_type: str, architecture: str) -> str:
    """Get the default task for a model type and architecture.
    
    Args:
        model_type: The model type (bert, gpt2, t5, etc.)
        architecture: The architecture type (encoder-only, decoder-only, etc.)
        
    Returns:
        The default task for the model type and architecture.
    """
    # Default tasks by architecture
    architecture_tasks = {
        "encoder-only": "fill-mask",
        "decoder-only": "text-generation",
        "encoder-decoder": "text2text-generation",
        "vision": "image-classification",
        "vision-text": "image-to-text",
        "speech": "automatic-speech-recognition"
    }
    
    # Specific model tasks that override the defaults
    model_tasks = {
        "bert": "fill-mask",
        "roberta": "fill-mask",
        "gpt2": "text-generation",
        "llama": "text-generation",
        "t5": "text2text-generation",
        "bart": "text2text-generation",
        "vit": "image-classification",
        "clip": "image-to-text",
        "whisper": "automatic-speech-recognition"
    }
    
    # First, check for model-specific task
    if model_type in model_tasks:
        return model_tasks[model_type]
    
    # Otherwise, use architecture-based task
    if architecture in architecture_tasks:
        return architecture_tasks[architecture]
    
    # Fallback to a generic task
    return "unknown"

def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get model information for a model type.
    
    Args:
        model_type: The model type (bert, gpt2, t5, etc.)
        
    Returns:
        Dict containing model information.
    """
    architecture = map_model_to_architecture(model_type)
    default_model = get_default_model(model_type)
    
    # Build class name from model type
    class_name = "".join(part.capitalize() for part in model_type.split("-"))
    
    return {
        "name": model_type,
        "id": default_model,
        "architecture": architecture,
        "class_name": class_name,
        "task": get_default_task(model_type, architecture),
        "type": model_type,  # For template compatibility
        "default": True
    }

def get_template_path(architecture: str) -> Optional[str]:
    """Get the path to the template file for an architecture.
    
    Args:
        architecture: The architecture type (encoder-only, decoder-only, etc.)
        
    Returns:
        Path to the template file, or None if not found.
    """
    # Map architecture to template file
    template_files = {
        "encoder-only": "encoder_only_template.py",
        "decoder-only": "decoder_only_template.py",
        "encoder-decoder": "encoder_decoder_template.py",
        "vision": "vision_template.py",
        "vision-text": "vision_text_template.py",
        "speech": "speech_template.py"
    }
    
    if architecture not in template_files:
        return None
    
    # Check template directory
    template_file = template_files[architecture]
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills", "templates")
    template_path = os.path.join(template_dir, template_file)
    
    if not os.path.exists(template_path):
        return None
    
    return template_path

def simple_strip_indentation(content: str) -> str:
    """Strip all extra indentation to fix Jinja template issues.
    
    Args:
        content: The generated code content
        
    Returns:
        Content with fixed indentation
    """
    # Split the content into lines
    lines = content.split('\n')
    
    # Process each line to remove extra indentation
    processed_lines = []
    in_multiline_string = False
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            processed_lines.append('')
            continue
            
        # Check for multiline string delimiters
        if '"""' in line or "'''" in line:
            # Toggle multiline string flag if odd number of delimiters
            if line.count('"""') % 2 == 1 or line.count("'''") % 2 == 1:
                in_multiline_string = not in_multiline_string
                
        # If we're in a multiline string, don't alter indentation
        if in_multiline_string:
            processed_lines.append(line)
            continue
            
        # Remove leading whitespace but preserve indentation for code blocks
        stripped = line.strip()
        
        # Preserve imports, class declarations, and function declarations
        if (stripped.startswith(('import ', 'from ', 'class ', 'def ', '@', '#', 
                                'if ', 'else:', 'elif ', 'try:', 'except:', 'finally:', 
                                'for ', 'while ', 'with '))):
            processed_lines.append(stripped)
        # Indent lines inside blocks
        elif stripped and any(lines[i].strip().endswith((':', '{', '[', '(')) for i in range(len(processed_lines)) if i > 0 and i == len(processed_lines) - 1):
            processed_lines.append('    ' + stripped)
        else:
            processed_lines.append(stripped)
    
    # Join the processed lines back into a single string
    return '\n'.join(processed_lines)

def normalize_template(content: str) -> str:
    """Normalize template content to prepare for Python syntax.
    
    Args:
        content: Template content
        
    Returns:
        Normalized content
    """
    # Replace template tags with empty strings
    import re
    # Remove {% ... %} blocks
    content = re.sub(r'{%.*?%}', '', content)
    # Remove {{ ... }} expressions
    content = re.sub(r'{{.*?}}', '', content)
    
    # Fix docstrings
    content = content.replace('""""', '"""')
    content = content.replace("''''", "'''")
    
    # Replace triple quotes with proper escaping
    content = content.replace('\\"""', '"""')
    content = content.replace("\\'\\'\\'", "'''")
    
    return content

def fix_indentation(content: str) -> str:
    """Fix indentation issues in the generated code.
    
    Args:
        content: The generated code content
        
    Returns:
        Content with fixed indentation
    """
    # First normalize the template
    content = normalize_template(content)
    
    # Then strip excess indentation
    content = simple_strip_indentation(content)
    
    # If we still have syntax errors, try a more aggressive approach
    is_valid, error = validate_python_syntax(content)
    if not is_valid:
        logger.warning(f"First indentation pass failed: {error}")
        
        # Split into lines and rebuild from scratch
        lines = content.split('\n')
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue
                
            # Decrease indent for closing statements
            if stripped.startswith(('}', ')', ']', 'else:', 'elif ', 'except:', 'finally:')):
                indent_level = max(0, indent_level - 1)
                
            # Add the line with current indentation
            fixed_lines.append('    ' * indent_level + stripped)
            
            # Increase indent after opening control statements
            if stripped.endswith((':', '{', '[', '(')):
                indent_level += 1
        
        content = '\n'.join(fixed_lines)
    
    return content

def validate_python_syntax(content: str) -> Tuple[bool, str]:
    """Validate Python syntax in the generated code.
    
    Args:
        content: The generated code content
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        compile(content, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {str(e)}"
    except IndentationError as e:
        return False, f"IndentationError: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def simple_render(template_str: str, context: Dict[str, Any]) -> str:
    """Simple string replacement for templates.
    
    Args:
        template_str: The template string
        context: The context for rendering
        
    Returns:
        The rendered string
    """
    rendered = template_str
    
    # First, collect all the Jinja blocks that might have indentation issues
    jinja_blocks = []
    
    # Use a simple approach to replace variables to avoid regex complexity
    
    # 1. Replace simple variables
    for key, value in context.items():
        if isinstance(value, (str, int, float, bool)):
            # Simple replacement
            placeholder = f"{{{{ {key} }}}}"
            rendered = rendered.replace(placeholder, str(value))
            
            # With filters
            rendered = rendered.replace(f"{{{{ {key}|capitalize }}}}", str(value).capitalize())
            rendered = rendered.replace(f"{{{{ {key}|upper }}}}", str(value).upper())
            rendered = rendered.replace(f"{{{{ {key}|lower }}}}", str(value).lower())
    
    # 2. Replace model_info dictionary values
    if "model_info" in context and isinstance(context["model_info"], dict):
        for key, value in context["model_info"].items():
            if isinstance(value, (str, int, float, bool)):
                # Simple replacement
                placeholder = f"{{{{ model_info.{key} }}}}"
                rendered = rendered.replace(placeholder, str(value))
                
                # With filters
                rendered = rendered.replace(f"{{{{ model_info.{key}|capitalize }}}}", str(value).capitalize())
                rendered = rendered.replace(f"{{{{ model_info.{key}|upper }}}}", str(value).upper())
                rendered = rendered.replace(f"{{{{ model_info.{key}|lower }}}}", str(value).lower())
    
    # 3. Handle conditional blocks - more comprehensive approach
    # First identify all conditional blocks
    import re
    
    # Create a regex for finding all conditional blocks
    block_pattern = r'{%\s*if\s+([a-zA-Z0-9_]+)\s*%}(.*?){%\s*endif\s*%}'
    
    # Find all blocks using regex with re.DOTALL to match across newlines
    for match in re.finditer(block_pattern, rendered, re.DOTALL):
        condition_var = match.group(1)
        block_content = match.group(2)
        full_block = match.group(0)
        
        # Check if the condition is in our context
        if condition_var in context:
            condition_value = context[condition_var]
            
            if condition_value:
                # Condition is true, keep the content but remove the markers
                rendered = rendered.replace(full_block, block_content)
            else:
                # Condition is false, remove the entire block
                rendered = rendered.replace(full_block, '')
        else:
            # Unknown condition, just remove the block markers
            rendered = rendered.replace(full_block, block_content)
    
    # 4. Clean up any remaining Jinja syntax
    # Use a safe approach that preserves content but removes markers
    rendered = re.sub(r'{%.*?%}', '', rendered)  # Remove {% ... %} blocks
    rendered = re.sub(r'{{.*?}}', '', rendered)  # Remove {{ ... }} expressions
    
    # 5. Fix indentation now that all template markers are gone
    # First normalize template content and remove problematic sequences
    rendered = rendered.replace('""""', '"""')
    rendered = rendered.replace("''''", "'''")
    rendered = rendered.replace('\\"""', '"""')
    rendered = rendered.replace("\\'\\'\\'", "'''")
    
    # 6. Properly fix indentation
    fixed_content = fix_indentation(rendered)
    
    # 7. Final validation
    is_valid, error = validate_python_syntax(fixed_content)
    if not is_valid:
        logger.warning(f"Final syntax validation failed: {error}")
        logger.warning("Using best-effort version of template")
    else:
        logger.info("Template validated successfully")
    
    return fixed_content

def generate_test(model_type: str, output_dir: str = "./generated_tests") -> Dict[str, Any]:
    """Generate a test file for a model type.
    
    Args:
        model_type: The model type (bert, gpt2, t5, etc.)
        output_dir: Directory to save the generated file
        
    Returns:
        Dict with generation results
    """
    start_time = time.time()
    logger.info(f"Generating test for model type: {model_type}")
    
    # Get model information
    model_info = get_model_info(model_type)
    architecture = model_info["architecture"]
    
    # Get template path
    template_path = get_template_path(architecture)
    if not template_path:
        error_msg = f"No template found for architecture: {architecture}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "model_type": model_type,
            "duration": time.time() - start_time
        }
    
    # Read template
    try:
        with open(template_path, 'r') as f:
            template_str = f.read()
    except Exception as e:
        error_msg = f"Error reading template file: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "model_type": model_type,
            "duration": time.time() - start_time
        }
    
    # Build context
    context = {
        "model_type": model_type,
        "model_info": model_info,
        "timestamp": datetime.datetime.now().isoformat(),
        "has_cuda": True,  # Simplified for demo
        "has_rocm": False,
        "has_mps": False,
        "has_openvino": False,
        "has_webnn": False,
        "has_webgpu": False
    }
    
    # Render template
    try:
        rendered = simple_render(template_str, context)
    except Exception as e:
        error_msg = f"Error rendering template: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "model_type": model_type,
            "duration": time.time() - start_time
        }
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Write output file
    output_file = os.path.join(output_dir, f"test_{model_type}.py")
    try:
        with open(output_file, 'w') as f:
            f.write(rendered)
    except Exception as e:
        error_msg = f"Error writing output file: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "model_type": model_type,
            "duration": time.time() - start_time
        }
    
    logger.info(f"Generated test file: {output_file}")
    
    return {
        "success": True,
        "output_file": output_file,
        "model_type": model_type,
        "architecture": architecture,
        "duration": time.time() - start_time
    }

def get_models_by_architecture(architecture: str) -> List[str]:
    """Get a list of model types for a specific architecture.
    
    Args:
        architecture: The architecture type (encoder-only, decoder-only, etc.)
        
    Returns:
        List of model types.
    """
    if architecture in ARCHITECTURE_MAPPING:
        return ARCHITECTURE_MAPPING[architecture]
    return []

def generate_for_architecture(architecture: str, output_dir: str) -> Dict[str, Any]:
    """Generate tests for all models of a specific architecture.
    
    Args:
        architecture: The architecture type (encoder-only, decoder-only, etc.)
        output_dir: Directory to save the generated files
        
    Returns:
        Dict with generation results
    """
    start_time = time.time()
    
    # Get models for architecture
    models = get_models_by_architecture(architecture)
    if not models:
        return {
            "success": False,
            "error": f"No models found for architecture: {architecture}",
            "duration": time.time() - start_time
        }
    
    logger.info(f"Found {len(models)} models for architecture: {architecture}")
    
    # Generate tests for each model
    results = {}
    for model_type in models:
        logger.info(f"Generating test for model type: {model_type}")
        result = generate_test(model_type, output_dir)
        results[model_type] = result
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for result in results.values() if result["success"])
    
    return {
        "success": successful > 0,
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "architecture": architecture,
        "duration": time.time() - start_time,
        "results": results
    }

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simple Generator for HuggingFace Test Files")
    parser.add_argument("--model", help="Model type to generate a test for (bert, gpt2, t5, etc.)")
    parser.add_argument("--architecture", choices=list(ARCHITECTURE_MAPPING.keys()),
                       help="Generate tests for all models of a specific architecture")
    parser.add_argument("--output-dir", default="./generated_tests", help="Output directory for generated files")
    parser.add_argument("--limit", type=int, default=0, 
                       help="Limit the number of models to generate (0 = no limit)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure either model or architecture is specified
    if not args.model and not args.architecture:
        print("Error: Either --model or --architecture must be specified")
        return 1
    
    # Generate test(s)
    if args.model:
        # Generate test for a single model
        result = generate_test(args.model, args.output_dir)
        
        # Print result
        if result["success"]:
            print(f"Generation successful!")
            print(f"Output written to: {result['output_file']}")
            print(f"Model type: {result['model_type']}")
            print(f"Architecture: {result['architecture']}")
            print(f"Duration: {result['duration']:.2f} seconds")
            return 0
        else:
            print(f"Generation failed: {result['error']}")
            return 1
    else:
        # Generate tests for all models of a specific architecture
        result = generate_for_architecture(args.architecture, args.output_dir)
        
        # Apply limit if specified
        if args.limit > 0 and "results" in result:
            # Limit the number of models
            model_types = list(result["results"].keys())
            if len(model_types) > args.limit:
                for model_type in model_types[args.limit:]:
                    del result["results"][model_type]
                
                # Update statistics
                result["total"] = len(result["results"])
                result["successful"] = sum(1 for r in result["results"].values() if r["success"])
                result["failed"] = result["total"] - result["successful"]
        
        # Print result
        if result["success"]:
            print(f"Generation for architecture {args.architecture} successful!")
            print(f"Total models: {result['total']}")
            print(f"Successful: {result['successful']}")
            print(f"Failed: {result['failed']}")
            print(f"Duration: {result['duration']:.2f} seconds")
            
            # Save summary
            summary_file = os.path.join(args.output_dir, f"{args.architecture}_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Summary saved to: {summary_file}")
            
            return 0
        else:
            print(f"Generation failed: {result['error']}")
            return 1

if __name__ == "__main__":
    sys.exit(main())