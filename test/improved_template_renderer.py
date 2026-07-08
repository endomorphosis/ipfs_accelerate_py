#!/usr/bin/env python3
"""
Improved Template Renderer for HuggingFace Test Files

This module implements a more robust template rendering system that handles complex
Jinja-like templates with better indentation management and syntax validation.
"""

import os
import sys
import re
import logging
import time
import datetime
import ast
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntaxValidator:
    """Validates and fixes Python syntax"""
    
    @staticmethod
    def validate(content: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate Python syntax and attempt to fix common issues.
        
        Args:
            content: String containing Python code
            
        Returns:
            Tuple of:
                - is_valid: Boolean indicating if syntax is valid
                - error_message: Error message if not valid, empty string otherwise
                - fixed_content: Fixed content if auto-fix was successful, None otherwise
        """
        try:
            # First try to compile the code directly
            compile(content, "<string>", "exec")
            return True, "", None
        except (SyntaxError, IndentationError) as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            
            # Try to fix the syntax
            fixed_content = SyntaxValidator._attempt_fix(content, error_message)
            if fixed_content:
                # Verify the fix worked
                try:
                    compile(fixed_content, "<string>", "exec")
                    return True, "", fixed_content
                except (SyntaxError, IndentationError):
                    # Fix didn't work
                    pass
            
            return False, error_message, None
    
    @staticmethod
    def _attempt_fix(content: str, error_message: str) -> Optional[str]:
        """
        Attempt to fix common syntax errors.
        
        Args:
            content: String containing Python code with syntax errors
            error_message: The error message from the syntax error
            
        Returns:
            Fixed content if successful, None otherwise
        """
        fixed_content = content
        
        # Fix unbalanced parentheses
        if "unexpected EOF" in error_message or "unexpected end of file" in error_message:
            # Count opening and closing of each bracket type
            parens_count = fixed_content.count('(') - fixed_content.count(')')
            braces_count = fixed_content.count('{') - fixed_content.count('}')
            brackets_count = fixed_content.count('[') - fixed_content.count(']')
            
            # Add missing closing brackets
            if parens_count > 0:
                fixed_content += ')' * parens_count
            if braces_count > 0:
                fixed_content += '}' * braces_count
            if brackets_count > 0:
                fixed_content += ']' * brackets_count
                
        # Fix indentation errors
        if "expected an indented block" in error_message:
            fixed_content = SyntaxValidator._fix_indentation(fixed_content)
        
        # Fix bad escape sequences
        if "bad escape" in error_message:
            # Replace common problematic escape sequences
            fixed_content = fixed_content.replace('\\u', '\\\\u')
            fixed_content = fixed_content.replace('\\N', '\\\\N')
        
        # Fix unbalanced triple quotes
        if "EOF while scanning triple-quoted string" in error_message:
            # Find unbalanced triple quotes
            triple_quotes = ['"""', "'''"]
            for quote in triple_quotes:
                if fixed_content.count(quote) % 2 == 1:
                    fixed_content += quote
        
        # If we've made any changes, return the fixed content
        if fixed_content != content:
            return fixed_content
            
        return None
    
    @staticmethod
    def _fix_indentation(content: str) -> str:
        """
        Fix common indentation issues.
        
        Args:
            content: String containing Python code with indentation errors
            
        Returns:
            Fixed content
        """
        lines = content.split('\n')
        fixed_lines = []
        indent_level = 0
        indent_stack = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Check for dedent
            if (stripped.startswith(('return', 'pass', 'break', 'continue', 'raise', 'else:', 'elif ', 'except:', 'finally:')) or
                stripped.startswith((')', '}', ']'))):
                if indent_level > 0:
                    indent_level -= 1
            
            # Add the line with current indentation
            fixed_lines.append('    ' * indent_level + stripped)
            
            # Check for indent
            if stripped.endswith((':', '{', '[', '(')):
                indent_level += 1
                indent_stack.append(stripped)
            
        return '\n'.join(fixed_lines)


class TemplateRenderer:
    """Renders Jinja-like templates with improved indentation handling"""
    
    @staticmethod
    def render(template_str: str, context: Dict[str, Any]) -> str:
        """
        Render a template string with the given context.
        
        Args:
            template_str: The template string
            context: The context for rendering
            
        Returns:
            The rendered string
        """
        # First, handle conditional blocks
        rendered = TemplateRenderer._render_conditionals(template_str, context)
        
        # Then handle variable substitutions
        rendered = TemplateRenderer._render_variables(rendered, context)
        
        # Clean up any remaining template syntax
        rendered = TemplateRenderer._cleanup_template_syntax(rendered)
        
        # Fix indentation
        rendered = TemplateRenderer._fix_indentation(rendered)
        
        # Final validation
        is_valid, error, fixed_content = SyntaxValidator.validate(rendered)
        if not is_valid:
            logger.warning(f"Syntax validation failed: {error}")
            if fixed_content:
                logger.info("Using auto-fixed version")
                rendered = fixed_content
        
        return rendered
    
    @staticmethod
    def _render_conditionals(template_str: str, context: Dict[str, Any]) -> str:
        """
        Render conditional blocks in a template.
        
        Args:
            template_str: The template string
            context: The context for rendering
            
        Returns:
            The rendered string with conditionals processed
        """
        result = template_str
        
        # Handle if-else blocks
        if_else_pattern = r'{%\s*if\s+([a-zA-Z0-9_\.]+)\s*%}(.*?){%\s*else\s*%}(.*?){%\s*endif\s*%}'
        for match in re.finditer(if_else_pattern, result, re.DOTALL):
            condition_var = match.group(1)
            if_block = match.group(2)
            else_block = match.group(3)
            full_block = match.group(0)
            
            # Handle nested properties (e.g., model_info.type)
            condition_value = TemplateRenderer._get_nested_value(context, condition_var)
            
            # Replace block based on condition
            if condition_value:
                result = result.replace(full_block, if_block)
            else:
                result = result.replace(full_block, else_block)
        
        # Handle simple if blocks (without else)
        if_pattern = r'{%\s*if\s+([a-zA-Z0-9_\.]+)\s*%}(.*?){%\s*endif\s*%}'
        for match in re.finditer(if_pattern, result, re.DOTALL):
            condition_var = match.group(1)
            block_content = match.group(2)
            full_block = match.group(0)
            
            # Handle nested properties
            condition_value = TemplateRenderer._get_nested_value(context, condition_var)
            
            # Replace block based on condition
            if condition_value:
                result = result.replace(full_block, block_content)
            else:
                result = result.replace(full_block, '')
        
        # Handle for loops
        for_pattern = r'{%\s*for\s+([a-zA-Z0-9_]+)\s+in\s+([a-zA-Z0-9_\.]+)\s*%}(.*?){%\s*endfor\s*%}'
        for match in re.finditer(for_pattern, result, re.DOTALL):
            loop_var = match.group(1)
            collection_var = match.group(2)
            loop_content = match.group(3)
            full_block = match.group(0)
            
            # Get the collection to iterate over
            collection = TemplateRenderer._get_nested_value(context, collection_var)
            
            if collection and isinstance(collection, (list, tuple, dict)):
                # Process the loop
                loop_result = []
                for item in collection:
                    # Create a new context with the loop variable
                    loop_context = context.copy()
                    loop_context[loop_var] = item
                    
                    # Render the loop content with the new context
                    rendered_content = TemplateRenderer._render_variables(loop_content, loop_context)
                    loop_result.append(rendered_content)
                
                # Replace the for block with the rendered content
                result = result.replace(full_block, ''.join(loop_result))
            else:
                # Empty or non-existent collection, remove the for block
                result = result.replace(full_block, '')
        
        return result
    
    @staticmethod
    def _render_variables(template_str: str, context: Dict[str, Any]) -> str:
        """
        Render variables in a template.
        
        Args:
            template_str: The template string
            context: The context for rendering
            
        Returns:
            The rendered string with variables replaced
        """
        result = template_str
        
        # Handle filters
        var_filter_pattern = r'{{(.+?)\|(.+?)}}' 
        for match in re.finditer(var_filter_pattern, result):
            expr = match.group(1).strip()
            filter_name = match.group(2).strip()
            full_match = match.group(0)
            
            # Get the variable value
            var_value = TemplateRenderer._get_nested_value(context, expr)
            
            # Apply the filter
            if var_value is not None:
                if filter_name == 'capitalize':
                    filter_result = str(var_value).capitalize()
                elif filter_name == 'upper':
                    filter_result = str(var_value).upper()
                elif filter_name == 'lower':
                    filter_result = str(var_value).lower()
                elif filter_name == 'title':
                    filter_result = str(var_value).title()
                elif filter_name == 'length' or filter_name == 'len':
                    filter_result = str(len(var_value) if hasattr(var_value, '__len__') else 0)
                elif filter_name == 'join':
                    filter_result = ','.join(var_value) if isinstance(var_value, (list, tuple)) else str(var_value)
                else:
                    # Unknown filter, just use the value
                    filter_result = str(var_value)
                
                # Replace the expression
                result = result.replace(full_match, filter_result)
            else:
                # Variable not found, remove the expression
                result = result.replace(full_match, '')
        
        # Handle regular variables
        var_pattern = r'{{(.+?)}}'
        for match in re.finditer(var_pattern, result):
            expr = match.group(1).strip()
            full_match = match.group(0)
            
            # Get the variable value
            var_value = TemplateRenderer._get_nested_value(context, expr)
            
            # Replace the expression
            if var_value is not None:
                result = result.replace(full_match, str(var_value))
            else:
                # Variable not found, remove the expression
                result = result.replace(full_match, '')
        
        return result
    
    @staticmethod
    def _get_nested_value(context: Dict[str, Any], expr: str) -> Any:
        """
        Get a value from a nested context based on a dotted expression.
        
        Args:
            context: The context
            expr: The expression (e.g., 'model_info.type')
            
        Returns:
            The value, or None if not found
        """
        parts = expr.split('.')
        current = context
        
        for part in parts:
            # Try to access the part from the current context
            if part in current:
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                # Part not found
                return None
        
        return current
    
    @staticmethod
    def _cleanup_template_syntax(template_str: str) -> str:
        """
        Remove any remaining template syntax.
        
        Args:
            template_str: The template string
            
        Returns:
            The cleaned string
        """
        # Remove {% ... %} blocks
        result = re.sub(r'{%.*?%}', '', template_str)
        
        # Remove {{ ... }} expressions
        result = re.sub(r'{{.*?}}', '', result)
        
        # Fix common issues with docstrings and raw strings
        result = result.replace('\\"""', '"""')
        result = result.replace('\\\'\\\'\\\'', "'''")
        result = result.replace('""""', '"""')
        result = result.replace("''''", "'''")
        
        # Fix common issues with escape sequences
        result = result.replace('\\\\n', '\\n')
        result = result.replace('\\\\t', '\\t')
        result = result.replace('\\\\r', '\\r')
        
        return result
    
    @staticmethod
    def _fix_indentation(template_str: str) -> str:
        """
        Fix indentation issues in the rendered code.
        
        Args:
            template_str: The rendered string
            
        Returns:
            String with fixed indentation
        """
        # Split into lines
        lines = template_str.split('\n')
        
        # First pass: remove leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
            
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Find minimum indentation of non-empty lines
        min_indent = float('inf')
        for line in lines:
            stripped = line.lstrip()
            if stripped:  # Skip empty lines
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)
        
        # If min_indent is still infinity, there were no non-empty lines
        if min_indent == float('inf'):
            min_indent = 0
        
        # Second pass: re-indent all lines
        fixed_lines = []
        
        in_string = False
        string_delimiter = None
        indent_stack = []
        current_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Check for string delimiters
            if '"""' in stripped or "'''" in stripped:
                # Toggle multi-line string state if odd number of delimiters
                if stripped.count('"""') % 2 == 1:
                    in_string = not in_string
                    string_delimiter = '"""' if in_string else None
                elif stripped.count("'''") % 2 == 1:
                    in_string = not in_string
                    string_delimiter = "'''" if in_string else None
            
            # If in a string, keep original indentation
            if in_string:
                fixed_lines.append(line)
                continue
            
            # Handle dedent for closing brackets
            if stripped.startswith(('}', ')', ']', 'else:', 'elif', 'except:', 'finally:')):
                if current_indent > 0:
                    current_indent -= 1
            
            # Apply current indentation
            fixed_lines.append(' ' * (4 * current_indent) + stripped)
            
            # Handle indent for colon blocks
            if (stripped.endswith(':') and 
                not stripped.startswith(('import', 'from')) and
                not (':' in stripped and '"' in stripped and ',' in stripped)):  # Skip dict definitions
                current_indent += 1
                indent_stack.append(':')
            
            # Handle indent for opening brackets
            if stripped.endswith(('{', '[', '(')):
                current_indent += 1
                indent_stack.append(stripped[-1])
        
        return '\n'.join(fixed_lines)


class TemplateManager:
    """Manages template files and rendering"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the template manager.
        
        Args:
            template_dir: Directory containing template files
        """
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Default to 'templates' directory relative to this file
            self.template_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "skills" / "templates"
            
        self.template_files = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all template files from the template directory."""
        if not self.template_dir.exists():
            logger.warning(f"Template directory {self.template_dir} does not exist")
            return
            
        # Map architecture types to template files
        self.template_files = {
            "encoder-only": "encoder_only_template.py",
            "decoder-only": "decoder_only_template.py",
            "encoder-decoder": "encoder_decoder_template.py",
            "vision": "vision_template.py",
            "vision-text": "vision_text_template.py",
            "speech": "speech_template.py"
        }
        
        # Verify templates exist
        for architecture, template_file in self.template_files.items():
            template_path = self.template_dir / template_file
            if not template_path.exists():
                logger.warning(f"Template file {template_file} for {architecture} architecture not found")
        
    def get_template_content(self, architecture: str) -> Optional[str]:
        """
        Get the content of a template file.
        
        Args:
            architecture: The architecture type
            
        Returns:
            Template content, or None if not found
        """
        if architecture not in self.template_files:
            logger.warning(f"No template found for architecture: {architecture}")
            return None
            
        template_file = self.template_files[architecture]
        template_path = self.template_dir / template_file
        
        if not template_path.exists():
            logger.warning(f"Template file {template_file} not found")
            return None
            
        try:
            with open(template_path, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading template file {template_file}: {str(e)}")
            return None
    
    def render_template(self, architecture: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Render a template for a specific architecture.
        
        Args:
            architecture: The architecture type
            context: The context for rendering
            
        Returns:
            Rendered template, or None if template not found
        """
        template_content = self.get_template_content(architecture)
        if not template_content:
            return None
            
        # Render the template
        rendered = TemplateRenderer.render(template_content, context)
        return rendered


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

def generate_test(model_type: str, output_dir: str = "./generated_tests_improved") -> Dict[str, Any]:
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
    
    # Initialize template manager
    template_manager = TemplateManager()
    
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
        rendered = template_manager.render_template(architecture, context)
        if not rendered:
            error_msg = f"Failed to render template for architecture: {architecture}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model_type": model_type,
                "duration": time.time() - start_time
            }
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
    
    # Validate generated file
    is_valid, error, _ = SyntaxValidator.validate(rendered)
    validation_status = "valid" if is_valid else f"invalid: {error}"
    
    return {
        "success": True,
        "output_file": output_file,
        "model_type": model_type,
        "architecture": architecture,
        "duration": time.time() - start_time,
        "validation": validation_status,
        "is_valid": is_valid
    }

def generate_all_tests(output_dir: str = "./generated_tests_improved") -> Dict[str, Any]:
    """Generate tests for all model types.
    
    Args:
        output_dir: Directory to save the generated files
        
    Returns:
        Dict with generation results
    """
    start_time = time.time()
    results = {}
    successful = 0
    failed = 0
    
    # Get sample of models for each architecture
    for architecture, models in ARCHITECTURE_MAPPING.items():
        # Choose representative models for this architecture
        sample_models = models[:2]  # Just use the first 2 models for each architecture
        
        for model_type in sample_models:
            logger.info(f"Generating test for model type: {model_type} ({architecture})")
            result = generate_test(model_type, output_dir)
            results[model_type] = result
            
            if result["success"]:
                successful += 1
            else:
                failed += 1
    
    # Save summary
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "duration": time.time() - start_time,
        "results": results
    }
    
    summary_file = os.path.join(output_dir, "generation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Generation complete: {successful} successful, {failed} failed")
    logger.info(f"Summary written to: {summary_file}")
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test files from templates")
    parser.add_argument("--model", type=str, help="Model type to generate a test for")
    parser.add_argument("--all", action="store_true", help="Generate tests for all architectures")
    parser.add_argument("--output-dir", type=str, default="./generated_tests_improved", 
                        help="Directory to save the generated files")
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_tests(args.output_dir)
    elif args.model:
        result = generate_test(args.model, args.output_dir)
        if result["success"]:
            print(f"Successfully generated test for {args.model}")
            print(f"Output file: {result['output_file']}")
            print(f"Validation: {result['validation']}")
        else:
            print(f"Failed to generate test for {args.model}")
            print(f"Error: {result['error']}")
    else:
        parser.print_help()