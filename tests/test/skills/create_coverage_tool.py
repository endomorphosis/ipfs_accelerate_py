#!/usr/bin/env python3

"""
Create a comprehensive test coverage tool for HuggingFace models.

This script implements a utility for generating, validating, and tracking test
coverage for HuggingFace models using the IPFS Accelerate framework.
"""

import os
import sys
import json
import logging
import argparse
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = ROOT_DIR / "templates"
FIXED_TESTS_DIR = ROOT_DIR / "fixed_tests"
OUTPUT_DIR = ROOT_DIR / "output_tests"
MODEL_REGISTRY_PATH = ROOT_DIR / "huggingface_model_types.json"
COVERAGE_REPORT_PATH = ROOT_DIR / "MODEL_TEST_COVERAGE.md"

# Architecture types for model mapping
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert", "ernie", "rembert", "squeezebert", "roformer", "funnel", "layoutlm", "canine", "tapas", "xlnet"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "opt", "gemma", "ctrl"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan-t5", "blenderbot", "bigbird", "longformer", "reformer", "prophetnet"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "resnet", "segformer", "detr", "mask2former", "yolos", "sam", "bit", "dpt", "levit", "mlp-mixer", "mobilevit", "regnet", "efficientnet", "donut"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip", "blip-2", "donut", "pix2struct", "vilt", "vinvl", "chinese-clip", "align", "florence"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5", "sew", "unispeech", "wavlm", "data2vec-audio", "clap", "musicgen", "encodec"],
    "multimodal": ["llava", "git", "paligemma", "video-llava", "flava", "idefics", "imagebind", "flamingo", "usm", "data2vec", "data2vec-vision", "data2vec-text"]
}

class ModelCoverageTool:
    """Tool for generating, validating, and tracking HuggingFace model test coverage."""
    
    def __init__(self, output_dir: str = str(OUTPUT_DIR), template_dir: str = str(TEMPLATES_DIR)):
        """Initialize the model coverage tool.
        
        Args:
            output_dir: Directory where test files will be generated
            template_dir: Directory containing template files
        """
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)
        self.model_registry = self._load_model_registry()
        self.fixed_tests_dir = FIXED_TESTS_DIR
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load the model registry from JSON."""
        try:
            with open(MODEL_REGISTRY_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Model registry not found at {MODEL_REGISTRY_PATH}. Using empty registry.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error parsing model registry at {MODEL_REGISTRY_PATH}.")
            return {}
            
    def get_model_defaults(self, model_type: str) -> Dict[str, Any]:
        """Get default model configuration based on model type.
        
        Args:
            model_type: The model type to get defaults for
            
        Returns:
            Dictionary with default model configuration
        """
        # If we have the model in our registry, use that
        if model_type in self.model_registry:
            default_model = self.model_registry[model_type].get("default_model")
            models = self.model_registry[model_type].get("models", [])
            
            return {
                "default_model": default_model,
                "models": models,
            }
        
        # Fallback to hardcoded defaults for common models
        fallback_defaults = {
            "bert": "google-bert/bert-base-uncased",
            "gpt2": "datificate/gpt2-small-spanish",
            "t5": "amazon/chronos-t5-small",
            "vit": "google/vit-base-patch16-224-in21k",
            "roberta": "roberta-base",
            "distilbert": "distilbert-base-uncased",
            "albert": "albert-base-v2",
            "xlm-roberta": "xlm-roberta-base",
            "electra": "google/electra-small-discriminator",
            "deberta": "microsoft/deberta-base",
            "deberta-v2": "microsoft/deberta-v2-xlarge",
            "bart": "facebook/bart-base",
            "pegasus": "google/pegasus-xsum",
            "mbart": "facebook/mbart-large-cc25",
            "whisper": "openai/whisper-base.en",
            "wav2vec2": "facebook/wav2vec2-base",
            "hubert": "facebook/hubert-base-ls960",
            "llama": "meta-llama/Llama-2-7b-hf",
            "gpt-j": "EleutherAI/gpt-j-6B",
            "gpt-neo": "EleutherAI/gpt-neo-1.3B",
            "opt": "facebook/opt-125m",
            "falcon": "tiiuae/falcon-7b",
            "phi": "microsoft/phi-1_5",
            "mistral": "mistralai/Mistral-7B-v0.1",
            "mixtral": "mistralai/Mixtral-8x7B-v0.1",
            "mpt": "mosaicml/mpt-7b",
            "bloom": "bigscience/bloom-560m",
            "clip": "openai/clip-vit-base-patch32",
            "blip": "Salesforce/blip-image-captioning-base"
        }
        
        if model_type in fallback_defaults:
            return {
                "default_model": fallback_defaults[model_type],
                "models": [fallback_defaults[model_type]]
            }
            
        # If we still don't have a default, construct a reasonable one
        return {
            "default_model": f"{model_type}-base",
            "models": [f"{model_type}-base"]
        }
        
    def get_architecture_type(self, model_type: str) -> str:
        """Determine architecture type for a given model_type.
        
        Args:
            model_type: The model type to determine architecture for
            
        Returns:
            Architecture type as string
        """
        model_type_lower = model_type.lower()
        
        for arch_type, models in ARCHITECTURE_TYPES.items():
            if any(model in model_type_lower for model in models):
                return arch_type
                
        # Default to encoder-only as fallback
        logger.warning(f"Could not determine architecture type for {model_type}. Defaulting to encoder-only.")
        return "encoder-only"
        
    def get_template_path(self, model_type: str) -> Path:
        """Get appropriate template path for a model type.
        
        Args:
            model_type: The model type to get template for
            
        Returns:
            Path to the template file
        """
        # For now, let's use the minimal template for all models to ensure compatibility
        minimal_template_path = self.template_dir / "minimal_bert_template.py"
        if minimal_template_path.exists():
            logger.info(f"Using minimal_bert_template.py for all models for maximum compatibility")
            return minimal_template_path
            
        # If we need to use specific templates later, we can enable this code
        '''
        # Get the architecture type
        arch_type = self.get_architecture_type(model_type)
        
        # Map architecture type to template
        template_map = {
            "encoder-only": self.template_dir / "encoder_only_template.py",
            "decoder-only": self.template_dir / "decoder_only_template.py",
            "encoder-decoder": self.template_dir / "encoder_decoder_template.py",
            "vision": self.template_dir / "vision_template.py",
            "vision-text": self.template_dir / "vision_text_template.py",
            "speech": self.template_dir / "speech_template.py",
            "multimodal": self.template_dir / "multimodal_template.py"
        }
        
        # Get the template path
        template_path = template_map.get(arch_type)
        
        # Check if the template exists, otherwise use minimal_bert_template as fallback
        if not template_path or not template_path.exists():
            logger.warning(f"Template for {arch_type} not found. Using minimal_bert_template.py")
            template_path = self.template_dir / "minimal_bert_template.py"
        '''
            
        return minimal_template_path
        
    def to_valid_identifier(self, text: str) -> str:
        """Convert text to a valid Python identifier.
        
        Args:
            text: Text to convert
            
        Returns:
            Valid Python identifier
        """
        # Replace hyphens with underscores
        text = text.replace("-", "_")
        # Remove any other invalid characters
        text = re.sub(r'[^a-zA-Z0-9_]', '', text)
        # Ensure it doesn't start with a number
        if text and text[0].isdigit():
            text = '_' + text
        return text
        
    def get_pascal_case_identifier(self, text: str) -> str:
        """Convert a model name to PascalCase for class names.
        
        Args:
            text: Model name (potentially hyphenated)
            
        Returns:
            PascalCase identifier
        """
        # Split by hyphens and capitalize each part
        parts = text.split('-')
        return ''.join(part.capitalize() for part in parts)
        
    def preprocess_template(self, template_content: str) -> str:
        """Preprocess a template by normalizing indentation and adding special markers.
        
        Args:
            template_content: The template content to preprocess
            
        Returns:
            Processed template content
        """
        lines = template_content.split('\n')
        processed_lines = []
        
        in_try_block = False
        try_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Mark top-level try blocks with special markers
            if stripped == 'try:':
                in_try_block = True
                try_indent = len(line) - len(line.lstrip())
                processed_lines.append(line)
                
                # Add a special marker for the try block
                processed_lines.append(f"{' ' * (try_indent + 4)}# <TEMPLATE:TRY_BLOCK>")
                continue
                
            # Mark the end of try blocks
            if in_try_block and stripped.startswith(('except', 'finally')):
                in_try_block = False
                processed_lines.append(line)
                
                # Add a special marker for the except/finally block
                processed_lines.append(f"{' ' * (try_indent + 4)}# <TEMPLATE:EXCEPT_BLOCK>")
                continue
            
            # Add special markers for class definitions
            if stripped.startswith('class ') and stripped.endswith(':'):
                processed_lines.append(line)
                
                # Add a marker for class body
                indent = len(line) - len(line.lstrip())
                processed_lines.append(f"{' ' * (indent + 4)}# <TEMPLATE:CLASS_DEF>")
                continue
                
            # Add special markers for function definitions
            if stripped.startswith('def ') and stripped.endswith(':'):
                processed_lines.append(line)
                
                # Add a marker for function body
                indent = len(line) - len(line.lstrip())
                processed_lines.append(f"{' ' * (indent + 4)}# <TEMPLATE:FUNCTION_DEF>")
                continue
            
            # Ensure triple quotes are properly terminated
            if '"""' in line and line.count('"""') % 2 != 0:
                # Add a marker for docstrings
                processed_lines.append(line)
                processed_lines.append("# <TEMPLATE:DOCSTRING>")
                continue
                
            # Process normal lines
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
        
    def token_based_replace(self, template: str, replacements: Dict[str, str]) -> str:
        """Replace tokens in the template using a more robust approach that preserves code structure.
        
        Args:
            template: The template content
            replacements: Dictionary of token -> replacement mappings
            
        Returns:
            The processed template with replacements applied
        """
        # Get lines for processing
        lines = template.split('\n')
        processed_lines = []
        
        # Track state
        in_string = False
        string_delimiter = None
        in_comment = False
        
        for line in lines:
            # Skip special template marker lines
            if '<TEMPLATE:' in line:
                continue
                
            # Process the line character by character
            processed_line = ''
            i = 0
            while i < len(line):
                # Check if we're at the start of a comment
                if line[i:i+1] == '#' and not in_string:
                    in_comment = True
                    processed_line += line[i]
                    i += 1
                    continue
                    
                # Check if we're at the start of a string
                if line[i:i+1] in ['"', "'"] and not in_string:
                    in_string = True
                    string_delimiter = line[i:i+1]
                    processed_line += line[i]
                    i += 1
                    continue
                    
                # Check if we're at the end of a string
                if in_string and line[i:i+len(string_delimiter)] == string_delimiter:
                    in_string = False
                    string_delimiter = None
                    processed_line += line[i]
                    i += 1
                    continue
                
                # If we're in a string or comment, don't do replacements
                if in_string or in_comment:
                    processed_line += line[i]
                    i += 1
                    continue
                    
                # Check for token replacements
                replaced = False
                for token, replacement in replacements.items():
                    if i + len(token) <= len(line) and line[i:i+len(token)] == token:
                        # Only replace whole words/identifiers
                        next_char = line[i+len(token):i+len(token)+1] if i+len(token) < len(line) else None
                        prev_char = line[i-1:i] if i > 0 else None
                        
                        # Check if this is a whole word/identifier
                        is_whole_word = (
                            (not next_char or not (next_char.isalnum() or next_char == '_')) and
                            (not prev_char or not (prev_char.isalnum() or prev_char == '_'))
                        )
                        
                        if is_whole_word:
                            processed_line += replacement
                            i += len(token)
                            replaced = True
                            break
                
                # If no replacement was made, keep the original character
                if not replaced:
                    processed_line += line[i]
                    i += 1
                    
            # Add the processed line to the result
            processed_lines.append(processed_line)
            
            # Reset comment state at the end of the line
            in_comment = False
        
        return '\n'.join(processed_lines)
    
    def fix_try_except_blocks(self, content: str) -> str:
        """Fix indentation in try/except blocks with a more robust approach.
        
        Args:
            content: The content to fix
            
        Returns:
            Fixed content
        """
        lines = content.split('\n')
        processed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Handle try blocks
            if stripped == 'try:':
                processed_lines.append(line)
                
                # Look ahead to ensure the next non-empty line is properly indented
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    processed_lines.append(lines[j])
                    j += 1
                    
                # If we found a line that needs indentation
                if j < len(lines) and not lines[j].lstrip().startswith(('except', 'finally')):
                    indent = len(line) - len(line.lstrip())
                    processed_lines.append(' ' * (indent + 4) + lines[j].lstrip())
                    i = j
                else:
                    # We didn't find anything to indent, include the next line as-is
                    if j < len(lines):
                        processed_lines.append(lines[j])
                        i = j
            else:
                processed_lines.append(line)
                
            i += 1
            
        return '\n'.join(processed_lines)
        
    def fix_unterminated_triple_quotes(self, content: str) -> str:
        """Fix unterminated triple quotes in the content more robustly.
        
        Args:
            content: The content to fix
            
        Returns:
            Fixed content
        """
        lines = content.split('\n')
        triple_quote_count = content.count('"""')
        single_triple_quote_count = content.count("'''")
        
        # Handle regular triple quotes """
        if triple_quote_count % 2 != 0:
            logger.info(f"Odd number of triple quotes found: {triple_quote_count}, fixing...")
            in_docstring = False
            docstring_start_line = None
            docstring_start_indent = 0
            
            for i, line in enumerate(lines):
                # Count triple quotes on this line
                quotes_in_line = line.count('"""')
                
                # Skip lines with even number of quotes (they open and close on same line)
                if quotes_in_line > 0 and quotes_in_line % 2 == 0:
                    continue
                    
                # Handle lines with odd number of quotes
                if quotes_in_line % 2 != 0:
                    if not in_docstring:
                        # Opening a docstring
                        in_docstring = True
                        docstring_start_line = i
                        docstring_start_indent = len(line) - len(line.lstrip())
                    else:
                        # Closing a docstring
                        in_docstring = False
                        docstring_start_line = None
            
            # If we're still in a docstring at the end, add closing quotes with proper indentation
            if in_docstring and docstring_start_line is not None:
                # Get same indentation as the opening quote line
                indent_str = ' ' * docstring_start_indent
                
                # Check if the last line has content and needs a newline before closing
                if lines[-1].strip():
                    lines.append(f"{indent_str}\"\"\"")
                else:
                    # If last line is already empty, just add the quotes with indentation
                    lines[-1] = f"{indent_str}\"\"\"" 
                    
                logger.info(f"Added missing closing triple quotes with matching indentation from line {docstring_start_line+1}")
        
        # Handle single triple quotes '''
        if single_triple_quote_count % 2 != 0:
            logger.info(f"Odd number of single triple quotes found: {single_triple_quote_count}, fixing...")
            in_docstring = False
            docstring_start_line = None
            docstring_start_indent = 0
            
            for i, line in enumerate(lines):
                # Count triple quotes on this line
                quotes_in_line = line.count("'''")
                
                # Skip lines with even number of quotes (they open and close on same line)
                if quotes_in_line > 0 and quotes_in_line % 2 == 0:
                    continue
                    
                # Handle lines with odd number of quotes
                if quotes_in_line % 2 != 0:
                    if not in_docstring:
                        # Opening a docstring
                        in_docstring = True
                        docstring_start_line = i
                        docstring_start_indent = len(line) - len(line.lstrip())
                    else:
                        # Closing a docstring
                        in_docstring = False
                        docstring_start_line = None
            
            # If we're still in a docstring at the end, add closing quotes with proper indentation
            if in_docstring and docstring_start_line is not None:
                # Get same indentation as the opening quote line
                indent_str = ' ' * docstring_start_indent
                
                # Check if the last line has content and needs a newline before closing
                if lines[-1].strip():
                    lines.append(f"{indent_str}'''")
                else:
                    # If last line is already empty, just add the quotes with indentation
                    lines[-1] = f"{indent_str}'''" 
                    
                logger.info(f"Added missing closing single triple quotes with matching indentation from line {docstring_start_line+1}")
        
        return '\n'.join(lines)
        
    def fix_docstring_method_definition_issues(self, content: str) -> str:
        """Fix common issues with docstrings and method definitions.
        
        Args:
            content: The content to fix
            
        Returns:
            Fixed content
        """
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for docstring followed by method definition on same line
            if '"""def ' in line:
                parts = line.split('"""')
                if len(parts) >= 2:
                    # Add the docstring part
                    fixed_lines.append(parts[0] + '"""')
                    
                    # Calculate the indentation
                    indent = len(line) - len(line.lstrip())
                    
                    # Add the method definition on a new line with proper indentation
                    fixed_lines.append(' ' * indent + parts[1].strip())
                    logger.info(f"Fixed docstring and method definition combined on line {i+1}")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
                
            i += 1
            
        return '\n'.join(fixed_lines)
        
    def fix_indentation_issues(self, content: str) -> str:
        """Fix common indentation issues.
        
        Args:
            content: The content to fix
            
        Returns:
            Fixed content
        """
        lines = content.split('\n')
        
        # First pass: track and fix blocks
        in_class = False
        in_method = False
        in_try_block = False
        in_except_block = False
        in_finally_block = False
        class_indent = 0
        method_indent = 0
        try_indent = 0
        
        # First, fix class docstring + method definition issues
        i = 0
        while i < len(lines)-1:
            # Look for class docstring followed by method definition issues
            if lines[i].strip().startswith('"""') and lines[i].strip().endswith('"""') and lines[i+1].strip().startswith('def '):
                # Check if the docstring and method definition are on the same line
                if '"""def ' in lines[i]:
                    parts = lines[i].split('"""')
                    if len(parts) >= 2:
                        # Extract the docstring and method definition
                        docstring = parts[0] + '"""'
                        method_def = parts[1].strip()
                        
                        # Fix by splitting into two lines
                        lines[i] = docstring
                        lines.insert(i+1, method_def)
                        logger.info(f"Fixed docstring and method definition combined on line {i+1}")
                        i += 1  # Skip the newly inserted line
            i += 1
        
        # Second pass: scan for blocks and fix indentation
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Check for missing newline between docstring and method definition
            if i < len(lines)-1 and stripped.endswith('"""') and not stripped.startswith('"""'):
                next_line = lines[i+1].strip()
                if next_line.startswith('def '):
                    # Make sure there's an empty line between docstring and method
                    lines.insert(i+1, '')
                    logger.info(f"Added missing empty line after docstring on line {i+1}")
                    i += 1  # Skip the newly inserted empty line
            
            # Track class definition
            if stripped.startswith('class ') and stripped.endswith(':'):
                in_class = True
                class_indent = indent
            
            # Track method definition
            elif in_class and stripped.startswith('def ') and stripped.endswith(':'):
                in_method = True
                method_indent = indent
                
            # Track try blocks - correctly handle nested try blocks
            elif stripped == 'try:':
                in_try_block = True
                try_indent = indent
                
                # Look ahead to ensure the next content line is properly indented
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                    
                # If we found a content line with bad indentation, fix it
                if j < len(lines):
                    next_indent = len(lines[j]) - len(lines[j].lstrip())
                    if lines[j].strip() and next_indent <= try_indent:
                        # Fix the indentation
                        lines[j] = ' ' * (try_indent + 4) + lines[j].lstrip()
                        logger.info(f"Fixed indentation after try: on line {j+1}")
            
            # Track except blocks
            elif in_try_block and stripped.startswith('except ') and stripped.endswith(':'):
                in_try_block = False
                in_except_block = True
                
                # Look ahead to ensure the next content line is properly indented
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                    
                # If we found a content line with bad indentation, fix it
                if j < len(lines):
                    next_indent = len(lines[j]) - len(lines[j].lstrip())
                    if lines[j].strip() and next_indent <= try_indent:
                        # Fix the indentation
                        lines[j] = ' ' * (try_indent + 4) + lines[j].lstrip()
                        logger.info(f"Fixed indentation after except: on line {j+1}")
            
            # Track finally blocks
            elif (in_try_block or in_except_block) and stripped == 'finally:':
                in_try_block = False
                in_except_block = False
                in_finally_block = True
                
                # Look ahead to ensure the next content line is properly indented
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                    
                # If we found a content line with bad indentation, fix it
                if j < len(lines):
                    next_indent = len(lines[j]) - len(lines[j].lstrip())
                    if lines[j].strip() and next_indent <= try_indent:
                        # Fix the indentation
                        lines[j] = ' ' * (try_indent + 4) + lines[j].lstrip()
                        logger.info(f"Fixed indentation after finally: on line {j+1}")
                
            # End of block detection
            elif in_except_block and indent <= try_indent and stripped and not stripped.startswith(('except ', 'finally:')):
                in_except_block = False
            elif in_finally_block and indent <= try_indent and stripped:
                in_finally_block = False
                
            # Fix indentation in method body
            elif in_method and indent < method_indent + 4 and stripped and not stripped.startswith(('def ', 'class ')):
                # This is a method body line with incorrect indentation
                lines[i] = ' ' * (method_indent + 4) + stripped
                logger.info(f"Fixed indentation in method body on line {i+1}")
                
            # Fix indentation in try/except blocks
            elif in_try_block and indent < try_indent + 4 and stripped and not stripped.startswith(('except ', 'finally:')):
                # This is a try block line with incorrect indentation
                lines[i] = ' ' * (try_indent + 4) + stripped
                logger.info(f"Fixed indentation in try block on line {i+1}")
                
            elif in_except_block and indent < try_indent + 4 and stripped and not stripped.startswith(('try:', 'except ', 'finally:')):
                # This is an except block line with incorrect indentation
                lines[i] = ' ' * (try_indent + 4) + stripped
                logger.info(f"Fixed indentation in except block on line {i+1}")
                
            elif in_finally_block and indent < try_indent + 4 and stripped:
                # This is a finally block line with incorrect indentation
                lines[i] = ' ' * (try_indent + 4) + stripped
                logger.info(f"Fixed indentation in finally block on line {i+1}")
            
            # Fix docstring-method definition run-on issues
            if '"""def ' in line:
                parts = line.split('"""')
                if len(parts) >= 2:
                    # Fix by splitting into two lines
                    lines[i] = parts[0] + '"""'
                    indent_str = ' ' * indent
                    lines.insert(i+1, indent_str + parts[1].strip())
                    logger.info(f"Fixed docstring and method definition combined on line {i+1}")
                    i += 1  # Skip the newly inserted line
            
            i += 1
        
        content = '\n'.join(lines)
        return content
        
    def final_cleanup(self, content: str) -> str:
        """Perform a final cleanup of the entire file to catch any remaining issues.
        
        Args:
            content: The content to clean up
            
        Returns:
            Cleaned content
        """
        lines = content.split('\n')
        cleaned_lines = []
        
        # Track state
        in_docstring = False
        docstring_start_line = None
        docstring_indent = 0
        
        for i, line in enumerate(lines):
            # Skip empty or whitespace-only lines at the start of the file
            if not cleaned_lines and not line.strip():
                continue
                
            # Fix indentation around docstrings
            if '"""' in line:
                # Count occurrence in this line
                quotes_count = line.count('"""')
                
                # Handle toggle between enter/exit docstring state
                if quotes_count % 2 != 0:
                    if not in_docstring:
                        # Start of docstring
                        in_docstring = True
                        docstring_start_line = i
                        docstring_indent = len(line) - len(line.lstrip())
                    else:
                        # End of docstring
                        in_docstring = False
            
            # Handle known problem: indentation in class __init__ method
            if line.strip().startswith('def __init__') and not line.strip().endswith(':'):
                # Add missing colon
                line = line.rstrip() + ':'
                logger.info(f"Added missing colon to __init__ method on line {i+1}")
                
            # Handle broken try blocks
            if line.strip() == 'try:' and i + 1 < len(lines):
                next_line = lines[i + 1]
                if not next_line.strip() or next_line.strip().startswith(('except', 'finally')):
                    # Try block is empty, add a pass statement
                    indent = len(line) - len(line.lstrip())
                    cleaned_lines.append(line)
                    cleaned_lines.append(' ' * (indent + 4) + 'pass')
                    logger.info(f"Added missing pass statement in empty try block on line {i+1}")
                    continue
                    
            # Fix common indent issues in if blocks at the end of file
            if line.strip().startswith('if ') and i + 1 >= len(lines):
                # if statement at the end with no body
                indent = len(line) - len(line.lstrip())
                cleaned_lines.append(line)
                cleaned_lines.append(' ' * (indent + 4) + 'pass')
                logger.info(f"Added missing pass statement to if block at EOF on line {i+1}")
                continue
                
            # Fix the common issue with main() at the end of file
            if i > 0 and lines[i-1].strip() == 'if __name__ == "__main__":' and line.strip() == 'sys.exit(main())':
                # Ensure proper indentation for the main function call
                indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                cleaned_lines.append(' ' * (indent + 4) + line.strip())
                logger.info(f"Fixed indentation of main function call on line {i+1}")
                continue
            
            # Keep the line (default case)
            cleaned_lines.append(line)
        
        # Check for common EOF issues
        if not cleaned_lines[-1]:
            # Remove trailing empty lines
            while cleaned_lines and not cleaned_lines[-1].strip():
                cleaned_lines.pop()
                
        # Always end with a newline
        if cleaned_lines and cleaned_lines[-1].strip():
            cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines)
        
    def post_process_generated_file(self, content: str) -> Tuple[str, bool, Optional[Exception]]:
        """Perform post-processing on the generated file to ensure it's valid Python.
        
        Args:
            content: The content to process
            
        Returns:
            Tuple with (processed_content, success, error)
        """
        # 1. Fix indentation in try/except blocks
        content = self.fix_try_except_blocks(content)
        
        # 2. Fix unterminated triple quotes
        content = self.fix_unterminated_triple_quotes(content)
        
        # 3. Fix docstring-method definition issues
        content = self.fix_docstring_method_definition_issues(content)
        
        # 4. Fix general indentation issues
        content = self.fix_indentation_issues(content)
        
        # 5. Final cleanup
        content = self.final_cleanup(content)
        
        # 6. Validate syntax
        try:
            compile(content, "<string>", 'exec')
            return content, True, None
        except SyntaxError as e:
            # If there's still a syntax error, return the error details
            return content, False, e
            
    def generate_test_file(self, model_type: str) -> Tuple[str, bool]:
        """Generate a test file for a specific model type.
        
        Args:
            model_type: The model type to generate test for
            
        Returns:
            Tuple with (output_file_path, success)
        """
        try:
            # Get model defaults
            model_defaults = self.get_model_defaults(model_type)
            default_model = model_defaults.get("default_model")
            
            # Fix hyphenated model names for valid Python identifiers
            model_family_valid = self.to_valid_identifier(model_type)
            
            # Define test class name using PascalCase
            model_pascal_case = self.get_pascal_case_identifier(model_type)
            test_class = f"Test{model_pascal_case}Models"
            
            # Get architecture type
            arch_type = self.get_architecture_type(model_type)
            
            # Get appropriate template
            template_path = self.get_template_path(model_type)
            logger.info(f"Using template {template_path.name} for {model_type}")
            
            # Generate the module name
            module_name = f"test_hf_{model_family_valid}"
            
            # Read the template
            with open(template_path, "r") as f:
                template_content = f.read()
                
            # Step 1: Preprocess the template
            preprocessed_template = self.preprocess_template(template_content)
            
            # Step 2: Define token replacements
            model_upper = model_family_valid.upper()
            
            # Set up replacements based on template type
            if "bert" in template_path.name.lower():
                # Using bert template
                replacements = {
                    # Registry name
                    "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                    
                    # Class name
                    "TestBertModels": test_class,
                    
                    # Model types
                    "bert-base-uncased": default_model,
                    "google-bert/bert-base-uncased": default_model,
                    
                    # Class identifiers
                    "BERT": model_pascal_case,
                    
                    # Lowercase identifiers
                    "bert": model_family_valid,
                    
                    # File paths
                    "hf_bert_": f"hf_{model_family_valid}_",
                    
                    # Other references
                    "bert_tester": f"{model_family_valid}_tester"
                }
            elif "vit" in template_path.name.lower():
                # Using vision template
                replacements = {
                    # Registry name
                    "VIT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                    
                    # Class name
                    "TestVitModels": test_class,
                    
                    # Model types
                    "google/vit-base-patch16-224": default_model,
                    
                    # Class identifiers
                    "ViT": model_pascal_case,
                    
                    # Lowercase identifiers
                    "vit": model_family_valid,
                    
                    # File paths
                    "hf_vit_": f"hf_{model_family_valid}_",
                    
                    # Other references
                    "vit_tester": f"{model_family_valid}_tester"
                }
            else:
                # Generic template
                primary_model = template_path.stem.split('_')[0]
                if '-' in primary_model:
                    primary_model = self.to_valid_identifier(primary_model)
                    
                primary_model_pascal = self.get_pascal_case_identifier(primary_model)
                primary_model_upper = primary_model.upper()
                
                replacements = {
                    # Registry name
                    f"{primary_model_upper}_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                    
                    # Class name
                    f"Test{primary_model_pascal}Models": test_class,
                    
                    # Class identifiers
                    primary_model_pascal: model_pascal_case,
                    
                    # Lowercase identifiers
                    primary_model: model_family_valid,
                    
                    # File paths
                    f"hf_{primary_model}_": f"hf_{model_family_valid}_",
                    
                    # Other references
                    f"{primary_model}_tester": f"{model_family_valid}_tester"
                }
                
            # Step 3: Apply token-based replacements
            content = self.token_based_replace(preprocessed_template, replacements)
            
            # Fix specific instances that might not be caught by token replacement
            content = content.replace(f"bert_tester.device", f"{model_family_valid}_tester.device")
            
            # Fix the indentation of the if __name__ == "__main__" block
            if "if __name__ == \"__main__\":" in content:
                content = content.replace(
                    "    if __name__ == \"__main__\":",
                    "if __name__ == \"__main__\":"
                )
                content = content.replace(
                    "        sys.exit(main())",
                    "    sys.exit(main())"
                )
            
            # Step 4: Post-process the generated file
            content, success, error = self.post_process_generated_file(content)
            
            # Define output path
            output_file = self.output_dir / module_name
            if not str(output_file).endswith(".py"):
                output_file = output_file.with_suffix(".py")
                
            # Write the file
            with open(output_file, "w") as f:
                f.write(content)
                
            if not success:
                logger.error(f"Generated file has syntax errors: {error}")
                return str(output_file), False
                
            logger.info(f"Successfully generated test file: {output_file}")
            return str(output_file), True
            
        except Exception as e:
            logger.error(f"Error generating test file for {model_type}: {e}")
            traceback.print_exc()
            return "", False
    
    def generate_batch(self, model_types: List[str]) -> Dict[str, bool]:
        """Generate test files for a batch of model types.
        
        Args:
            model_types: List of model types to generate
            
        Returns:
            Dictionary mapping model types to success status
        """
        results = {}
        
        for model_type in model_types:
            output_file, success = self.generate_test_file(model_type)
            results[model_type] = success
            
        return results
            
    def update_coverage_report(self) -> None:
        """Update the coverage report markdown file."""
        # Get all test files
        all_test_files = []
        for path in [self.fixed_tests_dir, self.output_dir]:
            if path.exists():
                all_test_files.extend(list(path.glob("test_hf_*.py")))
                
        # Count by architecture type
        arch_counts = {k: [] for k in ARCHITECTURE_TYPES.keys()}
        hyphenated_models = []
        
        for test_file in all_test_files:
            # Extract model name from file name (test_hf_bert.py -> bert)
            file_stem = test_file.stem
            model_parts = file_stem.split("_")
            if len(model_parts) >= 3:
                model_name = "_".join(model_parts[2:])
                
                # Check if it's a hyphenated model
                if "-" in model_name:
                    # This might be a hyphenated model that was converted to underscore
                    for arch_type, models in ARCHITECTURE_TYPES.items():
                        for m in models:
                            valid_id = self.to_valid_identifier(m)
                            if valid_id == model_name and "-" in m:
                                hyphenated_models.append((m, valid_id))
                                
                # Determine the architecture type
                arch_found = False
                for arch_type, models in ARCHITECTURE_TYPES.items():
                    # Convert model_name back to potential hyphenated form
                    for m in hyphenated_models:
                        if m[1] == model_name:
                            model_name_or_hyphenated = m[0]
                            break
                    else:
                        model_name_or_hyphenated = model_name.replace("_", "-")
                        
                    if any(model in model_name_or_hyphenated for model in models):
                        arch_counts[arch_type].append(model_name)
                        arch_found = True
                        break
                        
                if not arch_found:
                    # Default to encoder-only if we can't determine
                    arch_counts["encoder-only"].append(model_name)
                    
        # Generate the report
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        
        report = f"""# Model Test Coverage Report

Generated on: {now}

## Summary

Total test files: {len(all_test_files)}

## Coverage by Architecture Type

"""
        
        # Add architecture coverage
        for arch_type, models in arch_counts.items():
            report += f"### {arch_type} ({len(models)} models)\n\n"
            
            if models:
                for model in sorted(models):
                    report += f"- `{model}`\n"
            else:
                report += "*No tests implemented yet*\n"
                
            report += "\n"
            
        # Add hyphenated models section
        if hyphenated_models:
            report += f"## Hyphenated Models ({len(hyphenated_models)} models)\n\n"
            for orig, converted in sorted(hyphenated_models):
                report += f"- `{orig}` → `{converted}`\n"
                
            report += "\n"
            
        # Add footer
        report += """---

This report will be automatically updated by running:
```bash
python create_coverage_tool.py --update-report
```"""
        
        # Write the report
        with open(COVERAGE_REPORT_PATH, "w") as f:
            f.write(report)
            
        logger.info(f"Updated coverage report: {COVERAGE_REPORT_PATH}")
        
    def get_all_huggingface_models(self) -> List[str]:
        """Get a list of all model types from ARCHITECTURE_TYPES.
        
        Returns:
            List of all model types
        """
        all_models = []
        for models in ARCHITECTURE_TYPES.values():
            all_models.extend(models)
            
        return sorted(list(set(all_models)))
            
    def find_missing_tests(self) -> List[str]:
        """Find model types that don't have tests yet.
        
        Returns:
            List of model types without tests
        """
        # Get all test files
        all_test_files = []
        for path in [self.fixed_tests_dir, self.output_dir]:
            if path.exists():
                all_test_files.extend(list(path.glob("test_hf_*.py")))
                
        # Extract model names from test files
        tested_models = []
        for test_file in all_test_files:
            # Extract model name from file name (test_hf_bert.py -> bert)
            file_stem = test_file.stem
            model_parts = file_stem.split("_")
            if len(model_parts) >= 3:
                model_name = "_".join(model_parts[2:])
                tested_models.append(model_name.replace("_", "-"))
                
        # Get all model types
        all_models = self.get_all_huggingface_models()
        
        # Find missing models
        missing_models = [m for m in all_models if m not in tested_models]
        
        return missing_models

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HuggingFace Model Test Coverage Tool")
    
    # Main commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", type=str, help="Generate test for a specific model type")
    group.add_argument("--batch", type=int, help="Generate tests for a batch of models")
    group.add_argument("--list-missing", action="store_true", help="List model types without tests")
    group.add_argument("--update-report", action="store_true", help="Update the coverage report")
    group.add_argument("--list-all", action="store_true", help="List all model types")
    
    # Optional arguments
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory for test files")
    parser.add_argument("--temp-dir", type=str, default="/tmp/model_tests", help="Temporary directory for test files")
    
    args = parser.parse_args()
    
    # Initialize the tool
    tool = ModelCoverageTool(output_dir=args.output_dir)
    
    # Execute the requested command
    if args.generate:
        output_file, success = tool.generate_test_file(args.generate)
        if success:
            print(f"Successfully generated test file: {output_file}")
        else:
            print(f"Failed to generate test file for {args.generate}")
            sys.exit(1)
            
    elif args.batch:
        # Find missing models
        missing_models = tool.find_missing_tests()
        
        if not missing_models:
            print("No missing models found. All model types have tests!")
            sys.exit(0)
            
        # Take a batch of missing models
        batch_size = min(args.batch, len(missing_models))
        batch = missing_models[:batch_size]
        
        print(f"Generating tests for {batch_size} models: {', '.join(batch)}")
        
        # Generate tests
        results = tool.generate_batch(batch)
        
        # Print summary
        successful = sum(1 for success in results.values() if success)
        print(f"Generated {successful}/{batch_size} test files successfully")
        
        # Update the coverage report
        tool.update_coverage_report()
        
    elif args.list_missing:
        missing_models = tool.find_missing_tests()
        
        if not missing_models:
            print("No missing models found. All model types have tests!")
        else:
            print(f"Found {len(missing_models)} model types without tests:")
            for model in sorted(missing_models):
                print(f"  - {model}")
                
    elif args.update_report:
        tool.update_coverage_report()
        print(f"Updated coverage report: {COVERAGE_REPORT_PATH}")
        
    elif args.list_all:
        all_models = tool.get_all_huggingface_models()
        print(f"Found {len(all_models)} model types:")
        for model in sorted(all_models):
            print(f"  - {model}")

if __name__ == "__main__":
    main()