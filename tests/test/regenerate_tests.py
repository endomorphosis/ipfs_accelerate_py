#!/usr/bin/env python3
"""
Script to regenerate test files for HuggingFace model families.
"""

import os
import sys
import re
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"regenerate_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Basic model families that have been fixed and tested
BASE_MODEL_FAMILIES = ["bert", "gpt2", "t5", "vit"]

# Extended model families that are expected to work with the fixed generator
# These model families use the same architecture type as the base families
EXTENDED_MODEL_FAMILIES = {
    "encoder_only": [  # Same architecture as BERT, ViT
        "roberta", "distilbert", "albert", "electra", 
        "deit", "beit", "convnext", "clip"
    ],
    "decoder_only": [  # Same architecture as GPT-2
        "gpt_neo", "gpt_neox", "gptj", "opt", "llama", "bloom"
    ],
    "encoder_decoder": [  # Same architecture as T5
        "bart", "pegasus", "mbart", "mt5", "longt5"
    ],
    "vision": [  # Same modality as ViT
        "detr", "swin", "convnext"
    ],
    "audio": [  # Needs audio-specific handling
        "wav2vec2", "hubert", "whisper"
    ],
    "multimodal": [  # Needs multiple input handling
        "clip", "blip", "llava"
    ]
}

def get_available_model_families():
    """Get list of all potential model families from the fixed generator"""
    # This is hardcoded for now, but could be extracted from test_generator_fixed.py
    all_families = set(BASE_MODEL_FAMILIES)
    for category in EXTENDED_MODEL_FAMILIES.values():
        all_families.update(category)
    return sorted(list(all_families))

def fix_indentation_issues(file_path):
    """
    Fix common indentation issues in generated test files by applying 
    our enhanced multi-stage cleanup approach.
    
    This function:
    1. Applies targeted regex replacements for common indentation patterns
    2. Ensures spacing between method definitions
    3. Normalizes indentation throughout the file
    
    Args:
        file_path: Path to the test file to fix
    """
    try:
        # First try to use our enhanced cleanup script
        from cleanup_test_files import cleanup_test_file
        cleanup_test_file(file_path)
        logger.info(f"Applied enhanced indentation cleanup to {file_path}")
        return
    except ImportError:
        logger.warning("Enhanced cleanup_test_files module not found, using fallback method")
        
    # Fallback implementation if the enhanced cleanup isn't available
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply a series of regex fixes for common issues
    
    # 1. Fix method boundaries with proper spacing
    content = re.sub(r'return results\s+def', 'return results\n\n    def', content)
    content = re.sub(r'self\.performance_stats = \{\}\s+def', 
                    'self.performance_stats = {}\n\n    def', content)
    
    # 2. Fix dependency check indentation - normalize to 8 spaces
    content = re.sub(r'(\s+)if not HAS_(\w+):', r'        if not HAS_\2:', content)
    
    # 3. Fix nested indentation in control structures (if/else/elif)
    content = re.sub(r'(\s+)else:\n\s+results', r'\1else:\n\1    results', content)
    
    # 4. Fix method declarations
    content = re.sub(r'^(\s*)def test_(\w+)', r'    def test_\2', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s*)def run_tests', r'    def run_tests', content, flags=re.MULTILINE)
    
    # 5. Apply string replacement for common patterns
    string_fixes = [
        # Method declarations
        ('return results    def', 'return results\n\n    def'),
        
        # Error handling indentation
        ('            else:\n            results', '            else:\n                results'),
        ('            elif', '            elif'),
        
        # Common statement indentation
        ('    if device', '        if device'),
        ('    for _ in range', '        for _ in range'),
        ('    try:', '        try:'),
        ('    logger.', '        logger.'),
        
        # Nested indentation in try/except blocks
        ('        try:\n        with', '        try:\n            with'),
        ('        except Exception:\n        pass', '        except Exception:\n            pass'),
    ]
    
    # Apply all string fixes
    for old, new in string_fixes:
        content = content.replace(old, new)
    
    # Extract and fix specific methods using pattern matching
    
    # Fix test_pipeline method
    pipeline_pattern = r'(\s+def test_pipeline\(self,.*?(?=\s+def test_from_pretrained|\s+def run_tests|$))'
    pipeline_match = re.search(pipeline_pattern, content, re.DOTALL)
    if pipeline_match:
        original_method = pipeline_match.group(0)
        fixed_method = fix_method_content(original_method, 'test_pipeline')
        content = content.replace(original_method, fixed_method)
    
    # Fix test_from_pretrained method
    from_pretrained_pattern = r'(\s+def test_from_pretrained\(self,.*?(?=\s+def run_tests|$))'
    from_pretrained_match = re.search(from_pretrained_pattern, content, re.DOTALL)
    if from_pretrained_match:
        original_method = from_pretrained_match.group(0)
        fixed_method = fix_method_content(original_method, 'test_from_pretrained')
        content = content.replace(original_method, fixed_method)
    
    # Fix run_tests method
    run_tests_pattern = r'(\s+def run_tests\(self,.*?(?=\s+def save_results|$))'
    run_tests_match = re.search(run_tests_pattern, content, re.DOTALL)
    if run_tests_match:
        original_method = run_tests_match.group(0)
        fixed_method = fix_method_content(original_method, 'run_tests')
        content = content.replace(original_method, fixed_method)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Fixed indentation issues in {file_path}")

def fix_method_content(method_text, method_name):
    """
    Fix the indentation of a method's content.
    
    Args:
        method_text: The method text to fix
        method_name: The name of the method
        
    Returns:
        The fixed method text with proper indentation
    """
    lines = method_text.split('\n')
    fixed_lines = []
    
    # First line should be properly indented with 4 spaces for class methods
    first_line = lines[0]
    if not first_line.strip().startswith('def'):
        return method_text  # Not a method definition, return as is
    
    # Ensure the method definition has exactly 4 spaces
    fixed_first_line = f"    def {method_name}(self,"
    for part in first_line.split('def')[1].split('(self,')[1:]:
        fixed_first_line += f"(self,{part}"
    fixed_lines.append(fixed_first_line)
    
    # Process the rest of the method body with 8 spaces (4 for class + 4 for method content)
    for line in lines[1:]:
        if not line.strip():
            # Empty line
            fixed_lines.append(line)
            continue
            
        content = line.strip()
        # Determine the correct indentation
        if content.startswith('"""') or content.endswith('"""'):
            # Docstring line - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('if ') or content.startswith('for ') or content.startswith('try:') or content.startswith('else:') or content.startswith('elif '):
            # Control flow - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('return '):
            # Return statement - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('results[') or content.startswith('self.'):
            # Assignment to results or self - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('logger.'):
            # Logging - 8 spaces
            fixed_lines.append(f"        {content}")
        elif content.startswith('# '):
            # Comments - 8 spaces for top-level method comments
            fixed_lines.append(f"        {content}")
        elif content.startswith('}') or content.startswith(']'):
            # Closing brackets - 8 spaces for method level collections
            fixed_lines.append(f"        {content}")
        else:
            # Default indentation for nested blocks - 12 spaces
            # This is a heuristic that works well for this specific code structure
            fixed_lines.append(f"            {content}")
    
    return '\n'.join(fixed_lines)

def regenerate_test_file(family, generator_path, output_dir):
    """Regenerate a test file for a specific model family"""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        generator_path,
        "--family", family,
        "--output", output_dir
    ]
    
    logger.info(f"Regenerating test file for {family}...")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        output_path = os.path.join(output_dir, f"test_hf_{family}.py")
        
        # Post-process the file to fix any remaining indentation issues
        if os.path.exists(output_path):
            fix_indentation_issues(output_path)
            
        logger.info(f"✅ Successfully regenerated test file for {family}: {output_path}")
        return True
    else:
        logger.error(f"❌ Failed to regenerate test file for {family}")
        logger.error(f"Error: {result.stderr}")
        return False

def add_model_to_generator(family, template_family, generator_path):
    """
    Add a new model family to the generator based on an existing template.
    
    Args:
        family: Name of the new model family to add
        template_family: Existing family to use as a template
        generator_path: Path to the generator script
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Adding new model family '{family}' based on template '{template_family}'...")
        
        # Read the current generator content
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Find the MODEL_FAMILIES dictionary in the file
        import re
        model_families_pattern = r'MODEL_FAMILIES\s*=\s*\{(.*?)\}(?=\n\n)'
        model_families_match = re.search(model_families_pattern, content, re.DOTALL)
        
        if not model_families_match:
            logger.error("Could not find MODEL_FAMILIES dictionary in generator file")
            return False
        
        # Get the template family from the dictionary
        template_pattern = rf'"{template_family}"\s*:\s*\{{(.*?)\}},'
        template_match = re.search(template_pattern, model_families_match.group(1), re.DOTALL)
        
        if not template_match:
            logger.error(f"Could not find template family '{template_family}' in MODEL_FAMILIES")
            return False
        
        # Create new family entry based on template
        template_config = template_match.group(1)
        
        # Determine which architecture type to use for the new family
        architecture_type = None
        for category, families in EXTENDED_MODEL_FAMILIES.items():
            if family in families:
                architecture_type = category
                break
        
        # Adjust the template configuration for the new family
        # This is a simplified version that just copies the template with minimal changes
        new_config = template_config
        
        # Update model IDs based on family
        if family == "roberta":
            new_config = new_config.replace('"bert-base-uncased"', '"roberta-base"')
            new_config = new_config.replace('BertModel', 'RobertaModel')
            new_config = new_config.replace('BertTokenizer', 'RobertaTokenizer')
            new_config = new_config.replace('BertForMaskedLM', 'RobertaForMaskedLM')
        elif family == "distilbert":
            new_config = new_config.replace('"bert-base-uncased"', '"distilbert-base-uncased"')
            new_config = new_config.replace('BertModel', 'DistilBertModel')
            new_config = new_config.replace('BertTokenizer', 'DistilBertTokenizer')
            new_config = new_config.replace('BertForMaskedLM', 'DistilBertForMaskedLM')
        elif family == "gpt_neo":
            new_config = new_config.replace('"gpt2"', '"EleutherAI/gpt-neo-125M"')
            new_config = new_config.replace('GPT2', 'GPTNeo')
        elif family == "t5":
            # T5 is a template, so no changes needed
            pass
        elif family == "vit":
            # ViT is a template, so no changes needed
            pass
        else:
            # For other models, we'll keep most configurations the same but update the model ID
            # This is just a simplified approach for now
            new_config = new_config.replace(f'"{template_family}', f'"{family}')
        
        # Create the new family entry
        new_family_entry = f'    "{family}": {{\n{new_config}    }},\n'
        
        # Add the new family to the MODEL_FAMILIES dictionary
        updated_model_families = model_families_match.group(0).replace(
            '}', f'{new_family_entry}}}'
        )
        
        # Update the file content
        updated_content = content.replace(
            model_families_match.group(0), 
            updated_model_families
        )
        
        # Write the updated content back to the file
        with open(generator_path, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Successfully added '{family}' to MODEL_FAMILIES based on '{template_family}'")
        return True
        
    except Exception as e:
        logger.error(f"Error adding model family: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Regenerate test files for HuggingFace model families")
    
    # Define available model families
    all_families = get_available_model_families()
    
    # Basic options
    parser.add_argument("--families", nargs="+", choices=all_families, default=BASE_MODEL_FAMILIES,
                        help=f"Model families to regenerate (default: {', '.join(BASE_MODEL_FAMILIES)})")
    parser.add_argument("--all", action="store_true",
                        help="Regenerate all known model families")
    parser.add_argument("--output-dir", type=str, default="fixed_tests",
                        help="Output directory for test files (default: fixed_tests)")
    parser.add_argument("--generator", type=str, default="test_generator_fixed.py",
                        help="Path to the generator script (default: test_generator_fixed.py)")
    
    # Advanced options
    parser.add_argument("--list", action="store_true",
                        help="List all known model families and exit")
    parser.add_argument("--add-family", type=str,
                        help="Add a new model family to the generator")
    parser.add_argument("--template", type=str, choices=BASE_MODEL_FAMILIES,
                        help="Template family to use when adding a new family")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get absolute paths
    generator_path = str(Path(args.generator).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    
    # List mode
    if args.list:
        print("\nAvailable model families:")
        print("\nBase families (fully tested):")
        for family in BASE_MODEL_FAMILIES:
            print(f"  - {family}")
        
        print("\nExtended families by architecture type:")
        for arch_type, families in EXTENDED_MODEL_FAMILIES.items():
            print(f"\n  {arch_type.upper()}:")
            for family in families:
                if family in BASE_MODEL_FAMILIES:
                    print(f"    - {family} (base)")
                else:
                    print(f"    - {family}")
        return 0
    
    # Add new family mode
    if args.add_family:
        if not args.template:
            parser.error("--template is required when using --add-family")
        
        if add_model_to_generator(args.add_family, args.template, generator_path):
            logger.info(f"Successfully added new model family: {args.add_family}")
            # Now generate a test file for the new family
            regenerate_test_file(args.add_family, generator_path, output_dir)
        else:
            logger.error(f"Failed to add new model family: {args.add_family}")
        return 0
    
    # Determine which families to regenerate
    families_to_regenerate = []
    if args.all:
        families_to_regenerate = all_families
        logger.info(f"Regenerating test files for ALL {len(families_to_regenerate)} model families")
    else:
        families_to_regenerate = args.families
        logger.info(f"Regenerating test files for: {', '.join(families_to_regenerate)}")
    
    # Regenerate each family
    successful = 0
    failed = 0
    
    for family in families_to_regenerate:
        if regenerate_test_file(family, generator_path, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    logger.info("\nRegeneration Summary:")
    logger.info(f"- Successful: {successful}")
    logger.info(f"- Failed: {failed}")
    logger.info(f"- Total: {successful + failed}")
    logger.info(f"- Output directory: {output_dir}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())