#!/usr/bin/env python3
"""
Simple fix script for the test generator to address critical issues.
"""

import os
import sys
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_test_generator(generator_path="/home/barberb/ipfs_accelerate_py/test/skills/test_generator_fixed.py"):
    """Fix critical issues in the test generator."""
    if not os.path.exists(generator_path):
        logger.error(f"Generator file not found: {generator_path}")
        return False

    logger.info(f"Fixing test generator: {generator_path}")

    with open(generator_path, 'r') as f:
        content = f.read()

    # 1. Fix the indentation issue in the from_pretrained method
    content = re.sub(
        r"if device == \"cuda\":\s+try:[^\n]*\n\s+with torch\.no_grad\(\):[^\n]*\n\s+_ = model\(\*\*inputs\)[^\n]*\n\s+except Exception:[^\n]*\n\s+pass\n\s+(# Run multiple inference passes)",
        r"if device == \"cuda\":\n        try:\n            with torch.no_grad():\n                _ = model(**inputs)\n        except Exception:\n            pass\n\n    \1",
        content
    )

    # 2. Fix unterminated triple quotes
    triple_quote_count = content.count('"""')
    if triple_quote_count % 2 != 0:
        logger.info(f"Fixing unterminated triple quotes (found {triple_quote_count} instances)")
        lines = content.split('\n')
        
        # If we have an odd number, add a line with triple quotes at the end
        lines.append('"""')
        content = '\n'.join(lines)

    # 3. Fix registry duplication for hyphenated models
    content = re.sub(r"GPT_GPT_GPT_GPT_J_MODELS_REGISTRY", "GPT_J_MODELS_REGISTRY", content)
    content = re.sub(r"hf_gpt_j_j_j_j_j_j_j_", "hf_gpt_j_", content)

    # 4. Fix general function indentation issues
    content = re.sub(
        r"def fix_from_pretrained_indentation\(content\):[^}]*pattern2 = r\"([^\"]*)",
        r"def fix_from_pretrained_indentation(content):\n    \"\"\"Fix indentation issues specifically in the test_from_pretrained method.\"\"\"\n    pattern2 = r\"\1",
        content
    )

    # Save the fixed generator
    with open(generator_path, 'w') as f:
        f.write(content)

    logger.info(f"Applied critical fixes to generator: {generator_path}")
    return True

def fix_template_indentation(template_path):
    """Fix indentation issues in a template file."""
    if not os.path.exists(template_path):
        logger.error(f"Template file not found: {template_path}")
        return False
    
    logger.info(f"Fixing indentation in template: {template_path}")
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Fix indentation in test_from_pretrained method's CUDA block
    content = re.sub(
        r"if device == \"cuda\":\s+try:[^\n]*\n\s+with torch\.no_grad\(\):[^\n]*\n\s+_ = model\(\*\*inputs\)[^\n]*\n\s+except Exception:[^\n]*\n\s+pass\n\s+(# Run multiple inference passes)",
        r"if device == \"cuda\":\n        try:\n            with torch.no_grad():\n                _ = model(**inputs)\n        except Exception:\n            pass\n\n    \1",
        content
    )
    
    # Fix unterminated triple quotes
    triple_quote_count = content.count('"""')
    if triple_quote_count % 2 != 0:
        logger.info(f"Fixing unterminated triple quotes in {template_path}")
        lines = content.split('\n')
        lines.append('"""')
        content = '\n'.join(lines)
    
    # Write fixed template back
    with open(template_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Fixed template saved: {template_path}")
    return True

def fix_templates_directory(templates_dir="/home/barberb/ipfs_accelerate_py/test/skills/templates"):
    """Fix critical issues in all templates."""
    if not os.path.exists(templates_dir):
        logger.error(f"Templates directory not found: {templates_dir}")
        return False
    
    templates_fixed = 0
    for template_file in os.listdir(templates_dir):
        if template_file.endswith(".py") and not template_file.endswith(".bak"):
            template_path = os.path.join(templates_dir, template_file)
            if fix_template_indentation(template_path):
                templates_fixed += 1
    
    logger.info(f"Fixed {templates_fixed} templates in {templates_dir}")
    return templates_fixed > 0

def main():
    """Main function to fix the test generator and templates."""
    # 1. Fix the test generator
    generator_fixed = fix_test_generator()
    
    # 2. Fix all templates
    templates_fixed = fix_templates_directory()
    
    # Report results
    if generator_fixed:
        logger.info("Successfully fixed the test generator.")
    else:
        logger.error("Failed to fix the test generator.")
    
    if templates_fixed:
        logger.info("Successfully fixed the templates.")
    else:
        logger.error("Failed to fix the templates.")
    
    return generator_fixed and templates_fixed

if __name__ == "__main__":
    main()