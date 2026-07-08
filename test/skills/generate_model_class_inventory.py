#!/usr/bin/env python3

'''
Generate a comprehensive inventory of all HuggingFace Transformers classes 
that have a from_pretrained() method.

This script creates a JSON inventory of all model classes for testing coverage.
'''

import os
import sys
import json
import importlib
import inspect
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = None
    HAS_TRANSFORMERS = False
    logger.warning("transformers library not available, using introspection only mode")


def get_all_transformers_model_classes() -> List[str]:
    """Get all model classes from transformers library using introspection."""
    if not HAS_TRANSFORMERS:
        logger.warning("Cannot get model classes: transformers not available")
        return []
    
    model_classes = []
    
    # Look through transformers module for model classes
    for attr_name in dir(transformers):
        # Skip private attributes and common non-model attributes
        if attr_name.startswith('_') or attr_name in ['logging', 'utils', 'tokenization']:
            continue
        
        attr = getattr(transformers, attr_name)
        
        # Check if it's a class and has from_pretrained method
        if inspect.isclass(attr):
            if hasattr(attr, 'from_pretrained') and callable(getattr(attr, 'from_pretrained')):
                model_classes.append(attr_name)
    
    return model_classes


def categorize_model_classes(model_classes: List[str]) -> Dict[str, List[str]]:
    """Categorize model classes by architecture type."""
    categories = defaultdict(list)
    
    # Architecture detection patterns
    architecture_patterns = {
        "encoder-only": ["ForMaskedLM", "ForTokenClassification", "ForSequenceClassification", "ForQuestionAnswering", "Model"],
        "decoder-only": ["LMHeadModel", "ForCausalLM", "GPT", "Bloom", "Llama", "Mistral", "Falcon"],
        "encoder-decoder": ["EncoderDecoderModel", "T5", "Bart", "ForConditionalGeneration"],
        "vision": ["ImageClassification", "Vit", "Swin", "ForImageClassification", "ConvNext"],
        "vision-text": ["VisionTextDualEncoder", "Clip", "Blip", "VisionEncoderDecoder"],
        "speech": ["ForCTC", "ForAudioClassification", "Wav2Vec2", "Whisper", "Speech"],
        "multimodal": ["LlavaForConditionalGeneration", "Flava", "ImageBind"],
    }
    
    # Categorize each model class
    for model_class in model_classes:
        categorized = False
        
        for category, patterns in architecture_patterns.items():
            if any(pattern in model_class for pattern in patterns):
                categories[category].append(model_class)
                categorized = True
                break
        
        # If no category matched, put in "other"
        if not categorized:
            categories["other"].append(model_class)
    
    return dict(categories)


def check_from_pretrained_method(model_class_name: str) -> bool:
    """Check if the class has a from_pretrained method that works."""
    if not HAS_TRANSFORMERS:
        # Can't verify, assume it works if it was in the list
        return True
    
    try:
        # Get the class
        class_obj = getattr(transformers, model_class_name)
        
        # Check for class method
        if not hasattr(class_obj, 'from_pretrained'):
            return False
        
        # Check if it's callable
        from_pretrained = getattr(class_obj, 'from_pretrained')
        if not callable(from_pretrained):
            return False
        
        # Check if it's a class method (or static method)
        if isinstance(from_pretrained, classmethod) or isinstance(from_pretrained, staticmethod):
            return True
            
        # If we get here and it's callable, it's probably a method
        return True
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error checking {model_class_name}: {e}")
        return False


def find_test_file_for_model(model_class_name: str, test_dir: str = './fixed_tests') -> Optional[str]:
    """Find the test file that tests this model class."""
    if not os.path.exists(test_dir):
        logger.warning(f"Test directory {test_dir} not found")
        return None
    
    # Convert CamelCase to snake_case
    name_parts = []
    for i, char in enumerate(model_class_name):
        if i > 0 and char.isupper() and not model_class_name[i-1].isupper():
            name_parts.append('_')
        name_parts.append(char.lower())
    
    snake_case = ''.join(name_parts)
    
    # Look for files with this model name pattern
    test_files = []
    for file_name in os.listdir(test_dir):
        if not file_name.endswith('.py'):
            continue
        
        # Check for exact match first
        if snake_case in file_name.lower():
            return os.path.join(test_dir, file_name)
        
        # Then check for partial matches
        # Remove common prefixes/suffixes for more accurate matching
        clean_name = snake_case
        for prefix in ['for_', 'with_', 'using_']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        
        for suffix in ['_for_masked_lm', '_model', '_for_causal_lm', '_for_sequence_classification']:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
        
        if clean_name in file_name.lower():
            test_files.append(os.path.join(test_dir, file_name))
    
    # Return the first match if any
    return test_files[0] if test_files else None


def test_file_has_from_pretrained_test(test_file_path: str) -> bool:
    """Check if the test file has a test_from_pretrained method."""
    if not os.path.exists(test_file_path):
        return False
    
    try:
        with open(test_file_path, 'r') as f:
            content = f.read()
            # Check for method definition
            if 'def test_from_pretrained' in content:
                return True
            # Also check for method call
            if '.test_from_pretrained(' in content:
                return True
            return False
    except Exception as e:
        logger.warning(f"Error reading {test_file_path}: {e}")
        return False


def create_model_inventory(output_file: str = 'model_class_inventory.json', test_dir: str = './fixed_tests'):
    """Create a comprehensive inventory of models and their test coverage."""
    # Get all model classes
    logger.info("Discovering model classes...")
    model_classes = get_all_transformers_model_classes()
    logger.info(f"Found {len(model_classes)} model classes")
    
    # Categorize them
    logger.info("Categorizing model classes...")
    categorized = categorize_model_classes(model_classes)
    
    # Build the inventory
    inventory = {
        'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_classes': len(model_classes),
        'categories': {},
    }
    
    # Track coverage stats
    total_with_from_pretrained = 0
    total_with_test_file = 0
    total_with_from_pretrained_test = 0
    
    # Process each category
    for category, classes in categorized.items():
        inventory['categories'][category] = {
            'count': len(classes),
            'classes': []
        }
        
        for model_class in sorted(classes):
            # Check the from_pretrained method
            has_from_pretrained = check_from_pretrained_method(model_class)
            if has_from_pretrained:
                total_with_from_pretrained += 1
            
            # Find the test file
            test_file = find_test_file_for_model(model_class, test_dir)
            has_test_file = test_file is not None
            if has_test_file:
                total_with_test_file += 1
            
            # Check if the test file has a from_pretrained test
            has_from_pretrained_test = False
            if has_test_file:
                has_from_pretrained_test = test_file_has_from_pretrained_test(test_file)
                if has_from_pretrained_test:
                    total_with_from_pretrained_test += 1
            
            # Add to inventory
            inventory['categories'][category]['classes'].append({
                'name': model_class,
                'has_from_pretrained': has_from_pretrained,
                'test_file': test_file,
                'has_from_pretrained_test': has_from_pretrained_test
            })
    
    # Add coverage stats
    inventory['coverage'] = {
        'with_from_pretrained': total_with_from_pretrained,
        'with_from_pretrained_pct': round(total_with_from_pretrained / len(model_classes) * 100, 2) if model_classes else 0,
        'with_test_file': total_with_test_file,
        'with_test_file_pct': round(total_with_test_file / len(model_classes) * 100, 2) if model_classes else 0,
        'with_from_pretrained_test': total_with_from_pretrained_test,
        'with_from_pretrained_test_pct': round(total_with_from_pretrained_test / total_with_from_pretrained * 100, 2) if total_with_from_pretrained else 0,
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(inventory, f, indent=2)
    
    logger.info(f"Inventory written to {output_file}")
    logger.info(f"Coverage statistics:")
    logger.info(f"  - Classes with from_pretrained: {total_with_from_pretrained}/{len(model_classes)} ({inventory['coverage']['with_from_pretrained_pct']}%)")
    logger.info(f"  - Classes with test file: {total_with_test_file}/{len(model_classes)} ({inventory['coverage']['with_test_file_pct']}%)")
    logger.info(f"  - Classes with from_pretrained test: {total_with_from_pretrained_test}/{total_with_from_pretrained} ({inventory['coverage']['with_from_pretrained_test_pct']}%)")
    
    return inventory


def create_coverage_report(inventory: Dict[str, Any], output_file: str = 'from_pretrained_coverage.md'):
    """Create a markdown coverage report from the inventory."""
    with open(output_file, 'w') as f:
        f.write(f"# HuggingFace from_pretrained() Method Test Coverage\n\n")
        f.write(f"Generated on: {inventory['generated_date']}\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Total model classes:** {inventory['total_classes']}\n")
        f.write(f"- **Classes with from_pretrained:** {inventory['coverage']['with_from_pretrained']} ({inventory['coverage']['with_from_pretrained_pct']}%)\n")
        f.write(f"- **Classes with test file:** {inventory['coverage']['with_test_file']} ({inventory['coverage']['with_test_file_pct']}%)\n")
        f.write(f"- **Classes with from_pretrained test:** {inventory['coverage']['with_from_pretrained_test']} ({inventory['coverage']['with_from_pretrained_test_pct']}%)\n\n")
        
        f.write(f"## Coverage by Architecture Type\n\n")
        
        # For each category
        for category, data in inventory['categories'].items():
            f.write(f"### {category} ({data['count']} models)\n\n")
            
            # Count coverage
            classes_with_from_pretrained = sum(1 for c in data['classes'] if c['has_from_pretrained'])
            classes_with_test = sum(1 for c in data['classes'] if c['test_file'])
            classes_with_from_pretrained_test = sum(1 for c in data['classes'] if c['has_from_pretrained_test'])
            
            # Write category stats
            f.write(f"- **Classes with from_pretrained:** {classes_with_from_pretrained}/{data['count']} "
                   f"({round(classes_with_from_pretrained / data['count'] * 100, 2) if data['count'] else 0}%)\n")
            f.write(f"- **Classes with test file:** {classes_with_test}/{data['count']} "
                   f"({round(classes_with_test / data['count'] * 100, 2) if data['count'] else 0}%)\n")
            f.write(f"- **Classes with from_pretrained test:** {classes_with_from_pretrained_test}/{classes_with_from_pretrained} "
                   f"({round(classes_with_from_pretrained_test / classes_with_from_pretrained * 100, 2) if classes_with_from_pretrained else 0}%)\n\n")
            
            # Create a table of all classes in this category
            f.write(f"| Model Class | Has from_pretrained | Has Test File | Tests from_pretrained |\n")
            f.write(f"|-------------|-------------------|---------------|----------------------|\n")
            
            for cls in sorted(data['classes'], key=lambda x: x['name']):
                f.write(f"| {cls['name']} | ")
                f.write(f"{'✅' if cls['has_from_pretrained'] else '❌'} | ")
                f.write(f"{'✅' if cls['test_file'] else '❌'} | ")
                f.write(f"{'✅' if cls['has_from_pretrained_test'] else '❌'} |\n")
            
            f.write(f"\n")
        
        f.write(f"## Missing from_pretrained Tests\n\n")
        f.write(f"The following model classes have a from_pretrained method but no test for it:\n\n")
        
        missing_tests = []
        for category, data in inventory['categories'].items():
            for cls in data['classes']:
                if cls['has_from_pretrained'] and not cls['has_from_pretrained_test']:
                    missing_tests.append((category, cls['name']))
        
        if missing_tests:
            f.write(f"| Category | Model Class |\n")
            f.write(f"|----------|-------------|\n")
            
            for category, cls_name in sorted(missing_tests, key=lambda x: (x[0], x[1])):
                f.write(f"| {category} | {cls_name} |\n")
        else:
            f.write(f"✅ **All model classes with from_pretrained have tests!** ✅\n")
        
        f.write(f"\n\n---\n\n")
        f.write(f"This report was generated using `generate_model_class_inventory.py`.\n")
        f.write(f"To update this report, run:\n```bash\npython generate_model_class_inventory.py --update-report\n```\n")
    
    logger.info(f"Coverage report written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate an inventory of all HuggingFace model classes with from_pretrained')
    parser.add_argument('-o', '--output', default='model_class_inventory.json',
                        help='Output JSON file path (default: model_class_inventory.json)')
    parser.add_argument('-t', '--test-dir', default='./fixed_tests',
                        help='Directory containing test files (default: ./fixed_tests)')
    parser.add_argument('-r', '--report', default='from_pretrained_coverage.md',
                        help='Output markdown report file path (default: from_pretrained_coverage.md)')
    parser.add_argument('--update-report', action='store_true',
                        help='Update the report without being prompted')
    
    args = parser.parse_args()
    
    if not HAS_TRANSFORMERS:
        logger.error("Error: transformers library is required for this script")
        if args.update_report:
            # Continue in report-only mode if automated update
            logger.warning("Continuing in report-only mode for automated update")
            if os.path.exists(args.output):
                try:
                    with open(args.output, 'r') as f:
                        inventory = json.load(f)
                    create_coverage_report(inventory, args.report)
                    return 0
                except Exception as e:
                    logger.error(f"Error loading inventory: {e}")
        return 1
    
    # Create the inventory
    inventory = create_model_inventory(args.output, args.test_dir)
    
    # Create the report
    create_coverage_report(inventory, args.report)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
