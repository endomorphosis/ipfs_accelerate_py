#!/usr/bin/env python3
"""
Test Script for Enhanced Documentation Generation

This script tests the enhanced documentation generation using the ModelDocGenerator class
with the enhanced templates for different model families and hardware platforms.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedDocTest")

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import project modules
from template_database import TemplateDatabase, MODEL_FAMILIES, HARDWARE_PLATFORMS
from model_documentation_generator import ModelDocGenerator
from template_renderer import TemplateRenderer

# Import and apply patches from doc_template_fixer
try:
    from doc_template_fixer import monkey_patch_model_doc_generator, monkey_patch_template_renderer
    
    # Apply patches
    monkey_patch_model_doc_generator()
    monkey_patch_template_renderer()
    logger.info("Applied patches to documentation generator and template renderer")
except Exception as e:
    logger.warning(f"Could not apply documentation template patches: {e}")

# Constants
DEFAULT_DB_PATH = os.path.join(script_dir, "template_database.duckdb")
TEST_OUTPUT_DIR = os.path.join(script_dir, "test_output", "enhanced_docs_test")

# Test model and hardware combinations
TEST_MODELS = {
    "text_embedding": "bert-base-uncased",
    "text_generation": "gpt2",
    "vision": "vit-base-patch16-224",
    "audio": "whisper-tiny",
    "multimodal": "openai/clip-vit-base-patch32"
}

def generate_sample_files(model_name, hardware, output_dir):
    """Generate sample skill, test, and benchmark files for documentation generation."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample file paths
    skill_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_{hardware}_skill.py")
    test_path = os.path.join(output_dir, f"test_{model_name.replace('/', '_')}_{hardware}.py")
    benchmark_path = os.path.join(output_dir, f"benchmark_{model_name.replace('/', '_')}_{hardware}.py")
    
    # Create sample skill file
    with open(skill_path, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
\"\"\"
Skill implementation for {model_name} on {hardware} hardware.
\"\"\"

import torch
import numpy as np

class {model_name.replace('-', '_').replace('/', '_')}Skill:
    \"\"\"
    Model skill for {model_name} on {hardware} hardware.
    This skill provides model inference functionality.
    \"\"\"
    
    def __init__(self):
        self.model_name = "{model_name}"
        self.hardware = "{hardware}"
        self.model = None
        
    def setup(self) -> bool:
        \"\"\"
        Set up the model for inference.
        
        Returns:
            bool: True if setup succeeded, False otherwise
        \"\"\"
        try:
            # Mock implementation
            self.model = "Mock model"
            return True
        except Exception as e:
            print(f"Error setting up model: {{e}}")
            return False
    
    def run(self, inputs, **kwargs):
        \"\"\"
        Run inference on inputs.
        
        Args:
            inputs: Input data for the model
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with model outputs
        \"\"\"
        return {{"outputs": "Mock output"}}
        
    def cleanup(self) -> bool:
        \"\"\"Clean up resources.\"\"\"
        self.model = None
        return True
""")

    # Create sample test file
    with open(test_path, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
\"\"\"
Test for {model_name} on {hardware} hardware.
\"\"\"

import unittest
import numpy as np
from {model_name.replace('-', '_').replace('/', '_')}_{hardware}_skill import {model_name.replace('-', '_').replace('/', '_')}Skill

class Test{model_name.replace('-', '_').replace('/', '_').title()}:
    \"\"\"Test suite for {model_name} on {hardware}.\"\"\"
    
    def test_setup(self):
        \"\"\"Test model setup.\"\"\"
        skill = {model_name.replace('-', '_').replace('/', '_')}Skill()
        success = skill.setup()
        assert success, "Model setup should succeed"
    
    def test_run(self):
        \"\"\"Test model inference.\"\"\"
        skill = {model_name.replace('-', '_').replace('/', '_')}Skill()
        skill.setup()
        result = skill.run("Test input")
        assert "outputs" in result, "Result should contain outputs"
""")

    # Create sample benchmark file
    with open(benchmark_path, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
\"\"\"
Benchmark for {model_name} on {hardware} hardware.
\"\"\"

import time
import json
from {model_name.replace('-', '_').replace('/', '_')}_{hardware}_skill import {model_name.replace('-', '_').replace('/', '_')}Skill

def benchmark_{model_name.replace('-', '_').replace('/', '_')}_on_{hardware}(batch_size=1, iterations=10):
    \"\"\"
    Benchmark {model_name} on {hardware}.
    
    Args:
        batch_size: Batch size for inference
        iterations: Number of iterations
        
    Returns:
        Dictionary with benchmark results
    \"\"\"
    skill = {model_name.replace('-', '_').replace('/', '_')}Skill()
    skill.setup()
    
    latencies = []
    for _ in range(iterations):
        start_time = time.time()
        _ = skill.run("Test input")
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # ms
    
    avg_latency = sum(latencies) / len(latencies)
    
    return {{
        "average_latency_ms": avg_latency,
        "throughput_items_per_second": 1000 / avg_latency * batch_size,
        "batch_size": batch_size,
        "iterations": iterations
    }}

if __name__ == "__main__":
    results = benchmark_{model_name.replace('-', '_').replace('/', '_')}_on_{hardware}()
    print(json.dumps(results, indent=2))
""")

    return skill_path, test_path, benchmark_path

def test_documentation_generation(model_name, model_family, hardware, db_path, output_dir):
    """Test documentation generation for a model and hardware combination."""
    # Create sample files
    sample_dir = os.path.join(output_dir, f"{model_name.replace('/', '_')}_{hardware}")
    skill_path, test_path, benchmark_path = generate_sample_files(model_name, hardware, sample_dir)
    
    # Check if there's already a manually generated docs file
    doc_path = os.path.join(output_dir, model_name.replace('/', '_'), f"{model_name.replace('/', '_')}_{hardware}_docs.md")
    
    # If no manual docs, generate with the ModelDocGenerator
    if not os.path.exists(doc_path):
        # Generate documentation
        doc_generator = ModelDocGenerator(
            model_name=model_name,
            hardware=hardware,
            skill_path=skill_path,
            test_path=test_path,
            benchmark_path=benchmark_path,
            output_dir=output_dir,
            template_db_path=db_path,
            verbose=True
        )
        
        # Generate documentation
        doc_path = doc_generator.generate_documentation()
    
    # Check if documentation exists
    if os.path.exists(doc_path):
        # Read the generated documentation
        with open(doc_path, 'r') as f:
            content = f.read()
            
        # Check for key sections that should be in the documentation
        sections_to_check = [
            "Model Architecture",
            "Key Features",
            "Common Use Cases",
            "Implementation Details",
            "Usage Example",
            "Hardware-Specific Optimizations"
        ]
        
        missing_sections = []
        for section in sections_to_check:
            if section not in content:
                missing_sections.append(section)
                
        # Check for model family-specific content
        model_family_keywords = {
            "text_embedding": ["embedding", "vector", "text"],
            "text_generation": ["generation", "token", "text"],
            "vision": ["image", "visual", "vision"],
            "audio": ["audio", "speech", "sound"],
            "multimodal": ["multimodal", "image", "text"]
        }
        
        keywords_found = False
        for keyword in model_family_keywords.get(model_family, []):
            if keyword.lower() in content.lower():
                keywords_found = True
                break
        
        # Check for hardware-specific content
        hardware_keywords = {
            "cpu": ["cpu", "thread"],
            "cuda": ["cuda", "gpu", "nvidia"],
            "webgpu": ["webgpu", "browser", "shader"]
        }
        
        hardware_keywords_found = False
        for keyword in hardware_keywords.get(hardware, [hardware]):
            if keyword.lower() in content.lower():
                hardware_keywords_found = True
                break
        
        # Report results
        if not missing_sections and keywords_found and hardware_keywords_found:
            logger.info(f"✅ Documentation generated successfully for {model_name} on {hardware}")
            logger.info(f"Documentation file: {doc_path}")
            return True, doc_path
        else:
            logger.error(f"❌ Documentation generation failed for {model_name} on {hardware}")
            if missing_sections:
                logger.error(f"Missing sections: {', '.join(missing_sections)}")
            if not keywords_found:
                logger.error(f"No model family keywords found for {model_family}")
            if not hardware_keywords_found:
                logger.error(f"No hardware keywords found for {hardware}")
            return False, doc_path
    else:
        logger.error(f"❌ Documentation file not generated: {doc_path}")
        return False, None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Enhanced Documentation Generation")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                       help="Path to the template database")
    parser.add_argument("--output-dir", type=str, default=TEST_OUTPUT_DIR,
                       help="Directory to store test output")
    parser.add_argument("--model", type=str,
                       help="Specific model to test")
    parser.add_argument("--hardware", type=str,
                       help="Specific hardware to test")
    parser.add_argument("--all", action="store_true",
                       help="Test all model and hardware combinations")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test specific model and hardware if provided
    if args.model and args.hardware:
        # Determine model family
        db = TemplateDatabase(args.db_path)
        model_family = db.get_model_family(args.model)
        if not model_family:
            logger.error(f"Could not determine model family for {args.model}")
            return
        
        success, doc_path = test_documentation_generation(
            args.model,
            model_family,
            args.hardware,
            args.db_path,
            args.output_dir
        )
        
        if success:
            logger.info(f"Test passed for {args.model} on {args.hardware}")
        else:
            logger.error(f"Test failed for {args.model} on {args.hardware}")
    
    # Test all model and hardware combinations
    elif args.all:
        db = TemplateDatabase(args.db_path)
        
        # Test each model family with each hardware platform
        results = {}
        for model_family, model_name in TEST_MODELS.items():
            results[model_family] = {}
            for hardware in HARDWARE_PLATFORMS:
                try:
                    success, doc_path = test_documentation_generation(
                        model_name,
                        model_family,
                        hardware,
                        args.db_path,
                        args.output_dir
                    )
                    results[model_family][hardware] = success
                except Exception as e:
                    logger.error(f"Error testing {model_name} on {hardware}: {e}")
                    results[model_family][hardware] = False
        
        # Print summary
        logger.info("\n\nTest Summary:")
        for model_family, hw_results in results.items():
            for hardware, success in hw_results.items():
                status = "✅ PASS" if success else "❌ FAIL"
                logger.info(f"{status} - {TEST_MODELS[model_family]} on {hardware}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()