#!/usr/bin/env python3
"""
Documentation Integration Verification Tool

This script tests the integration of the enhanced documentation system with the
Integrated Component Test Runner by generating documentation for sample models
and verifying the output contains all required sections.
"""

import os
import sys
import logging
import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import project modules
from integrated_component_test_runner import IntegratedComponentTester
from doc_template_fixer import monkey_patch_model_doc_generator, monkey_patch_template_renderer
from integrate_documentation_system import integrate_enhanced_doc_generator, modify_integrated_component_tester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DocIntegrationVerifier")

# Constants
OUTPUT_DIR = os.path.join(script_dir, "test_output", "doc_integration_test")
TEMPLATE_DB_PATH = os.path.join(script_dir, "template_database.duckdb")

# Model families and hardware platforms to test
TEST_MODELS = {
    "text_embedding": "bert-base-uncased",
    "text_generation": "gpt2",
    "vision": "vit-base-patch16-224",
    "multimodal": "openai/clip-vit-base-patch32",
    "audio": "whisper-tiny"
}

TEST_HARDWARE = ["cpu", "cuda", "webgpu"]

# Required sections in generated documentation
REQUIRED_SECTIONS = [
    "# ", 
    "## Overview",
    "## Model Architecture",
    "## Implementation Details",
    "## Usage Example"
]

# Model family-specific keywords that should be present in the documentation
FAMILY_KEYWORDS = {
    "text_embedding": ["embedding", "token", "BERT"],
    "text_generation": ["generation", "language model", "GPT"],
    "vision": ["image", "vision", "patch"],
    "multimodal": ["multimodal", "text", "image"],
    "audio": ["audio", "speech", "waveform"]
}

# Hardware-specific keywords that should be present in the documentation
HARDWARE_KEYWORDS = {
    "cpu": ["CPU", "multi-threading", "portability"],
    "cuda": ["GPU", "CUDA", "NVIDIA"],
    "webgpu": ["WebGPU", "browser", "shader"]
}


def validate_documentation(doc_path: str, model_family: str, hardware: str) -> Dict[str, Any]:
    """
    Validate that the generated documentation contains all required sections
    and model/hardware-specific keywords.
    
    Args:
        doc_path: Path to the documentation file
        model_family: Model family (text_embedding, vision, etc.)
        hardware: Hardware platform (cpu, cuda, etc.)
        
    Returns:
        Dictionary with validation results
    """
    if not os.path.exists(doc_path):
        return {
            "valid": False, 
            "error": f"Documentation file does not exist: {doc_path}",
            "missing_sections": REQUIRED_SECTIONS,
            "missing_family_keywords": FAMILY_KEYWORDS.get(model_family, []),
            "missing_hardware_keywords": HARDWARE_KEYWORDS.get(hardware, [])
        }
    
    # Read documentation content
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Check required sections
    missing_sections = []
    for section in REQUIRED_SECTIONS:
        if section not in content:
            missing_sections.append(section)
    
    # Check family-specific keywords
    missing_family_keywords = []
    for keyword in FAMILY_KEYWORDS.get(model_family, []):
        if keyword.lower() not in content.lower():
            missing_family_keywords.append(keyword)
    
    # Check hardware-specific keywords
    missing_hardware_keywords = []
    for keyword in HARDWARE_KEYWORDS.get(hardware, []):
        if keyword.lower() not in content.lower():
            missing_hardware_keywords.append(keyword)
    
    # Check for unreplaced variables
    unreplaced_vars = re.findall(r'\${([a-zA-Z0-9_]+)}', content)
    
    # Calculate validation score (0-100)
    total_checks = (
        len(REQUIRED_SECTIONS) + 
        len(FAMILY_KEYWORDS.get(model_family, [])) + 
        len(HARDWARE_KEYWORDS.get(hardware, []))
    )
    
    passed_checks = (
        total_checks - 
        len(missing_sections) - 
        len(missing_family_keywords) - 
        len(missing_hardware_keywords)
    )
    
    score = int((passed_checks / total_checks) * 100) if total_checks > 0 else 0
    
    # Overall validation result
    is_valid = (
        len(missing_sections) == 0 and
        len(missing_family_keywords) == 0 and
        len(missing_hardware_keywords) == 0 and
        len(unreplaced_vars) == 0
    )
    
    return {
        "valid": is_valid,
        "score": score,
        "missing_sections": missing_sections,
        "missing_family_keywords": missing_family_keywords,
        "missing_hardware_keywords": missing_hardware_keywords,
        "unreplaced_vars": unreplaced_vars,
        "doc_length": len(content),
        "section_count": len(re.findall(r'^##? ', content, re.MULTILINE))
    }

def generate_and_validate_docs(model_name: str, hardware: str, model_family: str) -> Dict[str, Any]:
    """
    Generate documentation for a model and hardware combination, and validate the result.
    
    Args:
        model_name: Name of the model
        hardware: Hardware platform
        model_family: Model family
        
    Returns:
        Dictionary with generation and validation results
    """
    logger.info(f"Testing documentation generation for {model_name} on {hardware}...")
    
    # Set up test directory
    test_dir = os.path.join(OUTPUT_DIR, f"{model_name.replace('/', '_')}_{hardware}")
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create tester instance
        tester = IntegratedComponentTester(
            model_name=model_name,
            hardware=hardware,
            update_expected=False,
            generate_docs=True,
            template_db_path=TEMPLATE_DB_PATH,
            output_dir=OUTPUT_DIR,
            verbose=False,
            quick_test=True
        )
        
        # Generate components
        skill_file, test_file, benchmark_file = tester.generate_components(test_dir)
        
        # Create mock test results
        test_results = {
            "success": True,
            "test_count": 5,
            "execution_time": 2.5,
            "stdout": "All tests passed successfully"
        }
        
        # Create mock benchmark results
        benchmark_results = {
            "benchmark_results": {
                "results_by_batch": {
                    "1": {"average_latency_ms": 25.0, "average_throughput_items_per_second": 40.0},
                    "2": {"average_latency_ms": 45.0, "average_throughput_items_per_second": 44.4},
                    "4": {"average_latency_ms": 85.0, "average_throughput_items_per_second": 47.1},
                    "8": {"average_latency_ms": 165.0, "average_throughput_items_per_second": 48.5}
                }
            }
        }
        
        # Run test with documentation
        doc_result = tester.run_test_with_docs(test_dir, test_results)
        
        if not doc_result["success"]:
            return {
                "success": False,
                "error": doc_result.get("error", "Unknown error during documentation generation"),
                "model": model_name,
                "hardware": hardware,
                "model_family": model_family
            }
        
        # Validate generated documentation
        doc_path = doc_result["documentation_path"]
        validation = validate_documentation(doc_path, model_family, hardware)
        
        return {
            "success": True,
            "model": model_name,
            "hardware": hardware,
            "model_family": model_family,
            "doc_path": doc_path,
            "validation": validation
        }
    
    except Exception as e:
        logger.error(f"Error generating documentation for {model_name} on {hardware}: {e}")
        return {
            "success": False,
            "error": str(e),
            "model": model_name,
            "hardware": hardware,
            "model_family": model_family
        }

def run_integration_test(specific_model: Optional[str] = None, 
                      specific_hardware: Optional[str] = None) -> Dict[str, Any]:
    """
    Run integration tests for documentation generation across model families and hardware platforms.
    
    Args:
        specific_model: Optional specific model to test
        specific_hardware: Optional specific hardware to test
        
    Returns:
        Dictionary with test results
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Apply patches to fix documentation generation
    monkey_patch_model_doc_generator()
    monkey_patch_template_renderer()
    integrate_enhanced_doc_generator()
    modify_integrated_component_tester()
    
    results = []
    
    # Determine which models and hardware to test
    models_to_test = {specific_model: next((fam for fam, model in TEST_MODELS.items() if model == specific_model), "unknown")} if specific_model else TEST_MODELS
    hardware_to_test = [specific_hardware] if specific_hardware else TEST_HARDWARE
    
    # Run tests for each model and hardware combination
    for model_family, model_name in models_to_test.items():
        for hardware in hardware_to_test:
            result = generate_and_validate_docs(model_name, hardware, model_family)
            results.append(result)
    
    # Calculate overall success rate
    successful_tests = sum(1 for r in results if r["success"] and r.get("validation", {}).get("valid", False))
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) if total_tests > 0 else 0
    
    # Generate summary
    summary = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": success_rate,
        "results": results
    }
    
    # Save detailed results to file
    results_path = os.path.join(OUTPUT_DIR, "integration_test_results.json")
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def print_summary(summary: Dict[str, Any]):
    """Print a human-readable summary of test results."""
    print("\n" + "="*80)
    print(f"DOCUMENTATION INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Success Rate: {summary['success_rate']*100:.1f}%")
    print("-"*80)
    
    for result in summary["results"]:
        model = result["model"]
        hardware = result["hardware"]
        
        if result["success"]:
            validation = result.get("validation", {})
            is_valid = validation.get("valid", False)
            score = validation.get("score", 0)
            
            if is_valid:
                status = f"✅ PASSED ({score}%)"
            else:
                status = f"⚠️ PARTIAL ({score}%)"
                
                # Print validation issues
                missing_sections = validation.get("missing_sections", [])
                missing_family_keywords = validation.get("missing_family_keywords", [])
                missing_hardware_keywords = validation.get("missing_hardware_keywords", [])
                unreplaced_vars = validation.get("unreplaced_vars", [])
                
                if missing_sections:
                    print(f"  - Missing sections: {', '.join(missing_sections)}")
                if missing_family_keywords:
                    print(f"  - Missing family keywords: {', '.join(missing_family_keywords)}")
                if missing_hardware_keywords:
                    print(f"  - Missing hardware keywords: {', '.join(missing_hardware_keywords)}")
                if unreplaced_vars:
                    print(f"  - Unreplaced variables: {', '.join(unreplaced_vars)}")
        else:
            status = f"❌ FAILED: {result.get('error', 'Unknown error')}"
        
        print(f"{model:<25} + {hardware:<10}: {status}")
    
    print("-"*80)
    if summary['success_rate'] == 1.0:
        print("🎉 All documentation tests passed successfully!")
    else:
        print(f"⚠️ Some documentation tests failed. Check {os.path.join(OUTPUT_DIR, 'integration_test_results.json')} for details.")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Documentation Integration Verification Tool")
    parser.add_argument("--model", help="Specific model to test")
    parser.add_argument("--hardware", help="Specific hardware to test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run integration test
    print(f"Running documentation integration tests...")
    if args.model or args.hardware:
        print(f"Testing specific configuration: Model={args.model or 'all'}, Hardware={args.hardware or 'all'}")
    
    summary = run_integration_test(args.model, args.hardware)
    print_summary(summary)