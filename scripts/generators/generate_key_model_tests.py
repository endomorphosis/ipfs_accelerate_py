#!/usr/bin/env python3
"""
Generate Key Model Tests

This script generates test files for all key models with all hardware platforms
and stores them in a unified location for Phase 16 completion.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define key models with details
KEY_MODELS = [
    {"name": "bert", "full_name": "bert-base-uncased", "modality": "text", "description": "BERT text encoder"},
    {"name": "t5", "full_name": "t5-small", "modality": "text", "description": "T5 text encoder/decoder"},
    {"name": "vit", "full_name": "vit-base-patch16-224", "modality": "vision", "description": "Vision Transformer"},
    {"name": "clip", "full_name": "clip-vit-base-patch32", "modality": "multimodal", "description": "CLIP text-image model"},
    {"name": "whisper", "full_name": "whisper-tiny", "modality": "audio", "description": "Whisper speech recognition"},
    {"name": "wav2vec2", "full_name": "wav2vec2-base", "modality": "audio", "description": "Wav2Vec2 audio model"},
    {"name": "clap", "full_name": "clap-htsat-fused", "modality": "audio", "description": "CLAP audio-text model"},
    {"name": "llama", "full_name": "llama-7b", "modality": "text", "description": "LLaMA text generation"},
    {"name": "llava", "full_name": "llava-1.5-7b", "modality": "multimodal", "description": "LLaVA vision-language model"},
    {"name": "xclip", "full_name": "xclip-base-patch32", "modality": "video", "description": "XCLIP video-text model"},
    {"name": "detr", "full_name": "detr-resnet-50", "modality": "vision", "description": "DETR object detection"},
    {"name": "qwen2", "full_name": "qwen2-7b", "modality": "text", "description": "Qwen2 text generation"},
    {"name": "llava-next", "full_name": "llava-next-7b", "modality": "multimodal", "description": "LLaVA-Next vision-language model"},
]

# Hardware platforms with details
HARDWARE_PLATFORMS = [
    {"name": "cpu", "type": "REAL", "description": "CPU (available on all systems)"},
    {"name": "cuda", "type": "REAL", "description": "NVIDIA CUDA (GPU acceleration)"},
    {"name": "rocm", "type": "REAL", "description": "AMD ROCm (GPU acceleration)"},
    {"name": "mps", "type": "REAL", "description": "Apple Silicon MPS (GPU acceleration)"},
    {"name": "openvino", "type": "REAL", "description": "Intel OpenVINO acceleration"},
    {"name": "qualcomm", "type": "REAL", "description": "Qualcomm AI Engine acceleration"},
    {"name": "webnn", "type": "REAL", "description": "Browser WebNN API"},
    {"name": "webgpu", "type": "REAL", "description": "Browser WebGPU API"}
]

# Output directories
OUTPUT_DIR = "phase16_key_models"
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tests")
SKILL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "skills")
BENCHMARK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "benchmarks")

# Ensure all directories exist
for directory in [OUTPUT_DIR, TEST_OUTPUT_DIR, SKILL_OUTPUT_DIR, BENCHMARK_OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

def run_command(cmd):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"

def generate_test(model, output_dir, cross_platform=True):
    """Generate a test file for a model."""
    model_name = model["full_name"]
    platform_arg = "all" if cross_platform else "cpu,cuda"
    
    cmd = f"python fixed_merged_test_generator.py -g {model_name} -p {platform_arg}"
    if output_dir:
        cmd += f" -o {output_dir}/"
    
    success, output = run_command(cmd)
    if success:
        logger.info(f"Generated test for {model_name}: {output}")
        return True
    else:
        logger.error(f"Failed to generate test for {model_name}: {output}")
        return False

def generate_skill(model, output_dir):
    """Generate a skill file for a model."""
    model_name = model["full_name"]
    
    cmd = f"python integrated_skillset_generator.py -m {model_name} -p all"
    if output_dir:
        cmd += f" -o {output_dir}/"
    
    success, output = run_command(cmd)
    if success:
        logger.info(f"Generated skill for {model_name}: {output}")
        return True
    else:
        logger.error(f"Failed to generate skill for {model_name}: {output}")
        return False

def verify_file(file_path):
    """Verify that a file exists and is syntactically correct."""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile the file to check for syntax errors
        compile(content, file_path, 'exec')
        logger.info(f"Verified: {file_path}")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error verifying {file_path}: {e}")
        return False

def generate_coverage_report(results):
    """Generate a coverage report for the key models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"coverage_report_{timestamp}.md")
    json_path = os.path.join(OUTPUT_DIR, f"coverage_report_{timestamp}.json")
    
    # Calculate statistics
    total_models = len(KEY_MODELS)
    test_coverage = sum(1 for model in results if results[model]["test_success"]) / total_models * 100
    skill_coverage = sum(1 for model in results if results[model]["skill_success"]) / total_models * 100
    overall_coverage = sum(1 for model in results if results[model]["test_success"] and results[model]["skill_success"]) / total_models * 100
    
    # Create JSON report
    json_report = {
        "timestamp": timestamp,
        "statistics": {
            "total_models": total_models,
            "test_coverage_percent": test_coverage,
            "skill_coverage_percent": skill_coverage,
            "overall_coverage_percent": overall_coverage
        },
        "models": {}
    }
    
    for model in KEY_MODELS:
        model_name = model["name"]
        if model_name in results:
            json_report["models"][model_name] = {
                "full_name": model["full_name"],
                "modality": model["modality"],
                "description": model["description"],
                "test_generated": results[model_name]["test_success"],
                "skill_generated": results[model_name]["skill_success"],
                "test_file": results[model_name]["test_file"] if "test_file" in results[model_name] else None,
                "skill_file": results[model_name]["skill_file"] if "skill_file" in results[model_name] else None
            }
    
    # Write JSON report
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    # Create markdown report
    with open(report_path, 'w') as f:
        f.write("# Phase 16 Key Model Coverage Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Models**: {total_models}\n")
        f.write(f"- **Test Coverage**: {test_coverage:.1f}%\n")
        f.write(f"- **Skill Coverage**: {skill_coverage:.1f}%\n")
        f.write(f"- **Overall Coverage**: {overall_coverage:.1f}%\n\n")
        
        f.write("## Key Model Status\n\n")
        f.write("| Model | Modality | Test | Skill | Description |\n")
        f.write("|-------|----------|------|-------|-------------|\n")
        
        for model in KEY_MODELS:
            model_name = model["name"]
            modality = model["modality"]
            description = model["description"]
            
            if model_name in results:
                test_status = "✅" if results[model_name]["test_success"] else "❌"
                skill_status = "✅" if results[model_name]["skill_success"] else "❌"
            else:
                test_status = "❓"
                skill_status = "❓"
            
            f.write(f"| {model_name} | {modality} | {test_status} | {skill_status} | {description} |\n")
        
        f.write("\n## Hardware Platforms\n\n")
        f.write("| Platform | Type | Description |\n")
        f.write("|----------|------|-------------|\n")
        
        for platform in HARDWARE_PLATFORMS:
            f.write(f"| {platform['name']} | {platform['type']} | {platform['description']} |\n")
        
        f.write("\n## Next Steps\n\n")
        if overall_coverage < 100:
            f.write("To achieve 100% coverage:\n\n")
            for model in KEY_MODELS:
                model_name = model["name"]
                if model_name in results:
                    if not results[model_name]["test_success"]:
                        f.write(f"1. Generate test for {model_name}\n")
                    if not results[model_name]["skill_success"]:
                        f.write(f"2. Generate skill for {model_name}\n")
        else:
            f.write("All key models have complete coverage! Next steps:\n\n")
            f.write("1. Run benchmarks for all models\n")
            f.write("2. Verify cross-platform compatibility\n")
            f.write("3. Integrate with database for result storage\n")
    
    logger.info(f"Generated coverage report: {report_path}")
    logger.info(f"Generated JSON data: {json_path}")
    
    return report_path, json_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate tests for key models")
    parser.add_argument("--tests-only", action="store_true", help="Generate only tests, not skills")
    parser.add_argument("--skills-only", action="store_true", help="Generate only skills, not tests")
    parser.add_argument("--model", type=str, action="append", help="Generate for a specific model (can be used multiple times)")
    parser.add_argument("--cpu-only", action="store_true", help="Generate for CPU only, not all platforms")
    parser.add_argument("--verify", action="store_true", help="Verify generated files")
    parser.add_argument("--jobs", type=int, default=3, help="Number of parallel jobs (default: 3)")
    args = parser.parse_args()
    
    # Determine which models to process
    models_to_process = []
    if args.model:
        # Find the models in our list
        for model_name in args.model:
            model_found = False
            for model in KEY_MODELS:
                if model["name"] == model_name or model["full_name"] == model_name:
                    models_to_process.append(model)
                    model_found = True
                    break
            if not model_found:
                logger.error(f"Model {model_name} not found in key models list")
                return 1
    else:
        models_to_process = KEY_MODELS
    
    # Determine what to generate
    generate_tests = not args.skills_only
    generate_skills = not args.tests_only
    
    # Track results for reporting
    results = {}
    
    # Generate tests and skills
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        # Generate tests if requested
        if generate_tests:
            logger.info(f"Generating tests for {len(models_to_process)} models...")
            test_futures = []
            for model in models_to_process:
                future = executor.submit(generate_test, model, TEST_OUTPUT_DIR, not args.cpu_only)
                test_futures.append((model, future))
            
            for model, future in test_futures:
                model_name = model["name"]
                test_success = future.result()
                if model_name not in results:
                    results[model_name] = {}
                
                results[model_name]["test_success"] = test_success
                if test_success:
                    test_file = os.path.join(TEST_OUTPUT_DIR, f"test_hf_{model['full_name'].replace('-', '_')}.py")
                    results[model_name]["test_file"] = test_file
        
        # Generate skills if requested
        if generate_skills:
            logger.info(f"Generating skills for {len(models_to_process)} models...")
            skill_futures = []
            for model in models_to_process:
                future = executor.submit(generate_skill, model, SKILL_OUTPUT_DIR)
                skill_futures.append((model, future))
            
            for model, future in skill_futures:
                model_name = model["name"]
                skill_success = future.result()
                if model_name not in results:
                    results[model_name] = {}
                
                results[model_name]["skill_success"] = skill_success
                if skill_success:
                    skill_file = os.path.join(SKILL_OUTPUT_DIR, f"skill_hf_{model['full_name'].replace('-', '_')}.py")
                    results[model_name]["skill_file"] = skill_file
    
    # Verify files if requested
    if args.verify:
        logger.info("Verifying generated files...")
        
        # Verify tests
        if generate_tests:
            for model_name, info in results.items():
                if info.get("test_success") and "test_file" in info:
                    info["test_verified"] = verify_file(info["test_file"])
        
        # Verify skills
        if generate_skills:
            for model_name, info in results.items():
                if info.get("skill_success") and "skill_file" in info:
                    info["skill_verified"] = verify_file(info["skill_file"])
    
    # Generate coverage report
    report_path, json_path = generate_coverage_report(results)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 50)
    
    # Calculate statistics
    total_models = len(models_to_process)
    successful_tests = sum(1 for model_name in results if results[model_name].get("test_success", False))
    successful_skills = sum(1 for model_name in results if results[model_name].get("skill_success", False))
    
    if generate_tests:
        logger.info(f"Tests generated: {successful_tests}/{total_models} ({successful_tests/total_models*100:.1f}%)")
    
    if generate_skills:
        logger.info(f"Skills generated: {successful_skills}/{total_models} ({successful_skills/total_models*100:.1f}%)")
    
    if args.verify:
        verified_tests = sum(1 for model_name in results if results[model_name].get("test_verified", False))
        verified_skills = sum(1 for model_name in results if results[model_name].get("skill_verified", False))
        
        if generate_tests:
            logger.info(f"Tests verified: {verified_tests}/{successful_tests} ({verified_tests/successful_tests*100 if successful_tests else 0:.1f}%)")
        
        if generate_skills:
            logger.info(f"Skills verified: {verified_skills}/{successful_skills} ({verified_skills/successful_skills*100 if successful_skills else 0:.1f}%)")
    
    logger.info("=" * 50)
    logger.info(f"Coverage report: {report_path}")
    logger.info(f"JSON data: {json_path}")
    
    # Return success if all requested operations succeeded
    if generate_tests and successful_tests < total_models:
        return 1
    if generate_skills and successful_skills < total_models:
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())