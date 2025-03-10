#!/usr/bin/env python3
# Utility script to run tests for IPFS Accelerate skills (models)

import os
import sys
import argparse
import subprocess
import json
import time
import concurrent.futures
from datetime import datetime
import glob

# Define model categories
MODEL_CATEGORIES = {
    "language": [
        "bert", "distilbert", "roberta", "gpt_neo", "gptj", "bart", "mt5", 
        "mbart", "electra", "longformer", "deberta_v2", "gpt2", "dpr", 
        "mobilebert", "mpnet", "camembert", "flaubert", "codegen", "xlm_roberta", 
        "albert", "opt", "bloom", "squeezebert", "layoutlm", "deberta", 
        "llama", "pegasus", "led", "qwen2", "qwen2_moe", "phi", "phi3", "mistral", 
        "mixtral", "gemma", "gemma2", "falcon", "mamba", "mamba2", "stablelm"
    ],
    "vision": [
        "vit", "deit", "detr", "swin", "convnext", "clip", "videomae", "xclip",
        "beit", "convnextv2", "dino", "efficientnet", "mobilevit", "sam", "upernet", 
        "vit_mae", "mask2former"
    ],
    "audio": [
        "wav2vec2", "whisper", "hubert", "clap", "musicgen", "encodec", "speecht5",
        "wavlm", "audio_spectrogram_transformer", "data2vec_audio"
    ],
    "multimodal": [
        "llava", "llava_next", "qwen2_vl", "blip", "blip_2", "instructblip",
        "kosmos_2", "paligemma", "clip", "vilt", "video_llava", "llava_onevision"
    ],
    "conversational": [
        "blenderbot", "blenderbot_small"
    ],
    "default": [
        "default_embed", "default_lm"
    ]
}

# Helper functions
def get_test_file_path(model_name):
    """Get the path to the test file for a given model name."""
    if model_name.startswith("default_"):
        return f"skills/test_{model_name}.py"
    else:
        return f"skills/test_hf_{model_name}.py"

def run_test(test_file, platform=None, implementation=None):
    """Run a specific test file with optional platform and implementation flags."""
    cmd = [sys.executable, test_file]
    
    if platform:
        cmd.extend(["--platform", platform])
    
    if implementation == "real":
        cmd.append("--real")
    elif implementation == "mock":
        cmd.append("--mock")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        return {
            "file": test_file,
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        return {
            "file": test_file,
            "success": False,
            "output": "",
            "error": str(e),
            "elapsed_time": 0
        }

def find_all_test_files():
    """Find all test files in the skills directory."""
    return sorted(glob.glob("skills/test_*.py"))

def get_model_name_from_file(file_path):
    """Extract model name from test file path."""
    base_name = os.path.basename(file_path)
    if base_name.startswith("test_hf_"):
        return base_name[8:-3]  # Remove test_hf_ and .py
    elif base_name.startswith("test_"):
        return base_name[5:-3]  # Remove test_ and .py
    return None

def print_summary(results):
    """Print a summary of test results."""
    total = len(results)
    succeeded = sum(1 for r in results if r["success"])
    failed = total - succeeded
    
    print("\n" + "="*80)
    print(f"SUMMARY: {succeeded}/{total} tests passed ({failed} failed)")
    print("="*80)
    
    if failed > 0:
        print("\nFAILED TESTS:")
        for result in results:
            if not result["success"]:
                model_name = get_model_name_from_file(result["file"])
                print(f"  - {model_name} ({result['file']})")
                print(f"    Error: {result['error'][:100]}..." if len(result['error']) > 100 else f"    Error: {result['error']}")
    
    # Print timing stats
    print("\nPERFORMANCE:")
    sorted_results = sorted(results, key=lambda x: x["elapsed_time"], reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        model_name = get_model_name_from_file(result["file"])
        print(f"  {i+1}. {model_name}: {result['elapsed_time']:.2f} seconds")
    
    # Print platform support stats
    platforms = {"cpu": 0, "cuda": 0, "openvino": 0}
    for result in results:
        if result["success"]:
            for platform in platforms:
                if platform in result["output"].lower():
                    platforms[platform] += 1
    
    print("\nPLATFORM SUPPORT:")
    for platform, count in platforms.items():
        print(f"  {platform.upper()}: {count}/{succeeded} tests")
        
    # Print missing test files based on huggingface_model_types.json
    try:
        with open('huggingface_model_types.json', 'r') as f:
            model_types = json.load(f)
            
        all_test_files = find_all_test_files()
        tested_models = set()
        
        for file in all_test_files:
            model_name = get_model_name_from_file(file)
            if model_name:
                # Normalize names (convert underscores to hyphens)
                tested_models.add(model_name.replace('_', '-'))
        
        # Find models missing tests
        missing_models = []
        for model_type in model_types:
            normalized = model_type.lower()
            if normalized not in tested_models and not any(
                normalized.startswith(prefix) for prefix in ['openai-', 'optimized-']
            ):
                missing_models.append(model_type)
        
        if missing_models:
            print("\nMISSING TEST COVERAGE:")
            print(f"  {len(missing_models)}/{len(model_types)} models in huggingface_model_types.json are missing tests")
            print("  Top models needing tests:")
            for model in sorted(missing_models)[:10]:
                print(f"  - {model}")
            
    except Exception as e:
        print(f"\nError checking missing tests: {e}")

def generate_model_coverage_report():
    """Generate a report of test coverage for models in huggingface_model_types.json"""
    try:
        with open('huggingface_model_types.json', 'r') as f:
            model_types = json.load(f)
            
        # Find all test files
        all_test_files = find_all_test_files()
        tested_models = {}
        
        for file in all_test_files:
            model_name = get_model_name_from_file(file)
            if model_name:
                # Store the test file for each model
                tested_models[model_name.replace('_', '-')] = file
        
        # Group models into categories
        categories = {
            "language_models": [],
            "vision_models": [],
            "audio_models": [],
            "multimodal_models": [],
            "other_models": []
        }
        
        missing_models = []
        
        for model_type in model_types:
            normalized = model_type.lower()
            
            # Skip certain model prefixes
            if any(normalized.startswith(prefix) for prefix in ['openai-', 'optimized-']):
                continue
                
            # Determine category
            if any(term in normalized for term in ["gpt", "bert", "llama", "t5", "roberta", "gemma"]):
                category = "language_models"
            elif any(term in normalized for term in ["vision", "vit", "clip", "swin", "detr", "convnext"]):
                category = "vision_models"
            elif any(term in normalized for term in ["audio", "wav2vec", "whisper", "clap", "speech"]):
                category = "audio_models"
            elif any(term in normalized for term in ["llava", "blip", "multimodal", "instructblip", "kosmos"]):
                category = "multimodal_models"
            else:
                category = "other_models"
            
            # Check if model has a test
            if normalized in tested_models:
                categories[category].append({
                    "model": model_type,
                    "test_file": tested_models[normalized],
                    "has_test": True
                })
            else:
                categories[category].append({
                    "model": model_type,
                    "has_test": False
                })
                missing_models.append(model_type)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"model_coverage_report_{timestamp}.md"
        
        with open(report_path, "w") as f:
            f.write(f"# Model Test Coverage Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary stats
            total_models = len(model_types) - sum(1 for m in model_types if any(
                m.lower().startswith(p) for p in ['openai-', 'optimized-']
            ))
            tested_count = len(tested_models)
            coverage_percent = (tested_count / total_models * 100)
            
            f.write("## Coverage Summary\n\n")
            f.write(f"- **Total model types**: {total_models}\n")
            f.write(f"- **Models with tests**: {tested_count}\n")
            f.write(f"- **Coverage**: {coverage_percent:.1f}%\n")
            f.write(f"- **Missing tests**: {len(missing_models)}\n\n")
            
            # Write category tables
            for category_name, models in categories.items():
                # Skip empty categories
                if not models:
                    continue
                    
                # Calculate category stats
                total_in_category = len(models)
                tested_in_category = sum(1 for m in models if m["has_test"])
                coverage = (tested_in_category / total_in_category * 100) if total_in_category > 0 else 0
                
                f.write(f"## {category_name.replace('_', ' ').title()}\n\n")
                f.write(f"Coverage: {tested_in_category}/{total_in_category} ({coverage:.1f}%)\n\n")
                f.write("| Model | Test Status | Test File |\n")
                f.write("|-------|------------|----------|\n")
                
                # Sort models by test status then by name
                sorted_models = sorted(models, key=lambda m: (not m["has_test"], m["model"].lower()))
                
                for model in sorted_models:
                    status = "✅ TESTED" if model["has_test"] else "❌ MISSING"
                    test_file = model.get("test_file", "-")
                    f.write(f"| {model['model']} | {status} | {test_file} |\n")
                
                f.write("\n")
            
            # Write list of top missing models
            f.write("## Top Priority Models Missing Tests\n\n")
            priority_models = [
                m for m in missing_models if any(
                    term in m.lower() for term in ["llama", "gpt", "bert", "phi", "gemma", "mistral", "mixtral", "clip", "whisper"]
                )
            ]
            for model in sorted(priority_models)[:20]:
                f.write(f"- {model}\n")
        
        print(f"Coverage report generated: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"Error generating model coverage report: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run tests for IPFS Accelerate skills")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all tests")
    group.add_argument("--models", type=str, help="Comma-separated list of models to test")
    group.add_argument("--type", type=str, choices=MODEL_CATEGORIES.keys(), 
                      help="Run tests for specific model type")
    group.add_argument("--coverage-report", action="store_true",
                      help="Generate a model coverage report without running tests")
    
    parser.add_argument("--platform", type=str, choices=["cpu", "cuda", "openvino"],
                       help="Test specific platform only")
    parser.add_argument("--implementation", type=str, choices=["real", "mock"],
                       help="Force specific implementation")
    parser.add_argument("--parallel", type=int, default=4,
                       help="Number of tests to run in parallel")
    parser.add_argument("--output", type=str, 
                       help="Output file to save results (JSON format)")
    
    args = parser.parse_args()
    
    # Handle coverage report generation
    if args.coverage_report:
        report_path = generate_model_coverage_report()
        return 0 if report_path else 1
    
    # Determine which models to test
    if args.all:
        test_files = find_all_test_files()
    elif args.type:
        model_list = MODEL_CATEGORIES[args.type]
        test_files = [get_test_file_path(model) for model in model_list]
    else:
        model_list = [m.strip() for m in args.models.split(",")]
        test_files = [get_test_file_path(model) for model in model_list]
    
    # Filter out non-existent files
    valid_test_files = [f for f in test_files if os.path.exists(f)]
    if len(valid_test_files) < len(test_files):
        missing = set(test_files) - set(valid_test_files)
        print(f"Warning: {len(missing)} test files not found: {', '.join(missing)}")
    
    print(f"Running {len(valid_test_files)} tests...")
    
    # Run tests in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
        future_to_file = {
            executor.submit(run_test, file, args.platform, args.implementation): file 
            for file in valid_test_files
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
            file = future_to_file[future]
            model_name = get_model_name_from_file(file)
            
            try:
                result = future.result()
                status = "✅ Passed" if result["success"] else "❌ Failed"
                print(f"[{i}/{len(valid_test_files)}] Testing {model_name}: {status} ({result['elapsed_time']:.2f}s)")
                results.append(result)
            except Exception as e:
                print(f"[{i}/{len(valid_test_files)}] Testing {model_name}: ❌ Error: {str(e)}")
                results.append({
                    "file": file,
                    "success": False,
                    "output": "",
                    "error": str(e),
                    "elapsed_time": 0
                })
    
    # Print summary
    print_summary(results)
    
    # Save results if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(valid_test_files),
            "passed_tests": sum(1 for r in results if r["success"]),
            "failed_tests": sum(1 for r in results if not r["success"]),
            "results": results
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
        
    # Return an error code if any tests failed
    return 0 if all(r["success"] for r in results) else 1

if __name__ == "__main__":
    sys.exit(main())