#!/usr/bin/env python3

"""
Comprehensive test runner for HuggingFace models.

This script:
1. Runs tests for multiple model architectures
2. Captures detailed metrics and results
3. Integrates with DuckDB for result storage and analysis
4. Supports various hardware backends (CPU, CUDA, OpenVINO)

Usage examples:
    # Test all models
    python run_comprehensive_hf_model_test.py --all

    # Test specific model families
    python run_comprehensive_hf_model_test.py --encoder-only --vision

    # Test specific models
    python run_comprehensive_hf_model_test.py --models bert,roberta,vit

    # Specify hardware options
    python run_comprehensive_hf_model_test.py --vision --cpu-only
    python run_comprehensive_hf_model_test.py --vision-text --all-hardware
"""

import os
import sys
import json
import glob
import time
import argparse
import logging
import subprocess
import datetime
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"comprehensive_hf_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = SCRIPT_DIR / "skills"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"
RESULTS_DIR = FIXED_TESTS_DIR / "collected_results"

# Define architecture categories
MODEL_ARCHITECTURES = {
    "encoder-only": [
        "bert", "distilbert", "roberta", "electra", "camembert", 
        "xlm_roberta", "albert", "deberta", "ernie"
    ],
    "decoder-only": [
        "gpt2", "gpt_j", "gpt_neo", "gpt_neox", "bloom", "llama", 
        "mistral", "falcon", "phi", "mixtral", "gemma"
    ],
    "encoder-decoder": [
        "t5", "bart", "pegasus", "mbart", "longt5", "led", "mt5", 
        "flan_t5", "prophetnet"
    ],
    "vision": [
        "vit", "swin", "deit", "beit", "convnext", "dinov2", "mask2former",
        "segformer", "yolos", "sam"
    ],
    "vision-text": [
        "clip", "blip", "blip_2", "git", "flava", "paligemma"
    ],
    "speech": [
        "wav2vec2", "hubert", "whisper", "encodec", "clap", 
        "speecht5", "unispeech", "sew"
    ],
    "multimodal": [
        "llava", "video_llava", "idefics", "imagebind"
    ]
}

# Mapping from model family to test file
def get_test_file_for_model(model_family: str) -> str:
    """Get the test file path for a model family."""
    model_family = model_family.replace("-", "_")  # Handle hyphenated model names
    test_file = FIXED_TESTS_DIR / f"test_hf_{model_family}.py"
    if test_file.exists():
        return str(test_file)
    return None

def run_test(test_file: str, args: Any) -> Dict:
    """Run a specific test with appropriate arguments."""
    cmd = [sys.executable, test_file]
    
    # Add hardware flags
    if args.cpu_only:
        cmd.append("--cpu-only")
    if args.all_hardware:
        cmd.append("--all-hardware")
    
    # Add model specification if provided
    if args.model:
        cmd.extend(["--model", args.model])
    
    # Add save flag to collect results
    cmd.append("--save")
    
    # Add output directory
    cmd.extend(["--output-dir", str(RESULTS_DIR)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the test and capture output
        start_time = time.time()
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=1200  # 20 minute timeout
        )
        duration = time.time() - start_time
        
        # Process output
        stdout = process.stdout
        stderr = process.stderr
        success = process.returncode == 0
        
        # Save detailed output for troubleshooting
        test_name = os.path.basename(test_file).replace(".py", "")
        log_file = f"regenerate_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_file, "w") as f:
            f.write(f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
        
        result = {
            "test": test_name,
            "success": success,
            "duration": duration,
            "returncode": process.returncode,
            "log_file": log_file
        }
        
        if success:
            logger.info(f"✅ {test_name} completed successfully in {duration:.2f} seconds")
        else:
            logger.error(f"❌ {test_name} failed with code {process.returncode}")
            result["stderr"] = stderr
        
        return result
        
    except subprocess.TimeoutExpired:
        logger.error(f"🕒 Timeout running {test_file}")
        return {
            "test": os.path.basename(test_file),
            "success": False,
            "error": "timeout",
            "duration": 1200  # timeout value
        }
    except Exception as e:
        logger.error(f"💥 Error running {test_file}: {e}")
        return {
            "test": os.path.basename(test_file),
            "success": False,
            "error": str(e)
        }

def collect_results(results_dir: str) -> Dict:
    """Collect and aggregate test results from JSON files."""
    result_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    aggregated_results = {
        "models": {},
        "summary": {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "architectures": {}
        },
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result_data = json.load(f)
            
            # Extract model information
            if "metadata" in result_data:
                metadata = result_data["metadata"]
                model_id = metadata.get("model", os.path.basename(file_path))
                model_class = metadata.get("class", "unknown")
                model_type = metadata.get("model_type", "unknown")
                
                # Determine result status
                success = False
                for result_name, result_data in result_data.get("results", {}).items():
                    if result_data.get("pipeline_success", False):
                        success = True
                        break
                
                # Add to aggregated results
                aggregated_results["models"][model_id] = {
                    "class": model_class,
                    "type": model_type,
                    "success": success,
                    "file": os.path.basename(file_path)
                }
                
                # Update summary statistics
                aggregated_results["summary"]["total"] += 1
                if success:
                    aggregated_results["summary"]["successful"] += 1
                else:
                    aggregated_results["summary"]["failed"] += 1
                
                # Update architecture statistics
                arch_type = next((arch for arch, models in MODEL_ARCHITECTURES.items() 
                                 if any(m in model_id.lower() for m in models)), "other")
                
                if arch_type not in aggregated_results["summary"]["architectures"]:
                    aggregated_results["summary"]["architectures"][arch_type] = {
                        "total": 0, "successful": 0, "failed": 0
                    }
                
                aggregated_results["summary"]["architectures"][arch_type]["total"] += 1
                if success:
                    aggregated_results["summary"]["architectures"][arch_type]["successful"] += 1
                else:
                    aggregated_results["summary"]["architectures"][arch_type]["failed"] += 1
                
        except Exception as e:
            logger.error(f"Error processing result file {file_path}: {e}")
    
    return aggregated_results

def run_tests_for_architecture(arch_type: str, args: Any) -> List[Dict]:
    """Run tests for all models in an architecture type."""
    results = []
    
    for model_family in MODEL_ARCHITECTURES.get(arch_type, []):
        test_file = get_test_file_for_model(model_family)
        if test_file:
            logger.info(f"Testing {model_family} ({arch_type})")
            result = run_test(test_file, args)
            results.append(result)
        else:
            logger.warning(f"No test file found for {model_family}")
            results.append({
                "test": f"test_hf_{model_family}",
                "success": False,
                "error": "no_test_file"
            })
    
    return results

def run_vision_text_model_tests(args: Any) -> List[Dict]:
    """Run tests for all vision-text models - specifically CLIP and BLIP models."""
    results = []
    
    # Test CLIP models
    clip_test_file = get_test_file_for_model("clip")
    if clip_test_file:
        logger.info("Testing CLIP models")
        custom_args = argparse.Namespace(**vars(args))
        # Optionally override specific args
        result = run_test(clip_test_file, custom_args)
        results.append(result)
    else:
        logger.warning("No test file found for CLIP models")
    
    # Test BLIP models
    blip_test_file = get_test_file_for_model("blip")
    if blip_test_file:
        logger.info("Testing BLIP models")
        custom_args = argparse.Namespace(**vars(args))
        # Optionally override specific args
        result = run_test(blip_test_file, custom_args)
        results.append(result)
    else:
        logger.warning("No test file found for BLIP models")
    
    # Add any other vision-text model families here
    
    return results

def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run comprehensive HuggingFace model tests")
    
    # Architecture selection
    arch_group = parser.add_argument_group("Architecture Selection")
    arch_group.add_argument("--all", action="store_true", help="Test all model architectures")
    arch_group.add_argument("--encoder-only", action="store_true", help="Test encoder-only models (BERT, RoBERTa, etc.)")
    arch_group.add_argument("--decoder-only", action="store_true", help="Test decoder-only models (GPT-2, LLaMA, etc.)")
    arch_group.add_argument("--encoder-decoder", action="store_true", help="Test encoder-decoder models (T5, BART, etc.)")
    arch_group.add_argument("--vision", action="store_true", help="Test vision models (ViT, Swin, etc.)")
    arch_group.add_argument("--vision-text", action="store_true", help="Test vision-text models (CLIP, BLIP, etc.)")
    arch_group.add_argument("--speech", action="store_true", help="Test speech models (Wav2Vec2, Whisper, etc.)")
    arch_group.add_argument("--multimodal", action="store_true", help="Test multimodal models (LLaVA, etc.)")
    
    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument("--models", type=str, help="Comma-separated list of model families to test")
    model_group.add_argument("--model", type=str, help="Specific model to test for each family")
    
    # Hardware options
    hw_group = parser.add_argument_group("Hardware Options")
    hw_group.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    hw_group.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Directory for output files")
    output_group.add_argument("--summary-file", type=str, help="Path to write summary JSON file")
    output_group.add_argument("--store-results", action="store_true", help="Store results in DuckDB database")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which architectures to test
    selected_archs = []
    if args.all:
        selected_archs = list(MODEL_ARCHITECTURES.keys())
    else:
        if args.encoder_only:
            selected_archs.append("encoder-only")
        if args.decoder_only:
            selected_archs.append("decoder-only")
        if args.encoder_decoder:
            selected_archs.append("encoder-decoder")
        if args.vision:
            selected_archs.append("vision")
        if args.vision_text:
            selected_archs.append("vision-text")
        if args.speech:
            selected_archs.append("speech")
        if args.multimodal:
            selected_archs.append("multimodal")
    
    # If no architectures selected but specific models provided
    if not selected_archs and args.models:
        logger.info(f"Testing specific models: {args.models}")
        model_families = args.models.split(",")
        all_results = []
        
        for model_family in model_families:
            test_file = get_test_file_for_model(model_family.strip())
            if test_file:
                logger.info(f"Testing {model_family}")
                result = run_test(test_file, args)
                all_results.append(result)
            else:
                logger.warning(f"No test file found for {model_family}")
        
        logger.info(f"Completed testing {len(all_results)} models")
        
    # Otherwise run tests for selected architectures
    elif selected_archs:
        logger.info(f"Testing architectures: {', '.join(selected_archs)}")
        all_results = []
        
        for arch in selected_archs:
            logger.info(f"Starting tests for {arch} architecture")
            
            if arch == "vision-text":
                # Special handling for vision-text models
                results = run_vision_text_model_tests(args)
            else:
                # Standard handling for other architectures
                results = run_tests_for_architecture(arch, args)
                
            all_results.extend(results)
            logger.info(f"Completed {len(results)} tests for {arch} architecture")
        
        # Log summary of results
        successful = sum(1 for r in all_results if r.get("success", False))
        logger.info(f"Completed {len(all_results)} tests, {successful} successful")
        
    else:
        logger.error("No architectures or models selected. Use --all or specify architectures/models.")
        parser.print_help()
        return 1
    
    # Collect detailed results
    logger.info("Collecting detailed results from JSON files")
    results_summary = collect_results(args.output_dir)
    
    # Write summary file
    summary_path = args.summary_file or os.path.join(
        args.output_dir, 
        f"comprehensive_test_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Summary written to {summary_path}")
    
    # Store results in DuckDB if requested
    if args.store_results:
        try:
            # Import the DuckDB integration script
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            if any(arch == "vision-text" for arch in selected_archs):
                logger.info("Storing vision-text results in DuckDB...")
                from vision_text_duckdb_integration import connect_to_db, create_tables, import_results
                
                # Connect to database
                conn = connect_to_db()
                if conn:
                    # Create tables if needed
                    create_tables(conn)
                    
                    # Import results
                    result = import_results(conn, args.output_dir)
                    logger.info(f"Imported {result['imported']} vision-text files, {result['failed']} failed")
                    
                    # Close connection
                    conn.close()
                else:
                    logger.warning("Failed to connect to DuckDB for storing vision-text results")
            
            # Add other model types as needed here
            
        except Exception as e:
            logger.error(f"Error storing results in DuckDB: {e}")
    
    # Print final summary
    print("\nCOMPREHENSIVE TEST SUMMARY:")
    print(f"Tested {results_summary['summary']['total']} models")
    print(f"✅ Successful: {results_summary['summary']['successful']}")
    print(f"❌ Failed: {results_summary['summary']['failed']}")
    
    # Print architecture-specific results
    print("\nResults by architecture:")
    for arch, stats in results_summary['summary']['architectures'].items():
        success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  - {arch.upper()}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())