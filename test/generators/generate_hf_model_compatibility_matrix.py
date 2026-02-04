#!/usr/bin/env python3
"""
Generate HuggingFace Model Compatibility Matrix

This script addresses Priority #2: "Comprehensive HuggingFace Model Testing (300+ classes)"
by generating a comprehensive compatibility matrix for HuggingFace models across
different hardware platforms. It analyzes existing test files, runs tests on
available hardware, and stores results in DuckDB.

Features:
1. Analyzes existing test files to determine model coverage
2. Categorizes models by architecture type and modality
3. Runs tests on available hardware platforms
4. Generates a detailed compatibility matrix
5. Stores results in DuckDB for visualization
6. Identifies gaps in model coverage
7. Generates coverage reports and visualizations

Usage:
  python generate_hf_model_compatibility_matrix.py --analyze
  python generate_hf_model_compatibility_matrix.py --run-tests [--hardware cpu cuda]
  python generate_hf_model_compatibility_matrix.py --generate-matrix
  python generate_hf_model_compatibility_matrix.py --generate-report
  python generate_hf_model_compatibility_matrix.py --identify-gaps
"""

import os
import sys
import json
import logging
import argparse
import importlib
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"compatibility_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Path constants
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = ROOT_DIR / "skills"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"
REPORTS_DIR = ROOT_DIR / "reports"
DB_PATH = ROOT_DIR / "benchmark_db.duckdb"

# Create reports directory if it doesn't exist
REPORTS_DIR.mkdir(exist_ok=True)

# Hardware platforms for testing
HARDWARE_PLATFORMS = {
    "cpu": {
        "name": "CPU",
        "flag": "--device cpu",
        "priority": "high"
    },
    "cuda": {
        "name": "CUDA (NVIDIA GPU)",
        "flag": "--device cuda",
        "priority": "high"
    },
    "rocm": {
        "name": "ROCm (AMD GPU)",
        "flag": "--device rocm", 
        "priority": "high"
    },
    "mps": {
        "name": "Metal Performance Shaders (Apple Silicon)",
        "flag": "--device mps",
        "priority": "high"
    },
    "openvino": {
        "name": "OpenVINO (Intel)",
        "flag": "--device openvino",
        "priority": "high"
    },
    "qualcomm": {
        "name": "Qualcomm AI Engine",
        "flag": "--device qualcomm",
        "priority": "medium"
    },
    "webnn": {
        "name": "WebNN",
        "flag": "--web-platform webnn",
        "priority": "medium"
    },
    "webgpu": {
        "name": "WebGPU",
        "flag": "--web-platform webgpu",
        "priority": "medium"
    }
}

# Model categories and architectures
MODEL_CATEGORIES = {
    "text-encoders": {
        "description": "Text encoder models (BERT, RoBERTa, etc.)",
        "architecture_type": "encoder_only",
        "model_type": "text",
        "priority": "critical",
        "families": ["bert", "roberta", "distilbert", "albert", "electra", "xlm-roberta", "camembert", "deberta"]
    },
    "text-decoders": {
        "description": "Text decoder models (GPT-2, LLaMA, etc.)",
        "architecture_type": "decoder_only",
        "model_type": "text",
        "priority": "critical",
        "families": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "llama", "opt", "bloom", "falcon", "mistral", "gemma"]
    },
    "text-encoder-decoders": {
        "description": "Text encoder-decoder models (T5, BART, etc.)",
        "architecture_type": "encoder_decoder",
        "model_type": "text",
        "priority": "critical",
        "families": ["t5", "bart", "mbart", "pegasus", "mt5", "prophetnet", "led"]
    },
    "vision": {
        "description": "Vision models (ViT, DETR, etc.)",
        "architecture_type": "vision",
        "model_type": "vision",
        "priority": "high",
        "families": ["vit", "swin", "deit", "beit", "convnext", "detr", "sam", "yolos", "segformer"]
    },
    "speech": {
        "description": "Speech models (Whisper, Wav2Vec2, etc.)",
        "architecture_type": "speech",
        "model_type": "audio",
        "priority": "high",
        "families": ["whisper", "wav2vec2", "hubert", "unispeech", "sew", "speech-to-text", "speech-to-text-2"]
    },
    "multimodal": {
        "description": "Multimodal models (CLIP, LLaVA, etc.)",
        "architecture_type": "multimodal",
        "model_type": "multimodal",
        "priority": "high",
        "families": ["clip", "blip", "llava", "git", "paligemma", "idefics", "flava", "imagebind"]
    }
}

def get_installed_hardware():
    """
    Detect which hardware platforms are installed and available.
    
    Returns:
        Dict mapping hardware identifiers to availability status
    """
    hardware_status = {}
    
    # Check CPU (always available)
    hardware_status["cpu"] = True
    
    # Check CUDA (NVIDIA GPU)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        hardware_status["cuda"] = result.returncode == 0
    except FileNotFoundError:
        hardware_status["cuda"] = False
    
    # Check ROCm (AMD GPU)
    try:
        result = subprocess.run(["rocm-smi"], capture_output=True, text=True)
        hardware_status["rocm"] = result.returncode == 0
    except FileNotFoundError:
        hardware_status["rocm"] = False
    
    # Check OpenVINO
    try:
        import openvino
        hardware_status["openvino"] = True
    except ImportError:
        hardware_status["openvino"] = False
    
    # Check MPS (Apple Silicon)
    try:
        import torch
        hardware_status["mps"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        hardware_status["mps"] = False
    
    # WebNN and WebGPU would need a browser environment, so they are not detected here
    hardware_status["webnn"] = False  # Cannot reliably detect outside browser
    hardware_status["webgpu"] = False  # Cannot reliably detect outside browser
    
    # Qualcomm needs special detection
    try:
        # This is a simplified check that would need enhancement for real detection
        qualcomm_libs = ["/usr/lib/libQNN.so", "/usr/local/lib/libQNN.so"]
        hardware_status["qualcomm"] = any(os.path.exists(lib) for lib in qualcomm_libs)
    except:
        hardware_status["qualcomm"] = False
    
    return hardware_status

def extract_model_family_from_filename(filename):
    """Extract model family from a test file name."""
    if filename.startswith("test_hf_"):
        # Remove test_hf_ prefix and .py suffix
        family = filename[8:]
        if family.endswith(".py"):
            family = family[:-3]
        # Convert back to hyphenated form if needed
        family = family.replace("_", "-")
        return family
    return None

def analyze_test_files():
    """
    Analyze existing test files to determine model coverage.
    
    Returns:
        Dict containing analysis results
    """
    if not FIXED_TESTS_DIR.exists():
        logger.error(f"Fixed tests directory not found: {FIXED_TESTS_DIR}")
        return {}
    
    # Get all test files
    test_files = [f for f in FIXED_TESTS_DIR.glob("test_hf_*.py") if f.is_file() and not f.name.endswith(".bak")]
    logger.info(f"Found {len(test_files)} test files in {FIXED_TESTS_DIR}")
    
    # Analyze test files
    model_info = {}
    architecture_counts = defaultdict(int)
    category_counts = defaultdict(int)
    priority_counts = defaultdict(int)
    
    # Get set of backup files to exclude
    backup_files = {f.stem for f in FIXED_TESTS_DIR.glob("*.bak")}
    
    for test_file in test_files:
        # Skip backup files
        if test_file.stem in backup_files:
            continue
        
        # Extract model family from filename
        family = extract_model_family_from_filename(test_file.name)
        if not family:
            continue
        
        # Determine architecture type and category
        architecture_type = None
        category = None
        priority = "medium"  # Default priority
        
        for cat_name, cat_info in MODEL_CATEGORIES.items():
            if family in cat_info["families"]:
                architecture_type = cat_info["architecture_type"]
                category = cat_name
                priority = cat_info["priority"]
                break
        
        # If architecture not found in categories, try to determine from name
        if not architecture_type:
            if "bert" in family or "electra" in family or "albert" in family or "roberta" in family:
                architecture_type = "encoder_only"
                category = "text-encoders"
            elif "gpt" in family or "llama" in family or "bloom" in family:
                architecture_type = "decoder_only"
                category = "text-decoders"
            elif "t5" in family or "bart" in family:
                architecture_type = "encoder_decoder"
                category = "text-encoder-decoders"
            elif "vit" in family or "swin" in family or "detr" in family:
                architecture_type = "vision"
                category = "vision"
            elif "wav2vec" in family or "whisper" in family or "hubert" in family:
                architecture_type = "speech"
                category = "speech"
            elif "clip" in family or "blip" in family or "llava" in family:
                architecture_type = "multimodal"
                category = "multimodal"
            else:
                architecture_type = "unknown"
                category = "other"
        
        # Store model info
        model_info[family] = {
            "family": family,
            "test_file": str(test_file),
            "architecture_type": architecture_type,
            "category": category,
            "priority": priority
        }
        
        # Update counts
        architecture_counts[architecture_type] += 1
        category_counts[category] += 1
        priority_counts[priority] += 1
    
    # Prepare results
    results = {
        "total_models": len(model_info),
        "models": model_info,
        "architecture_counts": dict(architecture_counts),
        "category_counts": dict(category_counts),
        "priority_counts": dict(priority_counts)
    }
    
    return results

def run_tests(hardware_platforms=None, max_models=None, priority=None):
    """
    Run tests for models on specified hardware platforms.
    
    Args:
        hardware_platforms: List of hardware platforms to test on
        max_models: Maximum number of models to test
        priority: Priority level of models to test (critical, high, medium, all)
        
    Returns:
        Dict containing test results
    """
    # Get available hardware
    available_hardware = get_installed_hardware()
    
    # If no hardware specified, use all available
    if not hardware_platforms:
        hardware_platforms = [hw for hw, available in available_hardware.items() if available]
    else:
        # Filter unavailable hardware
        hardware_platforms = [hw for hw in hardware_platforms if available_hardware.get(hw, False)]
    
    if not hardware_platforms:
        logger.error("No available hardware platforms to test on")
        return {}
    
    logger.info(f"Running tests on hardware platforms: {', '.join(hardware_platforms)}")
    
    # Analyze existing test files
    analysis = analyze_test_files()
    model_info = analysis.get("models", {})
    
    # Filter by priority if specified
    if priority and priority != "all":
        model_info = {
            family: info for family, info in model_info.items()
            if info.get("priority") == priority or (priority == "high" and info.get("priority") == "critical")
        }
    
    # Limit to max_models if specified
    if max_models and max_models > 0:
        model_info = dict(list(model_info.items())[:max_models])
    
    logger.info(f"Running tests for {len(model_info)} models")
    
    # Dictionary to store results
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": 0,
        "successful_tests": 0,
        "failed_tests": 0,
        "test_results": []
    }
    
    # Run tests for each model on each hardware platform
    total_tests = len(model_info) * len(hardware_platforms)
    test_count = 0
    
    for family, info in model_info.items():
        test_file = info["test_file"]
        
        for hw in hardware_platforms:
            test_count += 1
            logger.info(f"[{test_count}/{total_tests}] Testing {family} on {hw}")
            
            # Get hardware flag
            hw_flag = HARDWARE_PLATFORMS[hw]["flag"]
            
            # Construct command
            cmd = [
                sys.executable,
                test_file,
                "--model", family
            ]
            
            # Add hardware flag
            if hw_flag:
                cmd.extend(hw_flag.split())
            
            # Add save flag to save results
            cmd.extend(["--save"])
            
            # Run command
            start_time = datetime.now()
            process = subprocess.run(cmd, capture_output=True, text=True)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Process results
            success = process.returncode == 0
            if success:
                results["successful_tests"] += 1
            else:
                results["failed_tests"] += 1
            
            results["total_tests"] += 1
            
            # Add test result
            test_result = {
                "model": family,
                "hardware": hw,
                "success": success,
                "returncode": process.returncode,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add stdout/stderr if there's an error
            if not success:
                test_result["stdout"] = process.stdout
                test_result["stderr"] = process.stderr
            
            results["test_results"].append(test_result)
            
            # Log result
            if success:
                logger.info(f"✅ {family} on {hw}: Success ({duration:.2f}s)")
            else:
                logger.error(f"❌ {family} on {hw}: Failed ({duration:.2f}s)")
    
    # Save results to file
    results_file = REPORTS_DIR / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {results_file}")
    
    return results

def generate_compatibility_matrix(test_results=None):
    """
    Generate a compatibility matrix from test results.
    
    Args:
        test_results: Test results dict (if None, load latest results)
        
    Returns:
        Path to the generated matrix file
    """
    # If no test results provided, load the latest
    if not test_results:
        # Find latest results file
        results_files = list(REPORTS_DIR.glob("test_results_*.json"))
        if not results_files:
            logger.error("No test results found")
            return None
        
        # Sort by modification time (newest first)
        results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = results_files[0]
        
        with open(latest_file, "r") as f:
            test_results = json.load(f)
    
    # Create compatibility matrix
    matrix = {}
    
    # Get model analysis
    analysis = analyze_test_files()
    model_info = analysis.get("models", {})
    
    # Initialize matrix with model information
    for family, info in model_info.items():
        matrix[family] = {
            "family": family,
            "architecture_type": info.get("architecture_type", "unknown"),
            "category": info.get("category", "other"),
            "priority": info.get("priority", "medium"),
            "hardware_compatibility": {}
        }
        
        # Initialize hardware compatibility
        for hw in HARDWARE_PLATFORMS:
            matrix[family]["hardware_compatibility"][hw] = {
                "tested": False,
                "success": False,
                "duration": None,
                "timestamp": None
            }
    
    # Update matrix with test results
    for result in test_results.get("test_results", []):
        model = result.get("model")
        hardware = result.get("hardware")
        
        if model in matrix and hardware in matrix[model]["hardware_compatibility"]:
            matrix[model]["hardware_compatibility"][hardware] = {
                "tested": True,
                "success": result.get("success", False),
                "duration": result.get("duration"),
                "timestamp": result.get("timestamp")
            }
    
    # Generate matrix files
    
    # JSON format
    json_file = REPORTS_DIR / f"compatibility_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, "w") as f:
        json.dump(matrix, f, indent=2)
    
    # Markdown format
    md_file = REPORTS_DIR / f"compatibility_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(md_file, "w") as f:
        f.write("# HuggingFace Model Compatibility Matrix\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group by category
        categories = defaultdict(list)
        for family, info in matrix.items():
            category = info.get("category", "other")
            categories[category].append(family)
        
        # Write matrix for each category
        for category, families in sorted(categories.items()):
            f.write(f"## {category}\n\n")
            
            # Header row
            f.write("| Model | Architecture | Priority |")
            for hw in HARDWARE_PLATFORMS:
                f.write(f" {hw} |")
            f.write("\n")
            
            # Separator row
            f.write("|-------|--------------|----------|")
            for _ in HARDWARE_PLATFORMS:
                f.write("---|")
            f.write("\n")
            
            # Data rows
            for family in sorted(families):
                info = matrix[family]
                arch_type = info.get("architecture_type", "unknown")
                priority = info.get("priority", "medium")
                
                f.write(f"| {family} | {arch_type} | {priority} |")
                
                for hw in HARDWARE_PLATFORMS:
                    hw_compat = info.get("hardware_compatibility", {}).get(hw, {})
                    tested = hw_compat.get("tested", False)
                    success = hw_compat.get("success", False)
                    
                    if not tested:
                        f.write(" - |")
                    elif success:
                        f.write(" ✅ |")
                    else:
                        f.write(" ❌ |")
                
                f.write("\n")
            
            f.write("\n")
    
    # Create CSV format
    csv_file = REPORTS_DIR / f"compatibility_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(csv_file, "w") as f:
        # Header row
        f.write("Model,Architecture,Category,Priority")
        for hw in HARDWARE_PLATFORMS:
            f.write(f",{hw}")
        f.write("\n")
        
        # Data rows
        for family, info in sorted(matrix.items()):
            arch_type = info.get("architecture_type", "unknown")
            category = info.get("category", "other")
            priority = info.get("priority", "medium")
            
            f.write(f"{family},{arch_type},{category},{priority}")
            
            for hw in HARDWARE_PLATFORMS:
                hw_compat = info.get("hardware_compatibility", {}).get(hw, {})
                tested = hw_compat.get("tested", False)
                success = hw_compat.get("success", False)
                
                if not tested:
                    f.write(",untested")
                elif success:
                    f.write(",success")
                else:
                    f.write(",failure")
            
            f.write("\n")
    
    # Update DuckDB if available
    try:
        import duckdb
        update_duckdb_matrix(matrix)
    except ImportError:
        logger.warning("DuckDB not available, skipping database update")
    
    logger.info(f"Compatibility matrix saved to:")
    logger.info(f"  - JSON: {json_file}")
    logger.info(f"  - Markdown: {md_file}")
    logger.info(f"  - CSV: {csv_file}")
    
    return {
        "json": str(json_file),
        "markdown": str(md_file),
        "csv": str(csv_file)
    }

def update_duckdb_matrix(matrix):
    """
    Update the DuckDB database with compatibility matrix data.
    
    Args:
        matrix: Compatibility matrix dict
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import duckdb
        
        # Connect to database
        conn = duckdb.connect(str(DB_PATH))
        
        # Check if model_compatibility table exists
        result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_compatibility'").fetchall()
        
        if not result:
            # Create table
            conn.execute("""
            CREATE TABLE model_compatibility (
                model_family VARCHAR,
                architecture_type VARCHAR,
                category VARCHAR,
                priority VARCHAR,
                cpu BOOLEAN DEFAULT FALSE,
                cuda BOOLEAN DEFAULT FALSE,
                rocm BOOLEAN DEFAULT FALSE,
                mps BOOLEAN DEFAULT FALSE,
                openvino BOOLEAN DEFAULT FALSE,
                qualcomm BOOLEAN DEFAULT FALSE,
                webnn BOOLEAN DEFAULT FALSE,
                webgpu BOOLEAN DEFAULT FALSE,
                test_date TIMESTAMP,
                last_updated TIMESTAMP
            )
            """)
        
        # Get current timestamp
        now = datetime.now()
        
        # Process matrix data
        for family, info in matrix.items():
            # Check if model exists
            result = conn.execute("SELECT * FROM model_compatibility WHERE model_family = ?", [family]).fetchall()
            
            # Prepare hardware compatibility data
            hw_data = {}
            test_date = None
            
            for hw, hw_info in info.get("hardware_compatibility", {}).items():
                hw_data[hw] = hw_info.get("success", False)
                
                # Get most recent test date
                if hw_info.get("tested", False) and hw_info.get("timestamp"):
                    hw_timestamp = datetime.fromisoformat(hw_info["timestamp"])
                    if test_date is None or hw_timestamp > test_date:
                        test_date = hw_timestamp
            
            if not test_date:
                test_date = now
            
            if result:
                # Update existing record
                conn.execute("""
                UPDATE model_compatibility SET
                    architecture_type = ?,
                    category = ?,
                    priority = ?,
                    cpu = ?,
                    cuda = ?,
                    rocm = ?,
                    mps = ?,
                    openvino = ?,
                    qualcomm = ?,
                    webnn = ?,
                    webgpu = ?,
                    test_date = ?,
                    last_updated = ?
                WHERE model_family = ?
                """, [
                    info.get("architecture_type", "unknown"),
                    info.get("category", "other"),
                    info.get("priority", "medium"),
                    hw_data.get("cpu", False),
                    hw_data.get("cuda", False),
                    hw_data.get("rocm", False),
                    hw_data.get("mps", False),
                    hw_data.get("openvino", False),
                    hw_data.get("qualcomm", False),
                    hw_data.get("webnn", False),
                    hw_data.get("webgpu", False),
                    test_date,
                    now,
                    family
                ])
            else:
                # Insert new record
                conn.execute("""
                INSERT INTO model_compatibility (
                    model_family,
                    architecture_type,
                    category,
                    priority,
                    cpu,
                    cuda,
                    rocm,
                    mps,
                    openvino,
                    qualcomm,
                    webnn,
                    webgpu,
                    test_date,
                    last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    family,
                    info.get("architecture_type", "unknown"),
                    info.get("category", "other"),
                    info.get("priority", "medium"),
                    hw_data.get("cpu", False),
                    hw_data.get("cuda", False),
                    hw_data.get("rocm", False),
                    hw_data.get("mps", False),
                    hw_data.get("openvino", False),
                    hw_data.get("qualcomm", False),
                    hw_data.get("webnn", False),
                    hw_data.get("webgpu", False),
                    test_date,
                    now
                ])
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info(f"Updated DuckDB database with compatibility matrix data: {DB_PATH}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating DuckDB database: {e}")
        return False

def generate_coverage_report():
    """
    Generate a coverage report for HuggingFace models.
    
    Returns:
        Path to the generated report file
    """
    # Analyze test files
    analysis = analyze_test_files()
    
    # Get all HuggingFace models from categories
    all_models = set()
    for category in MODEL_CATEGORIES.values():
        all_models.update(category["families"])
    
    # Get models with tests
    models_with_tests = set(analysis.get("models", {}).keys())
    
    # Find missing models
    missing_models = all_models - models_with_tests
    
    # Group by architecture type and priority
    architecture_coverage = defaultdict(lambda: {"total": 0, "covered": 0, "missing": []})
    priority_coverage = defaultdict(lambda: {"total": 0, "covered": 0, "missing": []})
    
    for category_name, category_info in MODEL_CATEGORIES.items():
        arch_type = category_info["architecture_type"]
        priority = category_info["priority"]
        
        for family in category_info["families"]:
            # Update architecture coverage
            architecture_coverage[arch_type]["total"] += 1
            if family in models_with_tests:
                architecture_coverage[arch_type]["covered"] += 1
            else:
                architecture_coverage[arch_type]["missing"].append(family)
            
            # Update priority coverage
            priority_coverage[priority]["total"] += 1
            if family in models_with_tests:
                priority_coverage[priority]["covered"] += 1
            else:
                priority_coverage[priority]["missing"].append(family)
    
    # Generate report
    report_file = REPORTS_DIR / f"model_coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, "w") as f:
        f.write("# HuggingFace Model Coverage Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall coverage
        f.write("## Overall Coverage\n\n")
        total_models = len(all_models)
        covered_models = len(models_with_tests)
        coverage_percent = (covered_models / total_models) * 100 if total_models > 0 else 0
        
        f.write(f"- Total model families: {total_models}\n")
        f.write(f"- Families with tests: {covered_models}\n")
        f.write(f"- Missing tests: {len(missing_models)}\n")
        f.write(f"- Coverage: {coverage_percent:.1f}%\n\n")
        
        # Architecture coverage
        f.write("## Coverage by Architecture Type\n\n")
        f.write("| Architecture Type | Coverage | Percentage |\n")
        f.write("|------------------|----------|------------|\n")
        
        for arch_type, coverage in sorted(architecture_coverage.items()):
            total = coverage["total"]
            covered = coverage["covered"]
            percentage = (covered / total) * 100 if total > 0 else 0
            
            f.write(f"| {arch_type} | {covered}/{total} | {percentage:.1f}% |\n")
        
        f.write("\n")
        
        # Priority coverage
        f.write("## Coverage by Priority\n\n")
        f.write("| Priority | Coverage | Percentage |\n")
        f.write("|----------|----------|------------|\n")
        
        for priority, coverage in sorted(priority_coverage.items()):
            total = coverage["total"]
            covered = coverage["covered"]
            percentage = (covered / total) * 100 if total > 0 else 0
            
            f.write(f"| {priority} | {covered}/{total} | {percentage:.1f}% |\n")
        
        f.write("\n")
        
        # Missing critical models
        missing_critical = priority_coverage["critical"]["missing"]
        if missing_critical:
            f.write("## Missing Critical Models\n\n")
            
            for family in sorted(missing_critical):
                # Find architecture type and description
                arch_type = None
                description = None
                
                for category_info in MODEL_CATEGORIES.values():
                    if family in category_info["families"]:
                        arch_type = category_info["architecture_type"]
                        description = category_info["description"]
                        break
                
                f.write(f"- **{family}** ({arch_type}): Part of {description}\n")
            
            f.write("\n")
        
        # Missing high priority models (limit to first 10)
        missing_high = priority_coverage["high"]["missing"]
        if missing_high:
            f.write("## Missing High Priority Models (Top 10)\n\n")
            
            for family in sorted(missing_high)[:10]:
                # Find architecture type and description
                arch_type = None
                description = None
                
                for category_info in MODEL_CATEGORIES.values():
                    if family in category_info["families"]:
                        arch_type = category_info["architecture_type"]
                        description = category_info["description"]
                        break
                
                f.write(f"- **{family}** ({arch_type}): Part of {description}\n")
            
            if len(missing_high) > 10:
                f.write(f"- ... and {len(missing_high) - 10} more high priority models\n")
            
            f.write("\n")
    
    logger.info(f"Coverage report saved to {report_file}")
    return str(report_file)

def identify_gaps():
    """
    Identify gaps in model coverage and generate a prioritized list of models to implement next.
    
    Returns:
        Dict containing gap analysis
    """
    # Analyze test files
    analysis = analyze_test_files()
    
    # Get models with tests
    models_with_tests = set(analysis.get("models", {}).keys())
    
    # Create a list of missing models with priority information
    missing_models = []
    
    for category_name, category_info in MODEL_CATEGORIES.items():
        category_priority = category_info["priority"]
        arch_type = category_info["architecture_type"]
        
        for family in category_info["families"]:
            if family not in models_with_tests:
                missing_models.append({
                    "family": family,
                    "category": category_name,
                    "architecture_type": arch_type,
                    "priority": category_priority
                })
    
    # Sort by priority (critical first, then high, then medium)
    priority_order = {"critical": 0, "high": 1, "medium": 2}
    missing_models.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    # Generate report
    report_file = REPORTS_DIR / f"model_gaps_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, "w") as f:
        f.write("# HuggingFace Model Coverage Gaps\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"Identified {len(missing_models)} model families without test coverage\n\n")
        
        # Prioritized list
        f.write("## Prioritized Implementation List\n\n")
        f.write("| Rank | Model Family | Architecture | Category | Priority |\n")
        f.write("|------|-------------|--------------|----------|----------|\n")
        
        for i, model in enumerate(missing_models, 1):
            family = model["family"]
            arch_type = model["architecture_type"]
            category = model["category"]
            priority = model["priority"]
            
            f.write(f"| {i} | {family} | {arch_type} | {category} | {priority} |\n")
        
        f.write("\n")
        
        # Next steps
        f.write("## Recommended Next Steps\n\n")
        f.write("To improve model coverage, implement tests for these models in order:\n\n")
        
        # List top 5 critical and high priority models
        critical_models = [m for m in missing_models if m["priority"] == "critical"][:5]
        high_models = [m for m in missing_models if m["priority"] == "high"][:5]
        
        if critical_models:
            f.write("### Critical Priority Models\n\n")
            for model in critical_models:
                f.write(f"1. **{model['family']}** ({model['architecture_type']})\n")
                f.write(f"   - Category: {model['category']}\n")
                f.write(f"   - Architecture: {model['architecture_type']}\n")
                f.write(f"   - Suggested template: `templates/{model['architecture_type'].replace('-', '_')}_template.py`\n")
                f.write("\n")
        
        if high_models:
            f.write("### High Priority Models\n\n")
            for model in high_models:
                f.write(f"1. **{model['family']}** ({model['architecture_type']})\n")
                f.write(f"   - Category: {model['category']}\n")
                f.write(f"   - Architecture: {model['architecture_type']}\n")
                f.write(f"   - Suggested template: `templates/{model['architecture_type'].replace('-', '_')}_template.py`\n")
                f.write("\n")
    
    logger.info(f"Gaps report saved to {report_file}")
    
    # Save JSON version for programmatic use
    json_file = REPORTS_DIR / f"model_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_missing": len(missing_models),
            "missing_models": missing_models
        }, f, indent=2)
    
    logger.info(f"Gaps data saved to {json_file}")
    
    return {
        "report": str(report_file),
        "data": str(json_file),
        "missing_models": missing_models
    }

def main():
    parser = argparse.ArgumentParser(description="Generate HuggingFace Model Compatibility Matrix")
    
    # Action options
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--analyze", action="store_true", 
                            help="Analyze existing test files")
    action_group.add_argument("--run-tests", action="store_true", 
                            help="Run tests on available hardware platforms")
    action_group.add_argument("--generate-matrix", action="store_true", 
                            help="Generate compatibility matrix from test results")
    action_group.add_argument("--generate-report", action="store_true", 
                            help="Generate coverage report")
    action_group.add_argument("--identify-gaps", action="store_true", 
                            help="Identify gaps in model coverage")
    
    # Test options
    parser.add_argument("--hardware", type=str, nargs="+", choices=HARDWARE_PLATFORMS.keys(),
                        help="Hardware platforms to test on")
    parser.add_argument("--max-models", type=int, 
                        help="Maximum number of models to test")
    parser.add_argument("--priority", type=str, choices=["critical", "high", "medium", "all"],
                        default="all", help="Priority level of models to test")
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze existing test files
        analysis = analyze_test_files()
        
        print("\nAnalysis Results:")
        print(f"Total models with tests: {analysis.get('total_models', 0)}")
        
        print("\nArchitecture Counts:")
        for arch, count in sorted(analysis.get("architecture_counts", {}).items()):
            print(f"  - {arch}: {count}")
        
        print("\nCategory Counts:")
        for category, count in sorted(analysis.get("category_counts", {}).items()):
            print(f"  - {category}: {count}")
        
        print("\nPriority Counts:")
        for priority, count in sorted(analysis.get("priority_counts", {}).items()):
            print(f"  - {priority}: {count}")
        
        # Save analysis to file
        analysis_file = REPORTS_DIR / f"model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nAnalysis saved to {analysis_file}")
    
    elif args.run_tests:
        # Run tests
        results = run_tests(
            hardware_platforms=args.hardware,
            max_models=args.max_models,
            priority=args.priority
        )
        
        print("\nTest Results:")
        print(f"Total tests: {results.get('total_tests', 0)}")
        print(f"Successful tests: {results.get('successful_tests', 0)}")
        print(f"Failed tests: {results.get('failed_tests', 0)}")
        
        success_rate = (results.get("successful_tests", 0) / results.get("total_tests", 1)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    elif args.generate_matrix:
        # Generate compatibility matrix
        matrix_files = generate_compatibility_matrix()
        
        print("\nCompatibility Matrix Files:")
        for format_name, file_path in matrix_files.items():
            print(f"  - {format_name.upper()}: {file_path}")
    
    elif args.generate_report:
        # Generate coverage report
        report_file = generate_coverage_report()
        
        print(f"\nCoverage report saved to {report_file}")
    
    elif args.identify_gaps:
        # Identify gaps in model coverage
        gaps = identify_gaps()
        
        print(f"\nGaps report saved to {gaps['report']}")
        print(f"Gaps data saved to {gaps['data']}")
        
        print(f"\nIdentified {len(gaps['missing_models'])} missing models")
        
        # Print top 3 critical models
        critical_models = [m for m in gaps['missing_models'] if m["priority"] == "critical"][:3]
        if critical_models:
            print("\nTop critical models to implement next:")
            for i, model in enumerate(critical_models, 1):
                print(f"  {i}. {model['family']} ({model['architecture_type']})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())