#!/usr/bin/env python3
"""
Script to test priority models and establish performance baselines.

This script:
1. Detects available hardware platforms
2. Runs tests for top 10 priority models
3. Establishes performance baselines
4. Generates a comprehensive report

Usage:
    python scripts/test_priority_models.py
    python scripts/test_priority_models.py --update-baselines
    python scripts/test_priority_models.py --hardware cpu
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Priority models to test
PRIORITY_MODELS = [
    {
        "name": "GPT-2",
        "test_file": "test_hf_gpt2_improved.py",
        "category": "text",
        "description": "Text generation"
    },
    {
        "name": "CLIP",
        "test_file": "test_hf_clip_improved.py",
        "category": "multimodal",
        "description": "Vision + text"
    },
    {
        "name": "LLaMA",
        "test_file": "test_hf_llama_improved.py",
        "category": "text",
        "description": "Large language model"
    },
    {
        "name": "Whisper",
        "test_file": "test_hf_whisper_improved.py",
        "category": "audio",
        "description": "Audio transcription"
    },
    {
        "name": "T5",
        "test_file": "test_hf_t5_improved.py",
        "category": "text",
        "description": "Text-to-text"
    },
    {
        "name": "ViT",
        "test_file": "test_hf_vit_improved.py",
        "category": "vision",
        "description": "Vision transformer"
    },
    {
        "name": "BERT",
        "test_file": "test_hf_bert_improved.py",
        "category": "text",
        "description": "Text encoding"
    },
    {
        "name": "ResNet",
        "test_file": "test_hf_resnet_improved.py",
        "category": "vision",
        "description": "Computer vision"
    },
    {
        "name": "Wav2Vec2",
        "test_file": "test_hf_wav2vec2_improved.py",
        "category": "audio",
        "description": "Audio encoding"
    },
    {
        "name": "BART",
        "test_file": "test_hf_bart_improved.py",
        "category": "text",
        "description": "Seq2seq"
    }
]


def detect_hardware():
    """Detect available hardware platforms."""
    hardware = {"cpu": True}
    
    try:
        import torch
        hardware["pytorch"] = torch.__version__
        
        if torch.cuda.is_available():
            hardware["cuda"] = True
            hardware["cuda_devices"] = torch.cuda.device_count()
            hardware["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hardware["mps"] = True
    except ImportError:
        hardware["pytorch"] = False
        print("WARNING: PyTorch not installed, cannot detect GPU hardware")
    
    try:
        import tensorflow as tf
        hardware["tensorflow"] = tf.__version__
    except ImportError:
        hardware["tensorflow"] = False
    
    return hardware


def check_test_file_exists(test_file):
    """Check if test file exists."""
    test_path = Path("test/improved") / test_file
    return test_path.exists()


def run_model_test(test_file, update_baselines=False, hardware="cpu"):
    """Run a single model test."""
    test_path = Path("test/improved") / test_file
    
    if not test_path.exists():
        return {
            "status": "SKIP",
            "reason": "Test file not found",
            "output": ""
        }
    
    # Build pytest command
    cmd = ["pytest", str(test_path), "-v", "--run-model-tests"]
    
    if update_baselines:
        cmd.append("--update-baselines")
    
    # Add hardware marker if not cpu
    if hardware != "cpu" and hardware != "all":
        cmd.extend(["-m", hardware])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "returncode": result.returncode,
            "output": result.stdout + result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT",
            "reason": "Test timed out after 5 minutes",
            "output": ""
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "reason": str(e),
            "output": ""
        }


def load_baselines():
    """Load existing performance baselines."""
    baseline_path = Path("test/.performance_baselines.json")
    
    if baseline_path.exists():
        try:
            with open(baseline_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load baselines: {e}")
            return {}
    
    return {}


def generate_report(results, hardware_info, baselines):
    """Generate a comprehensive test report."""
    report = []
    report.append("=" * 80)
    report.append("PHASE 3 PRIORITY MODELS TEST REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Hardware information
    report.append("HARDWARE INFORMATION")
    report.append("-" * 80)
    report.append(f"CPU: Available")
    
    if hardware_info.get("pytorch"):
        report.append(f"PyTorch: {hardware_info['pytorch']}")
        
        if hardware_info.get("cuda"):
            report.append(f"CUDA: Available ({hardware_info['cuda_devices']} device(s))")
            report.append(f"  Device: {hardware_info.get('cuda_device_name', 'Unknown')}")
        else:
            report.append("CUDA: Not available")
        
        if hardware_info.get("mps"):
            report.append("MPS (Apple Silicon): Available")
        else:
            report.append("MPS (Apple Silicon): Not available")
    else:
        report.append("PyTorch: Not installed")
    
    if hardware_info.get("tensorflow"):
        report.append(f"TensorFlow: {hardware_info['tensorflow']}")
    
    report.append("")
    
    # Test results
    report.append("TEST RESULTS")
    report.append("-" * 80)
    
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    
    for model_name, result in results.items():
        status = result["status"]
        
        if status == "PASS":
            passed += 1
            icon = "✅"
        elif status == "FAIL":
            failed += 1
            icon = "❌"
        elif status == "SKIP":
            skipped += 1
            icon = "⏭️"
        else:
            errors += 1
            icon = "⚠️"
        
        report.append(f"{icon} {model_name}: {status}")
        
        if "reason" in result:
            report.append(f"   Reason: {result['reason']}")
    
    report.append("")
    report.append(f"Summary: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
    report.append("")
    
    # Baseline information
    if baselines:
        report.append("PERFORMANCE BASELINES")
        report.append("-" * 80)
        report.append(f"Total baselines stored: {len(baselines)}")
        
        # Group by model
        model_baselines = {}
        for key in baselines.keys():
            model_name = key.split("_")[0] if "_" in key else key
            if model_name not in model_baselines:
                model_baselines[model_name] = []
            model_baselines[model_name].append(key)
        
        for model_name in sorted(model_baselines.keys()):
            report.append(f"  {model_name}: {len(model_baselines[model_name])} baseline(s)")
    else:
        report.append("PERFORMANCE BASELINES")
        report.append("-" * 80)
        report.append("No baselines established yet.")
        report.append("Run with --update-baselines to establish baselines.")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Test priority models and establish baselines"
    )
    parser.add_argument(
        "--update-baselines",
        action="store_true",
        help="Update performance baselines"
    )
    parser.add_argument(
        "--hardware",
        choices=["cpu", "cuda", "mps", "all"],
        default="cpu",
        help="Hardware platform to test on (default: cpu)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (default: all priority models)"
    )
    parser.add_argument(
        "--output",
        help="Output file for report (default: stdout)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Phase 3: Testing Priority Models")
    print("=" * 80)
    print()
    
    # Detect hardware
    print("🔍 Detecting hardware...")
    hardware_info = detect_hardware()
    print(f"   CPU: Available")
    if hardware_info.get("pytorch"):
        print(f"   PyTorch: {hardware_info['pytorch']}")
        if hardware_info.get("cuda"):
            print(f"   CUDA: {hardware_info['cuda_devices']} device(s)")
        if hardware_info.get("mps"):
            print(f"   MPS: Available")
    print()
    
    # Filter models if specific ones requested
    models_to_test = PRIORITY_MODELS
    if args.models:
        models_to_test = [
            m for m in PRIORITY_MODELS
            if m["name"].lower() in [name.lower() for name in args.models]
        ]
    
    print(f"📋 Testing {len(models_to_test)} priority model(s)...")
    print()
    
    # Run tests
    results = {}
    
    for i, model in enumerate(models_to_test, 1):
        model_name = model["name"]
        test_file = model["test_file"]
        
        print(f"[{i}/{len(models_to_test)}] Testing {model_name}...", end=" ", flush=True)
        
        result = run_model_test(test_file, args.update_baselines, args.hardware)
        results[model_name] = result
        
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️",
            "TIMEOUT": "⏱️",
            "ERROR": "⚠️"
        }.get(result["status"], "❓")
        
        print(f"{status_icon} {result['status']}")
        
        # Show error details if failed
        if result["status"] in ["FAIL", "ERROR", "TIMEOUT"] and "reason" in result:
            print(f"   Reason: {result['reason']}")
    
    print()
    
    # Load baselines
    baselines = load_baselines()
    
    # Generate report
    report = generate_report(results, hardware_info, baselines)
    
    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {output_path}")
    else:
        print()
        print(report)
    
    # Exit with appropriate code
    failed_count = sum(1 for r in results.values() if r["status"] in ["FAIL", "ERROR"])
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
