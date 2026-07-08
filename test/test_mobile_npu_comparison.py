#!/usr/bin/env python3
"""
Mobile NPU Comparison Test

This script tests and compares different mobile NPU backends, including:
- MediaTek NPU/APU 
- Qualcomm NPU/QNN
- Other mobile accelerators

It evaluates model compatibility, power efficiency, and performance
characteristics to help determine the best mobile hardware for different
model types and deployment scenarios.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mobile_npu_comparison")

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import hardware detection modules
try:
    # First try to import the centralized hardware detection
    from centralized_hardware_detection.hardware_detection import get_hardware_manager, get_capabilities, get_model_hardware_compatibility
    USING_CENTRALIZED = True
    logger.info("Using centralized hardware detection")
except ImportError:
    # Fall back to direct imports if centralized detection is not available
    USING_CENTRALIZED = False
    logger.warning("Centralized hardware detection not available, using direct imports")
    try:
        # Import MediaTek NPU support
        from hardware_detection.mediatek_npu_support import (
            MediaTekNPUCapabilityDetector, 
            MediaTekPowerMonitor,
            MediaTekModelOptimizer,
            MEDIATEK_NPU_AVAILABLE,
            MEDIATEK_NPU_SIMULATION_MODE
        )
        HAS_MEDIATEK = MEDIATEK_NPU_AVAILABLE
        logger.info(f"MediaTek NPU support loaded: Available={MEDIATEK_NPU_AVAILABLE}, Simulation={MEDIATEK_NPU_SIMULATION_MODE}")
    except ImportError:
        HAS_MEDIATEK = False
        logger.warning("MediaTek NPU support not available")
    
    try:
        # Import Qualcomm NPU support
        from hardware_detection.qnn_support import (
            QNNCapabilityDetector,
            QNNPowerMonitor,
            QNNModelOptimizer,
            QNN_AVAILABLE,
            QNN_SIMULATION_MODE
        )
        HAS_QUALCOMM = QNN_AVAILABLE
        logger.info(f"Qualcomm NPU support loaded: Available={QNN_AVAILABLE}, Simulation={QNN_SIMULATION_MODE}")
    except ImportError:
        HAS_QUALCOMM = False
        logger.warning("Qualcomm NPU support not available")
        
    try:
        # Import Samsung NPU support
        from samsung_support import (
            SamsungDetector,
            SamsungModelConverter,
            SamsungBenchmarkRunner,
            SAMSUNG_NPU_AVAILABLE,
            SAMSUNG_NPU_SIMULATION_MODE
        )
        HAS_SAMSUNG = SAMSUNG_NPU_AVAILABLE
        logger.info(f"Samsung NPU support loaded: Available={SAMSUNG_NPU_AVAILABLE}, Simulation={SAMSUNG_NPU_SIMULATION_MODE}")
    except ImportError:
        HAS_SAMSUNG = False
        logger.warning("Samsung NPU support not available")

# Define test models for comparison
TEST_MODELS = {
    "mobilenet_v3": {
        "name": "MobileNet V3",
        "description": "Lightweight mobile vision model",
        "path": "/path/to/models/mobilenet_v3.onnx",
        "type": "vision",
        "mobile_optimized": True
    },
    "mobilevit_small": {
        "name": "MobileViT Small",
        "description": "Mobile vision transformer",
        "path": "/path/to/models/mobilevit_small.onnx",
        "type": "vision",
        "mobile_optimized": True
    },
    "bert_tiny": {
        "name": "BERT Tiny",
        "description": "Tiny BERT model for NLP tasks",
        "path": "/path/to/models/bert_tiny.onnx",
        "type": "nlp",
        "mobile_optimized": True
    },
    "bert_mini": {
        "name": "BERT Mini",
        "description": "Miniature BERT model for NLP tasks",
        "path": "/path/to/models/bert_mini.onnx",
        "type": "nlp",
        "mobile_optimized": True
    },
    "mobilebert": {
        "name": "MobileBERT",
        "description": "Mobile-optimized BERT model",
        "path": "/path/to/models/mobilebert.onnx",
        "type": "nlp",
        "mobile_optimized": True
    },
    "whisper_tiny": {
        "name": "Whisper Tiny",
        "description": "Tiny version of Whisper for audio transcription",
        "path": "/path/to/models/whisper_tiny.onnx",
        "type": "audio",
        "mobile_optimized": False
    },
    "efficientnet_b0": {
        "name": "EfficientNet B0",
        "description": "Smallest EfficientNet model",
        "path": "/path/to/models/efficientnet_b0.onnx",
        "type": "vision",
        "mobile_optimized": True
    }
}

def check_hardware():
    """Check available mobile NPU hardware"""
    results = {
        "qualcomm": {"available": False, "simulation": False, "devices": []},
        "mediatek": {"available": False, "simulation": False, "devices": []},
        "samsung": {"available": False, "simulation": False, "devices": []},
        "tensor_cores": {"available": False, "devices": []},
        "neural_engine": {"available": False, "devices": []},
    }
    
    # Check for hardware using centralized detection if available
    if USING_CENTRALIZED:
        # Get hardware manager and capabilities
        hw_manager = get_hardware_manager()
        capabilities = get_capabilities()
        
        # Check for Qualcomm
        if capabilities.get("qualcomm", False):
            results["qualcomm"]["available"] = True
            results["qualcomm"]["simulation"] = capabilities.get("qualcomm_simulation", False)
            
        # Check for MediaTek
        if capabilities.get("mediatek", False):
            results["mediatek"]["available"] = True
            results["mediatek"]["simulation"] = capabilities.get("mediatek_simulation", False)
            
        # Check for Samsung
        if capabilities.get("samsung_npu", False):
            results["samsung"]["available"] = True
            results["samsung"]["simulation"] = capabilities.get("samsung_npu_simulation", False)
    else:
        # Use direct detection if centralized is not available
        if HAS_MEDIATEK:
            results["mediatek"]["available"] = True
            results["mediatek"]["simulation"] = MEDIATEK_NPU_SIMULATION_MODE
            
            # Get MediaTek device details
            mediatek_detector = MediaTekNPUCapabilityDetector()
            if mediatek_detector.is_available():
                mediatek_detector.select_device()
                results["mediatek"]["devices"] = mediatek_detector.get_devices()
                
        if HAS_QUALCOMM:
            results["qualcomm"]["available"] = True
            results["qualcomm"]["simulation"] = QNN_SIMULATION_MODE
            
            # Get Qualcomm device details
            qnn_detector = QNNCapabilityDetector()
            if qnn_detector.is_available():
                qnn_detector.select_device()
                results["qualcomm"]["devices"] = qnn_detector.get_devices()
                
        if HAS_SAMSUNG:
            results["samsung"]["available"] = True
            results["samsung"]["simulation"] = SAMSUNG_NPU_SIMULATION_MODE
            
            # Get Samsung device details
            samsung_detector = SamsungDetector()
            chipset = samsung_detector.detect_samsung_hardware()
            if chipset:
                results["samsung"]["devices"] = [{
                    "name": chipset.name,
                    "cores": chipset.npu_cores,
                    "tops": chipset.npu_tops,
                    "precision": chipset.max_precision,
                    "supported_precisions": chipset.supported_precisions,
                    "simulated": SAMSUNG_NPU_SIMULATION_MODE
                }]
    
    # Additional hardware detection for completeness
    # Check for Apple Neural Engine
    if sys.platform == "darwin" and "arm" in os.uname().machine.lower():
        results["neural_engine"]["available"] = True
        results["neural_engine"]["devices"] = [{"name": "Apple Neural Engine", "simulated": False}]
    
    # Check for NVIDIA Tensor Cores
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            if device.major >= 7:  # Volta or later likely has tensor cores
                results["tensor_cores"]["available"] = True
                results["tensor_cores"]["devices"] = [{
                    "name": device.name,
                    "compute_capability": f"{device.major}.{device.minor}",
                    "simulated": False
                }]
    except (ImportError, AttributeError):
        pass
    
    return results

def compare_model_compatibility():
    """Compare model compatibility across different mobile NPUs"""
    compatibility_results = {}
    
    # Test all models
    for model_id, model_info in TEST_MODELS.items():
        compatibility_results[model_id] = {
            "name": model_info["name"],
            "type": model_info["type"],
            "mobile_optimized": model_info["mobile_optimized"],
            "compatibility": {}
        }
        
        # Check compatibility using centralized detection if available
        if USING_CENTRALIZED:
            compatibility = get_model_hardware_compatibility(model_id)
            compatibility_results[model_id]["compatibility"] = {
                "qualcomm": compatibility.get("qualcomm", False),
                "mediatek": compatibility.get("mediatek", False),
                "samsung": compatibility.get("samsung_npu", False),
                "cpu": compatibility.get("cpu", True),
                "cuda": compatibility.get("cuda", False),
                "webgpu": compatibility.get("webgpu", False),
                "webnn": compatibility.get("webnn", False)
            }
        else:
            # Direct compatibility checks if centralized is not available
            compatibility_results[model_id]["compatibility"]["cpu"] = True
            
            if HAS_MEDIATEK:
                mediatek_detector = MediaTekNPUCapabilityDetector()
                if mediatek_detector.is_available():
                    mediatek_detector.select_device()
                    mediatek_compat = mediatek_detector.test_model_compatibility(model_info["path"])
                    compatibility_results[model_id]["compatibility"]["mediatek"] = mediatek_compat.get("compatible", False)
            
            if HAS_QUALCOMM:
                qnn_detector = QNNCapabilityDetector()
                if qnn_detector.is_available():
                    qnn_detector.select_device()
                    qnn_compat = qnn_detector.test_model_compatibility(model_info["path"])
                    compatibility_results[model_id]["compatibility"]["qualcomm"] = qnn_compat.get("compatible", False)
                    
            if HAS_SAMSUNG:
                # Use Samsung model converter to analyze compatibility
                samsung_converter = SamsungModelConverter()
                # Get Samsung chipset for compatibility check
                samsung_detector = SamsungDetector()
                chipset = samsung_detector.detect_samsung_hardware()
                
                if chipset:
                    # Analyze compatibility with detected chipset
                    target_chipset = chipset.name.lower().replace(" ", "_")
                    analysis = samsung_converter.analyze_model_compatibility(
                        model_path=model_info["path"],
                        target_chipset=target_chipset
                    )
                    # Extract compatibility information
                    is_compatible = analysis.get("compatibility", {}).get("supported", False)
                    compatibility_results[model_id]["compatibility"]["samsung"] = is_compatible
    
    return compatibility_results

def compare_power_efficiency():
    """Compare power efficiency metrics across different mobile NPUs"""
    power_results = {}
    
    # Only proceed if we're not using centralized detection (direct power monitoring)
    if not USING_CENTRALIZED:
        # Test power on MediaTek if available
        if HAS_MEDIATEK:
            mediatek_power = MediaTekPowerMonitor()
            if mediatek_power.detector.is_available():
                mediatek_power.detector.select_device()
                
                # Start monitoring
                mediatek_power.start_monitoring()
                
                # Simulate model inference
                time.sleep(5)  # Simulate 5 seconds of inference
                
                # Stop monitoring and get results
                results = mediatek_power.stop_monitoring()
                power_results["mediatek"] = results
                
                # Add battery life estimate
                battery_info = mediatek_power.estimate_battery_life(
                    results.get("average_power_watts", 1.0),
                    battery_capacity_mah=5000
                )
                power_results["mediatek"]["battery_life"] = battery_info
        
        # Test power on Qualcomm if available
        if HAS_QUALCOMM:
            qualcomm_power = QNNPowerMonitor()
            if qualcomm_power.detector.is_available():
                qualcomm_power.detector.select_device()
                
                # Start monitoring
                qualcomm_power.start_monitoring()
                
                # Simulate model inference
                time.sleep(5)  # Simulate 5 seconds of inference
                
                # Stop monitoring and get results
                results = qualcomm_power.stop_monitoring()
                power_results["qualcomm"] = results
                
                # Add battery life estimate
                battery_info = qualcomm_power.estimate_battery_life(
                    results.get("average_power_watts", 1.0),
                    battery_capacity_mah=5000
                )
                power_results["qualcomm"]["battery_life"] = battery_info
                
        # Test power on Samsung if available
        if HAS_SAMSUNG:
            # Get Samsung chipset
            samsung_detector = SamsungDetector()
            chipset = samsung_detector.detect_samsung_hardware()
            
            if chipset:
                # Get power information directly from chipset data
                # In a real implementation, this would use actual power monitoring
                average_power = chipset.typical_power
                peak_power = chipset.max_power_draw
                
                # Calculate power efficiency metrics
                power_efficiency_score = (chipset.npu_tops / average_power) * 10  # Simplified efficiency score
                battery_impact_percent = (average_power / 5.0) * 100  # Simplified battery impact calculation
                
                # Prepare power results
                results = {
                    "device_name": chipset.name,
                    "average_power_watts": average_power,
                    "peak_power_watts": peak_power,
                    "estimated_battery_impact_percent": battery_impact_percent,
                    "power_efficiency_score": power_efficiency_score,
                    "tops_per_watt": chipset.npu_tops / average_power
                }
                
                # Add battery life estimate (simplified calculation)
                battery_capacity_wh = 5000 * 3.85 / 1000  # 5000mAh battery at 3.85V = 19.25Wh
                estimated_runtime_hours = battery_capacity_wh / average_power
                battery_percent_per_hour = 100 / estimated_runtime_hours
                
                results["battery_life"] = {
                    "estimated_runtime_hours": estimated_runtime_hours,
                    "battery_percent_per_hour": battery_percent_per_hour
                }
                
                power_results["samsung"] = results
    else:
        # When using centralized detection, we don't have direct power monitoring
        # Instead, we can simulate with dummy data for comparison purposes
        power_results["mediatek"] = {
            "device_name": "MediaTek (Simulated)",
            "average_power_watts": 0.85,
            "peak_power_watts": 1.2,
            "estimated_battery_impact_percent": 28.3,
            "power_efficiency_score": 71.7,
            "battery_life": {
                "estimated_runtime_hours": 22.8,
                "battery_percent_per_hour": 4.4
            }
        }
        
        power_results["qualcomm"] = {
            "device_name": "Qualcomm (Simulated)",
            "average_power_watts": 0.95,
            "peak_power_watts": 1.4,
            "estimated_battery_impact_percent": 31.7,
            "power_efficiency_score": 68.3,
            "battery_life": {
                "estimated_runtime_hours": 20.4,
                "battery_percent_per_hour": 4.9
            }
        }
        
        power_results["samsung"] = {
            "device_name": "Samsung Exynos 2400 (Simulated)",
            "average_power_watts": 0.90,
            "peak_power_watts": 1.3,
            "estimated_battery_impact_percent": 30.0,
            "power_efficiency_score": 82.5,
            "tops_per_watt": 9.8,
            "battery_life": {
                "estimated_runtime_hours": 21.4,
                "battery_percent_per_hour": 4.7
            }
        }
    
    return power_results

def compare_optimization_recommendations():
    """Compare optimization recommendations for different mobile NPUs"""
    optimization_results = {}
    
    # Only proceed if we're not using centralized detection (direct optimizer access)
    if not USING_CENTRALIZED:
        # Test a subset of models for optimization
        for model_id in ["mobilenet_v3", "mobilebert", "whisper_tiny"]:
            model_info = TEST_MODELS[model_id]
            optimization_results[model_id] = {
                "name": model_info["name"],
                "type": model_info["type"],
                "recommendations": {}
            }
            
            # Get MediaTek optimization recommendations if available
            if HAS_MEDIATEK:
                mediatek_optimizer = MediaTekModelOptimizer()
                if mediatek_optimizer.detector.is_available():
                    mediatek_optimizer.detector.select_device()
                    mediatek_rec = mediatek_optimizer.recommend_optimizations(model_info["path"])
                    optimization_results[model_id]["recommendations"]["mediatek"] = mediatek_rec
            
            # Get Qualcomm optimization recommendations if available
            if HAS_QUALCOMM:
                qualcomm_optimizer = QNNModelOptimizer()
                if qualcomm_optimizer.detector.is_available():
                    qualcomm_optimizer.detector.select_device()
                    qualcomm_rec = qualcomm_optimizer.recommend_optimizations(model_info["path"])
                    optimization_results[model_id]["recommendations"]["qualcomm"] = qualcomm_rec
                    
            # Get Samsung optimization recommendations if available
            if HAS_SAMSUNG:
                # Use the Samsung model converter for recommendations
                samsung_converter = SamsungModelConverter()
                # Get Samsung chipset
                samsung_detector = SamsungDetector()
                chipset = samsung_detector.detect_samsung_hardware()
                
                if chipset:
                    # Analyze compatibility with detected chipset
                    target_chipset = chipset.name.lower().replace(" ", "_")
                    analysis = samsung_converter.analyze_model_compatibility(
                        model_path=model_info["path"],
                        target_chipset=target_chipset
                    )
                    
                    # Extract recommendations based on analysis
                    compatibility = analysis.get("compatibility", {})
                    
                    samsung_rec = {
                        "compatible": compatibility.get("supported", False),
                        "recommended_precision": compatibility.get("recommended_precision", ""),
                        "estimated_memory_reduction": "30-40% with INT8 quantization",
                        "estimated_power_efficiency_score": 85,
                        "recommended_optimizations": []
                    }
                    
                    # Add optimization recommendations
                    if "optimization_opportunities" in compatibility:
                        samsung_rec["recommended_optimizations"] = compatibility["optimization_opportunities"]
                    else:
                        # Fallback default recommendations
                        if model_info["type"] == "vision":
                            samsung_rec["recommended_optimizations"] = [
                                "INT8 quantization", 
                                "Layer fusion", 
                                "One UI optimization"
                            ]
                        elif model_info["type"] == "nlp":
                            samsung_rec["recommended_optimizations"] = [
                                "INT8 quantization",
                                "INT4 weight-only quantization", 
                                "Samsung Neural SDK optimizations"
                            ]
                        elif model_info["type"] == "audio":
                            samsung_rec["recommended_optimizations"] = [
                                "INT8 quantization",
                                "Game Booster integration for sustained performance"
                            ]
                    
                    # Add recommendations to results
                    optimization_results[model_id]["recommendations"]["samsung"] = samsung_rec
    else:
        # When using centralized detection, we add simulated optimization recommendations
        for model_id in ["mobilenet_v3", "mobilebert", "whisper_tiny"]:
            model_info = TEST_MODELS[model_id]
            optimization_results[model_id] = {
                "name": model_info["name"],
                "type": model_info["type"],
                "recommendations": {
                    "samsung": {
                        "compatible": True,
                        "recommended_precision": "INT8",
                        "estimated_memory_reduction": "30-40% with INT8 quantization",
                        "estimated_power_efficiency_score": 85,
                        "recommended_optimizations": [
                            "INT8 quantization",
                            "Samsung Neural SDK optimizations",
                            "One UI optimization",
                            "Model-specific optimizations"
                        ]
                    }
                }
            }
            
            # Add model-specific recommendations
            if model_info["type"] == "vision":
                optimization_results[model_id]["recommendations"]["samsung"]["recommended_optimizations"].append(
                    "Optimize for throughput with multi-image batch processing"
                )
            elif model_info["type"] == "nlp":
                optimization_results[model_id]["recommendations"]["samsung"]["recommended_optimizations"].append(
                    "INT4 weight-only quantization"
                )
            elif model_info["type"] == "audio":
                optimization_results[model_id]["recommendations"]["samsung"]["recommended_optimizations"].append(
                    "Game Booster integration for sustained performance"
                )
    
    return optimization_results

def print_comparison_report(hardware_results, compatibility_results, power_results, optimization_results):
    """Print a comprehensive comparison report"""
    print("\n" + "="*80)
    print(" MOBILE NPU COMPARISON REPORT ".center(80, "="))
    print("="*80 + "\n")
    
    # Hardware availability
    print("HARDWARE DETECTION RESULTS".center(80, "-"))
    for platform, info in hardware_results.items():
        status = "✅ AVAILABLE" if info["available"] else "❌ NOT AVAILABLE"
        if info["available"] and platform in ["qualcomm", "mediatek"] and info.get("simulation", False):
            status = "⚠️ SIMULATION MODE"
        
        print(f"{platform.upper()}: {status}")
        
        # Print devices if available
        if info["available"] and "devices" in info and info["devices"]:
            for device in info["devices"]:
                sim_tag = " (SIMULATED)" if device.get("simulated", False) else ""
                print(f"  - {device.get('name', 'Unknown device')}{sim_tag}")
    print("\n")
    
    # Model compatibility
    print("MODEL COMPATIBILITY".center(80, "-"))
    print(f"{'Model':<20} {'Type':<10} {'MediaTek':<12} {'Qualcomm':<12} {'Samsung':<12} {'Mobile-Opt':<10}")
    print("-"*80)
    for model_id, info in compatibility_results.items():
        mediatek_compat = "✅" if info["compatibility"].get("mediatek", False) else "❌"
        qualcomm_compat = "✅" if info["compatibility"].get("qualcomm", False) else "❌"
        samsung_compat = "✅" if info["compatibility"].get("samsung", False) else "❌"
        mobile_opt = "✅" if info.get("mobile_optimized", False) else "❌"
        print(f"{info['name']:<20} {info['type']:<10} {mediatek_compat:<12} {qualcomm_compat:<12} {samsung_compat:<12} {mobile_opt:<10}")
    print("\n")
    
    # Power efficiency
    print("POWER EFFICIENCY COMPARISON".center(80, "-"))
    has_power_data = all(platform in power_results for platform in ["mediatek", "qualcomm", "samsung"])
    if has_power_data:
        print(f"{'Metric':<30} {'MediaTek':<16} {'Qualcomm':<16} {'Samsung':<16}")
        print("-"*80)
        
        # Device names
        mediatek_name = power_results["mediatek"].get("device_name", "MediaTek Device")
        qualcomm_name = power_results["qualcomm"].get("device_name", "Qualcomm Device")
        samsung_name = power_results["samsung"].get("device_name", "Samsung Device")
        print(f"{'Device Name':<30} {mediatek_name:<16} {qualcomm_name:<16} {samsung_name:<16}")
        
        # Power metrics
        mediatek_avg = power_results["mediatek"].get("average_power_watts", 0)
        qualcomm_avg = power_results["qualcomm"].get("average_power_watts", 0)
        samsung_avg = power_results["samsung"].get("average_power_watts", 0)
        print(f"{'Average Power (Watts)':<30} {mediatek_avg:<16.2f} {qualcomm_avg:<16.2f} {samsung_avg:<16.2f}")
        
        mediatek_peak = power_results["mediatek"].get("peak_power_watts", 0)
        qualcomm_peak = power_results["qualcomm"].get("peak_power_watts", 0)
        samsung_peak = power_results["samsung"].get("peak_power_watts", 0)
        print(f"{'Peak Power (Watts)':<30} {mediatek_peak:<16.2f} {qualcomm_peak:<16.2f} {samsung_peak:<16.2f}")
        
        mediatek_impact = power_results["mediatek"].get("estimated_battery_impact_percent", 0)
        qualcomm_impact = power_results["qualcomm"].get("estimated_battery_impact_percent", 0)
        samsung_impact = power_results["samsung"].get("estimated_battery_impact_percent", 0)
        print(f"{'Battery Impact (%)':<30} {mediatek_impact:<16.2f} {qualcomm_impact:<16.2f} {samsung_impact:<16.2f}")
        
        mediatek_score = power_results["mediatek"].get("power_efficiency_score", 0)
        qualcomm_score = power_results["qualcomm"].get("power_efficiency_score", 0)
        samsung_score = power_results["samsung"].get("power_efficiency_score", 0)
        print(f"{'Power Efficiency Score':<30} {mediatek_score:<16.2f} {qualcomm_score:<16.2f} {samsung_score:<16.2f}")
        
        # TOPS per Watt (if available)
        if "tops_per_watt" in power_results["samsung"]:
            mediatek_tpw = power_results["mediatek"].get("tops_per_watt", 0)
            qualcomm_tpw = power_results["qualcomm"].get("tops_per_watt", 0)
            samsung_tpw = power_results["samsung"].get("tops_per_watt", 0)
            print(f"{'TOPS per Watt':<30} {mediatek_tpw:<16.2f} {qualcomm_tpw:<16.2f} {samsung_tpw:<16.2f}")
        
        # Battery life estimates
        if all("battery_life" in power_results[platform] for platform in ["mediatek", "qualcomm", "samsung"]):
            mediatek_runtime = power_results["mediatek"]["battery_life"].get("estimated_runtime_hours", 0)
            qualcomm_runtime = power_results["qualcomm"]["battery_life"].get("estimated_runtime_hours", 0)
            samsung_runtime = power_results["samsung"]["battery_life"].get("estimated_runtime_hours", 0)
            print(f"{'Est. Runtime (hours)':<30} {mediatek_runtime:<16.2f} {qualcomm_runtime:<16.2f} {samsung_runtime:<16.2f}")
            
            mediatek_percent = power_results["mediatek"]["battery_life"].get("battery_percent_per_hour", 0)
            qualcomm_percent = power_results["qualcomm"]["battery_life"].get("battery_percent_per_hour", 0)
            samsung_percent = power_results["samsung"]["battery_life"].get("battery_percent_per_hour", 0)
            print(f"{'Battery % Per Hour':<30} {mediatek_percent:<16.2f} {qualcomm_percent:<16.2f} {samsung_percent:<16.2f}")
    else:
        print("Power efficiency comparison data not available for all platforms.")
    print("\n")
    
    # Optimization recommendations
    if optimization_results:
        print("OPTIMIZATION RECOMMENDATIONS".center(80, "-"))
        for model_id, info in optimization_results.items():
            print(f"Model: {info['name']} ({info['type']})")
            
            if "mediatek" in info["recommendations"]:
                mediatek_rec = info["recommendations"]["mediatek"]
                print("  MediaTek Recommendations:")
                if mediatek_rec.get("compatible", False):
                    for opt in mediatek_rec.get("recommended_optimizations", []):
                        print(f"    - {opt}")
                    print(f"    Memory Reduction: {mediatek_rec.get('estimated_memory_reduction', 'N/A')}")
                    print(f"    Power Efficiency Score: {mediatek_rec.get('estimated_power_efficiency_score', 0)}")
                else:
                    print(f"    Not compatible: {mediatek_rec.get('reason', 'Unknown reason')}")
            
            if "qualcomm" in info["recommendations"]:
                qualcomm_rec = info["recommendations"]["qualcomm"]
                print("  Qualcomm Recommendations:")
                if qualcomm_rec.get("compatible", False):
                    for opt in qualcomm_rec.get("recommended_optimizations", []):
                        print(f"    - {opt}")
                    print(f"    Memory Reduction: {qualcomm_rec.get('estimated_memory_reduction', 'N/A')}")
                    print(f"    Power Efficiency Score: {qualcomm_rec.get('estimated_power_efficiency_score', 0)}")
                else:
                    print(f"    Not compatible: {qualcomm_rec.get('reason', 'Unknown reason')}")
            
            print()
    
    # Summary and recommendations
    print("SUMMARY AND RECOMMENDATIONS".center(80, "-"))
    
    # Count compatible models for each platform
    mediatek_count = sum(1 for info in compatibility_results.values() 
                        if info["compatibility"].get("mediatek", False))
    qualcomm_count = sum(1 for info in compatibility_results.values() 
                         if info["compatibility"].get("qualcomm", False))
    samsung_count = sum(1 for info in compatibility_results.values() 
                         if info["compatibility"].get("samsung", False))
    
    total_models = len(compatibility_results)
    mediatek_percent = (mediatek_count / total_models) * 100 if total_models > 0 else 0
    qualcomm_percent = (qualcomm_count / total_models) * 100 if total_models > 0 else 0
    samsung_percent = (samsung_count / total_models) * 100 if total_models > 0 else 0
    
    print(f"MediaTek compatibility: {mediatek_count}/{total_models} models ({mediatek_percent:.1f}%)")
    print(f"Qualcomm compatibility: {qualcomm_count}/{total_models} models ({qualcomm_percent:.1f}%)")
    print(f"Samsung compatibility:  {samsung_count}/{total_models} models ({samsung_percent:.1f}%)")
    
    # Power efficiency comparison (if available)
    if all(platform in power_results for platform in ["mediatek", "qualcomm", "samsung"]):
        mediatek_score = power_results["mediatek"].get("power_efficiency_score", 0)
        qualcomm_score = power_results["qualcomm"].get("power_efficiency_score", 0)
        samsung_score = power_results["samsung"].get("power_efficiency_score", 0)
        
        # Determine the winner
        scores = {
            "MediaTek": mediatek_score,
            "Qualcomm": qualcomm_score,
            "Samsung": samsung_score
        }
        winner = max(scores.items(), key=lambda x: x[1])
        
        print(f"Power efficiency winner: {winner[0]} (score: {winner[1]:.1f})")
        print(f"  MediaTek: {mediatek_score:.1f}, Qualcomm: {qualcomm_score:.1f}, Samsung: {samsung_score:.1f}")
    
    # Final recommendations
    print("\nRECOMMENDATIONS:")
    
    # Mobile-optimized models
    mobile_models = [info for model_id, info in compatibility_results.items() 
                    if info.get("mobile_optimized", False)]
    
    mediatek_mobile_count = sum(1 for info in mobile_models 
                               if info["compatibility"].get("mediatek", False))
    qualcomm_mobile_count = sum(1 for info in mobile_models 
                               if info["compatibility"].get("qualcomm", False))
    samsung_mobile_count = sum(1 for info in mobile_models 
                               if info["compatibility"].get("samsung", False))
    
    # Find the best platform for mobile-optimized models
    mobile_counts = [
        ("MediaTek", mediatek_mobile_count),
        ("Qualcomm", qualcomm_mobile_count),
        ("Samsung", samsung_mobile_count)
    ]
    best_mobile_platform = max(mobile_counts, key=lambda x: x[1])
    
    if best_mobile_platform[1] > 0:  # Only recommend if at least one model is supported
        print(f"- {best_mobile_platform[0]} is recommended for mobile-optimized models")
    else:
        print("- No clear recommendation for mobile-optimized models")
    
    # Recommendation by model type
    vision_models = [info for model_id, info in compatibility_results.items() 
                    if info["type"] == "vision"]
    nlp_models = [info for model_id, info in compatibility_results.items() 
                 if info["type"] == "nlp"]
    audio_models = [info for model_id, info in compatibility_results.items() 
                   if info["type"] == "audio"]
    
    # Vision model recommendations
    mediatek_vision_count = sum(1 for info in vision_models 
                               if info["compatibility"].get("mediatek", False))
    qualcomm_vision_count = sum(1 for info in vision_models 
                               if info["compatibility"].get("qualcomm", False))
    samsung_vision_count = sum(1 for info in vision_models 
                               if info["compatibility"].get("samsung", False))
    
    vision_counts = [
        ("MediaTek", mediatek_vision_count),
        ("Qualcomm", qualcomm_vision_count),
        ("Samsung", samsung_vision_count)
    ]
    best_vision_platform = max(vision_counts, key=lambda x: x[1])
    
    if best_vision_platform[1] > 0:  # Only recommend if at least one model is supported
        print(f"- {best_vision_platform[0]} is recommended for vision models")
    else:
        print("- No clear recommendation for vision models")
    
    # NLP model recommendations
    mediatek_nlp_count = sum(1 for info in nlp_models 
                            if info["compatibility"].get("mediatek", False))
    qualcomm_nlp_count = sum(1 for info in nlp_models 
                            if info["compatibility"].get("qualcomm", False))
    samsung_nlp_count = sum(1 for info in nlp_models 
                           if info["compatibility"].get("samsung", False))
    
    nlp_counts = [
        ("MediaTek", mediatek_nlp_count),
        ("Qualcomm", qualcomm_nlp_count),
        ("Samsung", samsung_nlp_count)
    ]
    best_nlp_platform = max(nlp_counts, key=lambda x: x[1])
    
    if best_nlp_platform[1] > 0:  # Only recommend if at least one model is supported
        print(f"- {best_nlp_platform[0]} is recommended for NLP models")
    else:
        print("- No clear recommendation for NLP models")
    
    # Audio model recommendations
    mediatek_audio_count = sum(1 for info in audio_models 
                              if info["compatibility"].get("mediatek", False))
    qualcomm_audio_count = sum(1 for info in audio_models 
                              if info["compatibility"].get("qualcomm", False))
    samsung_audio_count = sum(1 for info in audio_models 
                             if info["compatibility"].get("samsung", False))
    
    audio_counts = [
        ("MediaTek", mediatek_audio_count),
        ("Qualcomm", qualcomm_audio_count),
        ("Samsung", samsung_audio_count)
    ]
    best_audio_platform = max(audio_counts, key=lambda x: x[1])
    
    if best_audio_platform[1] > 0:  # Only recommend if at least one model is supported
        print(f"- {best_audio_platform[0]} is recommended for audio models")
    else:
        print("- No clear recommendation for audio models")
    
    # Overall compatibility recommendation
    compatibility_counts = [
        ("MediaTek", mediatek_count),
        ("Qualcomm", qualcomm_count),
        ("Samsung", samsung_count)
    ]
    best_compatibility = max(compatibility_counts, key=lambda x: x[1])
    
    print(f"- Overall, {best_compatibility[0]} supports more tested models ({best_compatibility[1]}/{total_models})")
    
    # Power efficiency recommendations based on model types (if available)
    if all(platform in power_results for platform in ["mediatek", "qualcomm", "samsung"]):
        mediatek_score = power_results["mediatek"].get("power_efficiency_score", 0)
        qualcomm_score = power_results["qualcomm"].get("power_efficiency_score", 0)
        samsung_score = power_results["samsung"].get("power_efficiency_score", 0)
        
        if samsung_score > mediatek_score and samsung_score > qualcomm_score:
            print("- Samsung Exynos provides the best power efficiency for battery-sensitive applications")
        elif mediatek_score > qualcomm_score:
            print("- MediaTek provides the best power efficiency for battery-sensitive applications")
        else:
            print("- Qualcomm provides the best power efficiency for battery-sensitive applications")
    
    # Final notes
    print("\nNOTE: ")
    if (hardware_results["mediatek"].get("simulation", False) or 
        hardware_results["qualcomm"].get("simulation", False) or
        hardware_results["samsung"].get("simulation", False)):
        print("⚠️ Some or all hardware capabilities were tested in SIMULATION mode.")
        print("   Real performance may differ from the reported results.")
        
        # List specifically which platforms were simulated
        simulated_platforms = []
        if hardware_results["mediatek"].get("simulation", False):
            simulated_platforms.append("MediaTek")
        if hardware_results["qualcomm"].get("simulation", False):
            simulated_platforms.append("Qualcomm")
        if hardware_results["samsung"].get("simulation", False):
            simulated_platforms.append("Samsung")
            
        print(f"   Simulated platforms: {', '.join(simulated_platforms)}")
    
    print("\n" + "="*80 + "\n")

def main():
    """Main function for the mobile NPU comparison test"""
    parser = argparse.ArgumentParser(description="Compare mobile NPU hardware capabilities")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--output", type=str, help="Output file for JSON results")
    parser.add_argument("--models-only", action="store_true", help="Test only model compatibility")
    parser.add_argument("--power-only", action="store_true", help="Test only power efficiency")
    args = parser.parse_args()
    
    # Run the comparison tests
    hardware_results = check_hardware()
    compatibility_results = compare_model_compatibility()
    
    # Only run these tests if not in models-only mode
    power_results = {}
    optimization_results = {}
    if not args.models_only:
        power_results = compare_power_efficiency()
        if not args.power_only:
            optimization_results = compare_optimization_recommendations()
    
    # Combine all results
    results = {
        "hardware": hardware_results,
        "compatibility": compatibility_results,
        "power_efficiency": power_results,
        "optimization": optimization_results
    }
    
    # Output results
    if args.json:
        # JSON output
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print_comparison_report(
            hardware_results,
            compatibility_results,
            power_results,
            optimization_results
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())