#!/usr/bin/env python3
"""
Example Template Generator using the Enhanced Template System
This script demonstrates how to use the enhanced template system.
"""

import os
import sys
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import duckdb
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.error("DuckDB not available. This script requires DuckDB.")
    sys.exit(1)

# Default database path
DEFAULT_DB_PATH = "./template_db.duckdb"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Example template generator using the enhanced template system"
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model name (e.g. bert-base-uncased)"
    )
    parser.add_argument(
        "--template-type", "-t", type=str, default="test",
        choices=["test", "benchmark", "skill", "helper"],
        help="Template type (default: test)"
    )
    parser.add_argument(
        "--hardware", type=str, default=None,
        help="Hardware platform (if none specified, a generic template will be used)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output file path (if not specified, output to console)"
    )
    parser.add_argument(
        "--db-path", type=str, default=DEFAULT_DB_PATH,
        help=f"Path to template database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--detect-hardware", action="store_true",
        help="Detect available hardware on the system"
    )
    return parser.parse_args()

def get_model_type(model_name: str) -> str:
    """Determine model type from model name"""
    model_name_lower = model_name.lower()
    
    # Check for specific model families
    if "bert" in model_name_lower:
        return "bert"
    elif "t5" in model_name_lower:
        return "t5"
    elif "llama" in model_name_lower or "opt" in model_name_lower:
        return "llama"
    elif "vit" in model_name_lower:
        return "vit"
    elif "clip" in model_name_lower and "x" not in model_name_lower:
        return "clip"
    elif "whisper" in model_name_lower:
        return "whisper"
    elif "wav2vec" in model_name_lower:
        return "wav2vec2"
    elif "clap" in model_name_lower:
        return "clap"
    elif "llava" in model_name_lower:
        return "llava"
    elif "xclip" in model_name_lower:
        return "xclip"
    elif "qwen" in model_name_lower:
        return "qwen"
    elif "detr" in model_name_lower:
        return "detr"
    else:
        return "default"

def detect_hardware() -> Dict[str, bool]:
    """Detect available hardware platforms on the system"""
    hardware_support = {
        "cpu": True,  # CPU is always available
        "cuda": False,
        "rocm": False,
        "mps": False,
        "openvino": False,
        "qualcomm": False,
        "samsung": False,
        "webnn": False,
        "webgpu": False
    }
    
    # Check for CUDA
    try:
        import torch
        hardware_support["cuda"] = torch.cuda.is_available()
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch, 'mps') and hasattr(torch.backends, 'mps'):
            hardware_support["mps"] = torch.backends.mps.is_available()
    except ImportError:
        pass
    
    # Check for OpenVINO
    try:
        import openvino
        hardware_support["openvino"] = True
    except ImportError:
        pass
    
    # Future: Add checks for other hardware platforms
    
    return hardware_support

def get_template_from_db(db_path: str, model_type: str, template_type: str, hardware_platform: Optional[str] = None) -> Optional[str]:
    """Get a template from the database"""
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB not available, cannot get template")
        return None
    
    try:
        conn = duckdb.connect(db_path)
        
        # Query for hardware-specific template first if hardware_platform provided
        if hardware_platform:
            result = conn.execute("""
            SELECT template FROM templates
            WHERE model_type = ? AND template_type = ? AND hardware_platform = ?
            """, [model_type, template_type, hardware_platform]).fetchone()
            
            if result:
                conn.close()
                return result[0]
        
        # Fall back to generic template
        result = conn.execute("""
        SELECT template FROM templates
        WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
        """, [model_type, template_type]).fetchone()
        
        if result:
            conn.close()
            return result[0]
        
        # Check if model has a parent template
        result = conn.execute("""
        SELECT parent_template FROM templates
        WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
        """, [model_type, template_type]).fetchone()
        
        if result and result[0]:
            parent_type = result[0]
            logger.info(f"Using parent template: {parent_type}")
            
            # Query parent template
            result = conn.execute("""
            SELECT template FROM templates
            WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
            """, [parent_type, template_type]).fetchone()
            
            if result:
                conn.close()
                return result[0]
        
        # Fall back to default template type
        result = conn.execute("""
        SELECT template FROM templates
        WHERE model_type = 'default' AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
        """, [template_type]).fetchone()
        
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        return None

def prepare_template_context(model_name: str, hardware_platform: Optional[str] = None) -> Dict[str, Any]:
    """Prepare context for template rendering"""
    import re
    
    # Normalize model name for class names
    normalized_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name).title()
    
    # Hardware detection
    hardware = detect_hardware()
    
    # Prepare context
    context = {
        "model_name": model_name,
        "normalized_name": normalized_name,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "has_cuda": str(hardware.get("cuda", False)),
        "has_rocm": str(hardware.get("rocm", False)),
        "has_mps": str(hardware.get("mps", False)),
        "has_openvino": str(hardware.get("openvino", False)),
        "has_webnn": str(hardware.get("webnn", False)),
        "has_webgpu": str(hardware.get("webgpu", False)),
    }
    
    # Determine best hardware platform
    if hardware_platform:
        context["best_hardware"] = hardware_platform
    elif hardware.get("cuda", False):
        context["best_hardware"] = "cuda"
    elif hardware.get("mps", False):
        context["best_hardware"] = "mps"
    elif hardware.get("openvino", False):
        context["best_hardware"] = "openvino"
    else:
        context["best_hardware"] = "cpu"
    
    # Set torch device based on best hardware
    if context["best_hardware"] == "cuda":
        context["torch_device"] = "cuda"
    elif context["best_hardware"] == "mps":
        context["torch_device"] = "mps"
    else:
        context["torch_device"] = "cpu"
    
    return context

def render_template(template: str, context: Dict[str, Any]) -> str:
    """Render template with context variables"""
    try:
        # Try to use the enhanced placeholder helpers if available
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from template_utilities.placeholder_helpers import render_template as enhanced_render
        result = enhanced_render(template, context)
        logger.info("Using enhanced template rendering")
    except ImportError:
        # Fallback to basic string formatting
        logger.info("Using basic template rendering")
        try:
            result = template.format(**context)
        except KeyError as e:
            logger.error(f"Missing placeholder in template: {e}")
            # Add missing placeholder with a default value
            context[str(e).strip("'")] = f"<<MISSING:{str(e).strip('')}>>"
            result = template.format(**context)
    
    return result

def main():
    """Main function"""
    args = parse_args()
    
    # Detect hardware if requested
    if args.detect_hardware:
        hardware = detect_hardware()
        print("\nDetected Hardware:")
        print("-" * 30)
        for platform, available in hardware.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"{platform:<10}: {status}")
        return 0
    
    # Determine model type from model name
    model_type = get_model_type(args.model)
    logger.info(f"Detected model type: {model_type}")
    
    # Get template from database
    template = get_template_from_db(args.db_path, model_type, args.template_type, args.hardware)
    
    if not template:
        logger.error(f"Template not found for {model_type}/{args.template_type}/{args.hardware or 'generic'}")
        return 1
    
    # Prepare context for template rendering
    context = prepare_template_context(args.model, args.hardware)
    
    # Render template
    rendered_template = render_template(template, context)
    
    # Output rendered template
    if args.output:
        with open(args.output, 'w') as f:
            f.write(rendered_template)
        logger.info(f"Template rendered to {args.output}")
    else:
        print("\nRendered Template:")
        print("=" * 80)
        print(rendered_template)
        print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())