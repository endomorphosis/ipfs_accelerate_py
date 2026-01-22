#!/usr/bin/env python3
"""
Check template files for mock detection implementation status.

This script examines all template files to verify the presence
of the key components of the mock detection system:
1. Environment variable control
2. Mock implementation checks
3. Detection logic
4. Visual indicators
5. Metadata enrichment

Usage:
    python check_template_mock_status.py [--detailed]
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def check_template(template_path, detailed=False):
    """
    Check a template file for mock detection implementation.
    
    Args:
        template_path: Path to the template file
        detailed: If True, print detailed report
        
    Returns:
        dict: Results of the check
    """
    results = {
        "file": os.path.basename(template_path),
        "components": {
            "env_vars": {
                "MOCK_TORCH": False,
                "MOCK_TRANSFORMERS": False, 
                "MOCK_TOKENIZERS": False,
                "MOCK_SENTENCEPIECE": False
            },
            "mock_checks": {
                "torch": False,
                "transformers": False,
                "tokenizers": False,
                "sentencepiece": False
            },
            "detection_logic": {
                "using_real_inference": False,
                "using_mocks": False
            },
            "visual_indicators": {
                "real_inference": False,
                "mock_objects": False
            },
            "color_codes": {
                "GREEN": False,
                "BLUE": False,
                "RESET": False
            },
            "metadata": {
                "has_transformers": False,
                "has_torch": False,
                "has_tokenizers": False,
                "has_sentencepiece": False,
                "using_real_inference": False,
                "using_mocks": False,
                "test_type": False
            }
        },
        "complete": False
    }
    
    try:
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check for environment variables
        if "MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'" in content:
            results["components"]["env_vars"]["MOCK_TORCH"] = True
        if "MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'" in content:
            results["components"]["env_vars"]["MOCK_TRANSFORMERS"] = True
        if "MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'" in content:
            results["components"]["env_vars"]["MOCK_TOKENIZERS"] = True
        if "MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'" in content:
            results["components"]["env_vars"]["MOCK_SENTENCEPIECE"] = True
        
        # Check for mock dependency checks
        if "if MOCK_TORCH:" in content:
            results["components"]["mock_checks"]["torch"] = True
        if "if MOCK_TRANSFORMERS:" in content:
            results["components"]["mock_checks"]["transformers"] = True
        if "if MOCK_TOKENIZERS:" in content:
            results["components"]["mock_checks"]["tokenizers"] = True
        if "if MOCK_SENTENCEPIECE:" in content:
            results["components"]["mock_checks"]["sentencepiece"] = True
        
        # Check for detection logic
        if "using_real_inference = HAS_TRANSFORMERS and HAS_TORCH" in content:
            results["components"]["detection_logic"]["using_real_inference"] = True
        if "using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE" in content:
            results["components"]["detection_logic"]["using_mocks"] = True
        
        # Check for visual indicators
        if "🚀 Using REAL INFERENCE with actual models" in content:
            results["components"]["visual_indicators"]["real_inference"] = True
        if "🔷 Using MOCK OBJECTS for CI/CD testing only" in content:
            results["components"]["visual_indicators"]["mock_objects"] = True
        
        # Check for color codes
        if 'GREEN = "\\033[32m"' in content or "GREEN = '\\033[32m'" in content:
            results["components"]["color_codes"]["GREEN"] = True
        if 'BLUE = "\\033[34m"' in content or "BLUE = '\\033[34m'" in content:
            results["components"]["color_codes"]["BLUE"] = True
        if 'RESET = "\\033[0m"' in content or "RESET = '\\033[0m'" in content:
            results["components"]["color_codes"]["RESET"] = True
        
        # Check for metadata
        if '"has_transformers": HAS_TRANSFORMERS' in content:
            results["components"]["metadata"]["has_transformers"] = True
        if '"has_torch": HAS_TORCH' in content:
            results["components"]["metadata"]["has_torch"] = True
        if '"has_tokenizers": HAS_TOKENIZERS' in content:
            results["components"]["metadata"]["has_tokenizers"] = True
        if '"has_sentencepiece": HAS_SENTENCEPIECE' in content:
            results["components"]["metadata"]["has_sentencepiece"] = True
        if '"using_real_inference": using_real_inference' in content:
            results["components"]["metadata"]["using_real_inference"] = True
        if '"using_mocks": using_mocks' in content:
            results["components"]["metadata"]["using_mocks"] = True
        if '"test_type": "REAL INFERENCE"' in content or '"test_type": "MOCK OBJECTS' in content:
            results["components"]["metadata"]["test_type"] = True
        
        # Determine overall status
        all_components_complete = all([
            all(results["components"]["env_vars"].values()),
            all(results["components"]["mock_checks"].values()),
            all(results["components"]["detection_logic"].values()),
            all(results["components"]["visual_indicators"].values()),
            all(results["components"]["color_codes"].values()),
            all(results["components"]["metadata"].values())
        ])
        
        # Make adjustment for vision/speech templates which might not need tokenizers/sentencepiece
        is_vision_template = "vision_template.py" in template_path or "speech_template.py" in template_path
        
        if is_vision_template:
            # Don't require tokenizers and sentencepiece for vision templates
            vision_components_complete = all([
                all(results["components"]["env_vars"].values()[:2]),  # Only MOCK_TORCH and MOCK_TRANSFORMERS
                all(results["components"]["mock_checks"].values()[:2]),  # Only torch and transformers
                all(results["components"]["detection_logic"].values()),
                all(results["components"]["visual_indicators"].values()),
                all(results["components"]["color_codes"].values()),
                all([v for k, v in results["components"]["metadata"].items() if not k.startswith("has_t") or k == "has_transformers" or k == "has_torch"])
            ])
            
            results["complete"] = vision_components_complete
        else:
            results["complete"] = all_components_complete
        
        # Print detailed report if requested
        if detailed:
            print(f"\n{BLUE}===== Template: {os.path.basename(template_path)} ====={RESET}")
            
            # Environment Variables
            print(f"\nEnvironment Variables:")
            for var, present in results["components"]["env_vars"].items():
                status = f"{GREEN}✓{RESET}" if present else f"{RED}✗{RESET}"
                print(f"  {status} {var}")
            
            # Mock Checks
            print(f"\nMock Checks:")
            for dep, present in results["components"]["mock_checks"].items():
                status = f"{GREEN}✓{RESET}" if present else f"{RED}✗{RESET}"
                print(f"  {status} {dep}")
            
            # Detection Logic
            print(f"\nDetection Logic:")
            for logic, present in results["components"]["detection_logic"].items():
                status = f"{GREEN}✓{RESET}" if present else f"{RED}✗{RESET}"
                print(f"  {status} {logic}")
            
            # Visual Indicators
            print(f"\nVisual Indicators:")
            for indicator, present in results["components"]["visual_indicators"].items():
                status = f"{GREEN}✓{RESET}" if present else f"{RED}✗{RESET}"
                print(f"  {status} {indicator}")
            
            # Color Codes
            print(f"\nColor Codes:")
            for code, present in results["components"]["color_codes"].items():
                status = f"{GREEN}✓{RESET}" if present else f"{RED}✗{RESET}"
                print(f"  {status} {code}")
            
            # Metadata
            print(f"\nMetadata:")
            for field, present in results["components"]["metadata"].items():
                status = f"{GREEN}✓{RESET}" if present else f"{RED}✗{RESET}"
                print(f"  {status} {field}")
            
            # Overall Status
            status = f"{GREEN}COMPLETE{RESET}" if results["complete"] else f"{RED}INCOMPLETE{RESET}"
            print(f"\nOverall Status: {status}")
        
        return results
    
    except Exception as e:
        print(f"{RED}Error checking {template_path}: {e}{RESET}")
        return {
            "file": os.path.basename(template_path),
            "error": str(e),
            "complete": False
        }

def main():
    parser = argparse.ArgumentParser(description="Check template files for mock detection implementation status")
    parser.add_argument("--detailed", action="store_true", help="Print detailed report for each template")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--directory", type=str, default="templates", help="Directory containing template files")
    
    args = parser.parse_args()
    
    # Find template files
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.directory)
    template_files = [os.path.join(template_dir, f) for f in os.listdir(template_dir) if f.endswith("_template.py")]
    
    if not template_files:
        print(f"{RED}No template files found in {template_dir}{RESET}")
        return 1
    
    print(f"Found {len(template_files)} template files")
    
    # Check each template
    results = []
    complete_count = 0
    incomplete_count = 0
    
    for template_file in template_files:
        result = check_template(template_file, args.detailed)
        results.append(result)
        
        if result.get("complete", False):
            complete_count += 1
        else:
            incomplete_count += 1
    
    # Output JSON if requested
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    
    # Print summary
    print(f"\n{BLUE}===== Mock Detection Implementation Summary ====={RESET}")
    print(f"Total templates: {len(results)}")
    print(f"Complete: {complete_count}")
    print(f"Incomplete: {incomplete_count}")
    
    # List incomplete templates
    if incomplete_count > 0:
        print(f"\n{YELLOW}Incomplete templates:{RESET}")
        for result in results:
            if not result.get("complete", False):
                print(f"  - {result['file']}")
    
    return 0 if incomplete_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())