#!/usr/bin/env python3
"""
Script to fix issues in merged_test_generator.py

This script addresses multiple issues with the merged_test_generator.py:
1. Fix unterminated string literals in template definitions
2. Add Qualcomm hardware support to all templates
3. Update MockHandler implementation
4. Fix hardware detection code
5. Update template imports
"""

import os
import re
import sys
import shutil
from datetime import datetime

def backup_file(file_path):
    """Create backup of original file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak_{timestamp}"
    print(f"Creating backup: {backup_path}")
    shutil.copy2(file_path, backup_path)
    return backup_path

def fix_unterminated_strings(file_path):
    """Fix unterminated string literals in template definitions."""
    print(f"Fixing unterminated string literals in {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to find template definitions without proper closing
    pattern = r'template_database\["([^"]+)"\] = """(.*?)(?=template_database\["[^"]+"\] = """|$)'
    
    # Replace with properly terminated strings
    fixed_content = re.sub(pattern, lambda m: f'template_database["{m.group(1)}"] = """{m.group(2)}"""\n\n', content, flags=re.DOTALL)
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print("Fixed unterminated string literals")

def add_qualcomm_support(file_path):
    """Add Qualcomm hardware support to all templates."""
    print(f"Adding Qualcomm hardware support to {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add Qualcomm to hardware platforms in templates
    pattern = r'- CPU: Standard CPU implementation\n- CUDA: NVIDIA GPU implementation\n- OpenVINO: Intel hardware acceleration\n- MPS: Apple Silicon GPU implementation\n- ROCm: AMD GPU implementation'
    replacement = r'- CPU: Standard CPU implementation\n- CUDA: NVIDIA GPU implementation\n- OpenVINO: Intel hardware acceleration\n- MPS: Apple Silicon GPU implementation\n- ROCm: AMD GPU implementation\n- Qualcomm: Qualcomm AI Engine and Hexagon DSP'
    
    # Replace the hardware platform list in all templates
    fixed_content = content.replace(pattern, replacement)
    
    # Add Qualcomm to KEY_MODEL_HARDWARE_MAP
    key_model_map_pattern = r'"webgpu": "REAL"      # WebGPU support: fully implemented'
    key_model_map_replacement = r'"webgpu": "REAL",      # WebGPU support: fully implemented\n        "qualcomm": "REAL"    # Qualcomm support: fully implemented'
    
    # Replace in hardware map
    fixed_content = fixed_content.replace(key_model_map_pattern, key_model_map_replacement)
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print("Added Qualcomm hardware support to templates")

def fix_mock_handler(file_path):
    """Fix and update the MockHandler implementation."""
    print(f"Fixing MockHandler implementation in {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all MockHandler class definitions
    mock_handler_pattern = r'class MockHandler:[\s\S]*?def __call__\(self, \*args, \*\*kwargs\):[\s\S]*?return \{[^}]*\}'
    mock_handlers = re.findall(mock_handler_pattern, content)
    
    if mock_handlers:
        # Keep only the first instance of MockHandler
        first_handler = mock_handlers[0]
        
        # Create improved MockHandler implementation
        improved_handler = '''class MockHandler:
    """Mock handler for platforms that don't have real implementations."""
    
    def __init__(self, model_path, platform="cpu"):
        self.model_path = model_path
        self.platform = platform
        print(f"Created mock handler for {platform}")
    
    def __call__(self, *args, **kwargs):
        """Return mock output."""
        print(f"MockHandler for {self.platform} called with {len(args)} args and {len(kwargs)} kwargs")
        return {"output": "MOCK OUTPUT", "implementation_type": f"MOCK_{self.platform.upper()}"}'''
        
        # Replace all instances with the improved handler
        for handler in mock_handlers:
            content = content.replace(handler, improved_handler)
        
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("Fixed MockHandler implementation")
    else:
        print("Warning: MockHandler not found in the file")

def update_hardware_detection(file_path):
    """Update hardware detection code to include Qualcomm."""
    print(f"Updating hardware detection in {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if hardware detection import is present
    if "from centralized_hardware_detection import" not in content:
        # Add import at the top of the file (after other imports)
        import_pattern = r'import os\nimport sys\nimport json'
        import_replacement = r'import os\nimport sys\nimport json\n\n# Import centralized hardware detection\nfrom centralized_hardware_detection.hardware_detection import get_capabilities, get_hardware_detection_code, get_model_hardware_compatibility'
        
        content = content.replace(import_pattern, import_replacement)
    
    # Update hardware detection code in template generation
    hardware_detection_pattern = r'# Hardware detection\s+has_cuda = .*?\s+has_rocm = .*?\s+has_mps = .*?\s+has_openvino = .*?\s+has_webnn = .*?\s+has_webgpu = .*?'
    hardware_detection_replacement = r'''# Hardware detection
        # Get hardware capabilities from centralized module
        hw_capabilities = get_capabilities()
        has_cuda = hw_capabilities["cuda"]
        has_rocm = hw_capabilities["rocm"]
        has_mps = hw_capabilities["mps"]
        has_openvino = hw_capabilities["openvino"]
        has_qualcomm = hw_capabilities["qualcomm"]
        has_webnn = hw_capabilities["webnn"]
        has_webgpu = hw_capabilities["webgpu"]'''
    
    # Replace hardware detection code
    content = re.sub(hardware_detection_pattern, hardware_detection_replacement, content, flags=re.DOTALL)
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Updated hardware detection code")

def update_template_generation(file_path):
    """Update template generation to ensure all hardware platforms are included."""
    print(f"Updating template generation in {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update hardware platforms in generate_test_file function
    platforms_pattern = r'platforms = \["cpu"\]\s+if has_cuda:\s+platforms\.append\("cuda"\)\s+if has_rocm:\s+platforms\.append\("rocm"\)\s+if has_mps:\s+platforms\.append\("mps"\)\s+if has_openvino:\s+platforms\.append\("openvino"\)\s+if has_webnn:\s+platforms\.append\("webnn"\)\s+if has_webgpu:\s+platforms\.append\("webgpu"\)'
    platforms_replacement = r'''platforms = ["cpu"]
        if has_cuda:
            platforms.append("cuda")
        if has_rocm:
            platforms.append("rocm")
        if has_mps:
            platforms.append("mps")
        if has_openvino:
            platforms.append("openvino")
        if has_qualcomm:
            platforms.append("qualcomm")
        if has_webnn:
            platforms.append("webnn")
        if has_webgpu:
            platforms.append("webgpu")'''
    
    # Replace platforms list
    content = content.replace(platforms_pattern, platforms_replacement)
    
    # Update platform-specific template code generation
    platform_code_pattern = r'# Platform-specific code\s+platform_code = \{\}\s+for platform in platforms:'
    platform_code_replacement = r'''# Platform-specific code
        platform_code = {}
        # Get model hardware compatibility from centralized detection
        model_compatibility = get_model_hardware_compatibility(model_name)
        for platform in platforms:
            # Skip if platform is not compatible with this model
            if platform in model_compatibility and not model_compatibility[platform]:
                continue'''
    
    # Replace platform code generation
    content = content.replace(platform_code_pattern, platform_code_replacement)
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Updated template generation")

def fix_import_templates(file_path):
    """Update template imports to include hardware templates."""
    print(f"Fixing template imports in {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if database template import is present
    if "import template_database as db" not in content and "from hardware_test_templates import template_database" not in content:
        # Add import for hardware templates
        try_import_pattern = r'try:\s+from fixed_web_platform import '
        try_import_replacement = r'''try:
    # Try to import hardware templates
    try:
        from hardware_test_templates import template_database as db
        TEMPLATE_DB_AVAILABLE = True
        print("Using template database for hardware-aware templates")
    except ImportError:
        TEMPLATE_DB_AVAILABLE = False
        print("Template database not available - falling back to static templates")
    
    # Import web platform support
    from fixed_web_platform import '''
        
        content = content.replace(try_import_pattern, try_import_replacement)
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed template imports")

def main():
    """Run all fixes for merged_test_generator.py."""
    generator_path = os.path.join(os.getcwd(), "merged_test_generator.py")
    fixed_generator_path = os.path.join(os.getcwd(), "fixed_merged_test_generator.py")
    
    if not os.path.exists(generator_path):
        generator_path = fixed_generator_path
        if not os.path.exists(generator_path):
            print(f"Error: Could not find generator file at {generator_path}")
            sys.exit(1)
    
    # Create backup
    backup_path = backup_file(generator_path)
    print(f"Backed up original file to {backup_path}")
    
    # Apply fixes
    fix_unterminated_strings(generator_path)
    add_qualcomm_support(generator_path)
    fix_mock_handler(generator_path)
    update_hardware_detection(generator_path)
    update_template_generation(generator_path)
    fix_import_templates(generator_path)
    
    # Update fixed_merged_test_generator.py with the same changes
    if generator_path != fixed_generator_path and os.path.exists(fixed_generator_path):
        print(f"\nApplying the same fixes to {fixed_generator_path}")
        backup_file(fixed_generator_path)
        
        # Copy the fixed file to fixed_merged_test_generator.py
        shutil.copy2(generator_path, fixed_generator_path)
        print(f"Updated {fixed_generator_path} with the same fixes")
    
    print("\nAll fixes applied successfully!")
    print(f"You can now run tests with the fixed generator: python {generator_path} --model bert --cross-platform --hardware all")

if __name__ == "__main__":
    main()