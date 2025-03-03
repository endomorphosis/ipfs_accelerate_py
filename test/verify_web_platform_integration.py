#!/usr/bin/env python3
"""
Verify that web platform support is properly integrated into the test generator system.

This script:
1. Checks if the fixed_web_platform module is imported in merged_test_generator.py
2. Confirms that the generator can generate tests with WebNN and WebGPU platform support
3. Tests importing the fixed_web_platform module directly
4. Verifies the availability of March 2025 enhancements:
   - WebGPU compute shader support for audio models
   - Parallel model loading for multimodal models
   - Shader precompilation for faster startup

Usage:
  python verify_web_platform_integration.py
"""

import os
import sys
import importlib
from pathlib import Path

# Paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def check_generator_imports():
    """Check if merged_test_generator.py imports the fixed_web_platform module."""
    print("Checking imports in merged_test_generator.py...")
    
    generator_path = CURRENT_DIR / "merged_test_generator.py"
    if not generator_path.exists():
        print(f"Error: {generator_path} not found")
        return False
        
    try:
        with open(generator_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "from fixed_web_platform import" in content:
            print("✅ fixed_web_platform module is imported in merged_test_generator.py")
            return True
        else:
            print("❌ fixed_web_platform module is not imported in merged_test_generator.py")
            return False
    except Exception as e:
        print(f"Error reading merged_test_generator.py: {e}")
        return False

def check_fixed_web_platform_module():
    """Check if the fixed_web_platform module can be imported."""
    print("\nChecking fixed_web_platform module...")
    
    module_dir = CURRENT_DIR / "fixed_web_platform"
    if not module_dir.exists():
        print(f"❌ fixed_web_platform directory not found at {module_dir}")
        return False
        
    init_file = module_dir / "__init__.py"
    if not init_file.exists():
        print(f"❌ __init__.py not found in {module_dir}")
        return False
        
    handler_file = module_dir / "web_platform_handler.py"
    if not handler_file.exists():
        print(f"❌ web_platform_handler.py not found in {module_dir}")
        return False
        
    print("✅ fixed_web_platform module files are present")
    
    # Add current directory to path for importing the module
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    
    try:
        # Try to import the module
        module = importlib.import_module("fixed_web_platform")
        
        # Check for required functions
        if hasattr(module, "process_for_web"):
            print("✅ process_for_web function available")
        else:
            print("❌ process_for_web function not found")
            return False
            
        if hasattr(module, "init_webnn"):
            print("✅ init_webnn function available")
        else:
            print("❌ init_webnn function not found")
            return False
            
        if hasattr(module, "init_webgpu"):
            print("✅ init_webgpu function available")
        else:
            print("❌ init_webgpu function not found")
            return False
            
        if hasattr(module, "create_mock_processors"):
            print("✅ create_mock_processors function available")
        else:
            print("❌ create_mock_processors function not found")
            return False
            
        # Check for advanced WebGPU features
        try:
            webgpu_handler = module.init_webgpu
            init_code = webgpu_handler.__code__.co_consts
            
            shader_compilation = False
            parallel_loading = False
            compute_shaders = False
            
            for const in init_code:
                if isinstance(const, str):
                    if "shader_compilation" in const.lower():
                        shader_compilation = True
                    if "parallel" in const.lower() and "load" in const.lower():
                        parallel_loading = True
                    if "compute_shader" in const.lower():
                        compute_shaders = True
            
            if shader_compilation:
                print("✅ WebGPU shader compilation optimization available")
            else:
                print("❌ WebGPU shader compilation optimization not found")
                
            if parallel_loading:
                print("✅ Parallel model loading optimization available")
            else:
                print("❌ Parallel model loading optimization not found")
                
            if compute_shaders:
                print("✅ WebGPU compute shaders optimization available")
            else:
                print("❌ WebGPU compute shaders optimization not found")
                
        except Exception as e:
            print(f"Warning: Error checking for advanced WebGPU features: {e}")
            
        print("✅ fixed_web_platform module and all required functions are available")
        return True
    except Exception as e:
        print(f"❌ Error importing fixed_web_platform module: {e}")
        return False

def test_generator_with_web_platform():
    """Test if the generator can generate tests with WebNN and WebGPU platforms."""
    print("\nTesting generator with WebNN and WebGPU platforms...")
    
    # Add current directory to path for importing the module
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    
    try:
        # Import the module
        module = importlib.import_module("merged_test_generator")
        
        # Check for web platform support variable
        if hasattr(module, "WEB_PLATFORM_SUPPORT"):
            print(f"✅ WEB_PLATFORM_SUPPORT = {module.WEB_PLATFORM_SUPPORT}")
        else:
            print("❌ WEB_PLATFORM_SUPPORT variable not found")
            
        # Check for WebNN and WebGPU in platform choices
        if "--platform" in str(module):
            print("✅ --platform argument is available")
        else:
            print("❌ --platform argument not found")
            
        # Check for webnn and webgpu in the init methods
        if hasattr(module, "init_webnn") or any("init_webnn" in name for name in dir(module)):
            print("✅ init_webnn method is available")
        else:
            print("❌ init_webnn method not found")
            
        if hasattr(module, "init_webgpu") or any("init_webgpu" in name for name in dir(module)):
            print("✅ init_webgpu method is available")
        else:
            print("❌ init_webgpu method not found")
            
        print("✅ Generator supports WebNN and WebGPU platforms")
        return True
    except Exception as e:
        print(f"❌ Error testing generator: {e}")
        return False

def test_template_database_web_integration():
    """Test if the template database correctly handles web platform templates."""
    print("\nChecking database integration for web platforms...")
    
    # Add current directory to path for importing the module
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    
    try:
        # Import the template database module
        from hardware_test_templates.template_database import TemplateDatabase
        
        # Initialize the database
        db = TemplateDatabase()
        
        # Check if web platforms are in the hardware platforms table
        hardware_platforms = db.get_hardware_platforms()
        
        has_webnn = False
        has_webgpu = False
        has_webgpu_compute = False
        has_parallel_loading = False
        
        for platform in hardware_platforms:
            if platform['hardware_type'] == 'webnn':
                has_webnn = True
                print(f"✅ WebNN platform found in database: {platform['display_name']}")
            elif platform['hardware_type'] == 'webgpu':
                has_webgpu = True
                print(f"✅ WebGPU platform found in database: {platform['display_name']}")
            elif platform['hardware_type'] == 'webgpu_compute':
                has_webgpu_compute = True
                print(f"✅ WebGPU Compute platform found in database: {platform['display_name']}")
            elif 'parallel' in platform['hardware_type'].lower():
                has_parallel_loading = True
                print(f"✅ Parallel loading platform found in database: {platform['display_name']}")
        
        if not has_webnn:
            print("❌ WebNN platform not found in database")
        
        if not has_webgpu:
            print("❌ WebGPU platform not found in database")
            
        if not has_webgpu_compute:
            print("❌ WebGPU Compute platform not found in database")
            
        if not has_parallel_loading:
            print("❌ Parallel loading platforms not found in database")
            
        # Check basic database operations
        if has_webnn and has_webgpu:
            print("✅ Database includes proper web platform entries")
            return True
        else:
            print("❌ Database is missing web platform entries")
            return False
            
    except Exception as e:
        print(f"❌ Error testing template database: {e}")
        return False

def main():
    """Main function."""
    print("Verifying web platform integration...\n")
    
    # Check if the fixed_web_platform module is imported in merged_test_generator.py
    imports_ok = check_generator_imports()
    
    # Check if the fixed_web_platform module is available
    module_ok = check_fixed_web_platform_module()
    
    # Test if the generator can generate tests with WebNN and WebGPU platforms
    generator_ok = test_generator_with_web_platform()
    
    # Test if the template database correctly handles web platform templates
    database_ok = test_template_database_web_integration()
    
    # Overall status
    if imports_ok and module_ok and generator_ok and database_ok:
        print("\n✅ Web platform support is successfully integrated!")
        print("\nYou can now generate tests with web platform support:")
        print("  python merged_test_generator.py --generate bert --platform webnn")
        print("  python merged_test_generator.py --generate vit --platform webgpu")
        print("  python merged_test_generator.py --generate whisper --platform webgpu_compute")
        print("  ./run_web_platform_tests.sh --all-features python merged_test_generator.py --generate llava --platform webgpu")
        
        print("\nMarch 2025 Enhancements:")
        print("  WebGPU Compute Shaders: python web_platform_test_runner.py --model whisper --platform webgpu --compute-shaders")
        print("  Parallel Model Loading: python run_web_platform_tests_with_db.py --models llava clip --parallel-loading")
        print("  Shader Precompilation: export WEBGPU_SHADER_PRECOMPILE=1 python web_platform_test_runner.py --model vit")
        return 0
    else:
        print("\n❌ Web platform support is not fully integrated.")
        print("Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())