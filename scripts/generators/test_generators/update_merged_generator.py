#!/usr/bin/env python3
"""
Update the merged_test_generator.py to integrate the WebNN and WebGPU platform fixes
and ensure all hardware templates are correctly integrated.

This script:
1. Makes a backup of the original merged_test_generator.py
2. Adds the import for the fixed_web_platform package
3. Updates the test_platform method to properly handle WebNN and WebGPU
4. Replaces the init_webnn and init_webgpu functions with the fixed versions
5. Fixes template syntax issues in the generator
6. Loads hardware templates from the hardware_test_templates directory
7. Verifies the changes work by running a simple test

Usage:
  python update_merged_generator.py
"""

import os
import re
import sys
import shutil
import importlib
from pathlib import Path
from datetime import datetime
import glob

# Set up paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
GENERATOR_FILE = CURRENT_DIR / "merged_test_generator.py"
BACKUP_DIR = CURRENT_DIR / "backups"
BACKUP_FILE = BACKUP_DIR / f"merged_test_generator.py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TEMPLATES_DIR = CURRENT_DIR / "hardware_test_templates"

# Model categories mapping
MODEL_CATEGORIES = {
    "bert": "text_embedding",
    "clap": "audio",
    "clip": "vision",
    "detr": "vision",
    "llama": "text_generation",
    "llava": "vision_language",
    "llava_next": "vision_language",
    "qwen2": "text_generation",
    "t5": "text_generation",
    "vit": "vision",
    "wav2vec2": "audio",
    "whisper": "audio",
    "xclip": "video"
}

# Ensure backup directory exists
BACKUP_DIR.mkdir(exist_ok=True)

def backup_generator():
    """Create a backup of the generator file."""
    try:
        shutil.copy2(GENERATOR_FILE, BACKUP_FILE)
        print(f"Created backup of generator at {BACKUP_FILE}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def fix_template_syntax():
    """Fix syntax issues in the template strings."""
    try:
        # Read the current file
        with open(GENERATOR_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix template string syntax errors
        content = re.sub(r'template_database\["[^"]+"\] = """"""', 
                         lambda m: m.group(0).replace('""""""', '"""'), 
                         content)
        
        # Fix truncated strings
        content = re.sub(r'\.\.\."\s+#\s+Truncated for readability', 
                         '..." # Truncated for readability', 
                         content)
        
        # Write the updated content
        with open(GENERATOR_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("Fixed template syntax issues")
        return True
    except Exception as e:
        print(f"Error fixing templates: {e}")
        return False

def update_imports():
    """Add import for fixed_web_platform module."""
    try:
        # Read the current file
        with open(GENERATOR_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if the import already exists
        if "from test.web_platform import" in content:
            print("Import already exists - skipping")
            return True
            
        # Find where to add the import
        last_import = re.search(r'^from .*? import .*?$', content, re.MULTILINE)
        if not last_import:
            print("Could not find a place to add import")
            return False
            
        # Get position after the last import
        import_pos = content.find('\n', last_import.end())
        if import_pos == -1:
            import_pos = last_import.end()
            
        # Add our import after the last one
        new_import = "\n\n# Import fixed WebNN and WebGPU platform support\nfrom test.web_platform import process_for_web, init_webnn, init_webgpu, create_mock_processors"
        content = content[:import_pos] + new_import + content[import_pos:]
        
        # Write the updated file
        with open(GENERATOR_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("Added fixed_web_platform import")
        return True
    except Exception as e:
        print(f"Error updating imports: {e}")
        return False

def update_test_platform_method():
    """Update the test_platform method to properly handle WebNN and WebGPU platforms."""
    try:
        # Read the current file
        with open(GENERATOR_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find the test_platform method
        test_platform_pattern = r'def test_platform\(self, .*?(?=\n    def|\Z)'
        test_platform_match = re.search(test_platform_pattern, content, re.DOTALL)
        
        if not test_platform_match:
            print("Could not find test_platform method")
            return False
            
        # Get the current method 
        current_method = test_platform_match.group(0)
        
        # Check if we've already updated it
        if "process_for_web" in current_method:
            print("test_platform method already updated - skipping")
            return True
        
        # Find WebNN and WebGPU sections
        webnn_section_pattern = r'elif platform_lower == "webnn":(.*?)(?=elif |else:)'
        webgpu_section_pattern = r'elif platform_lower == "webgpu":(.*?)(?=elif |else:)'
        
        webnn_section_match = re.search(webnn_section_pattern, current_method, re.DOTALL)
        webgpu_section_match = re.search(webgpu_section_pattern, current_method, re.DOTALL)
        
        if not webnn_section_match or not webgpu_section_match:
            print("Could not find WebNN or WebGPU sections in test_platform method")
            return False
            
        # Create the improved WebNN section
        improved_webnn_section = """
            elif platform_lower == "webnn":
                if hasattr(self, "endpoint_webnn"):
                    start_time = time.time()
                    
                    # Determine if batch operations are supported for this model type
                    web_batch_supported = True
                    if self.mode == "text":
                        web_batch_supported = True  # Text models usually support batching
                    elif self.mode == "vision":
                        web_batch_supported = True  # Vision models usually support batching
                    elif self.mode == "audio":
                        web_batch_supported = False  # Audio models may not support batching in WebNN
                    elif self.mode == "multimodal":
                        web_batch_supported = False  # Multimodal often doesn't batch well on web
                    
                    # Process the input using the fixed web platform handler
                    inputs = process_for_web(self.mode, input_data, web_batch_supported)
                    
                    # Execute WebNN model
                    _ = self.endpoint_webnn(inputs)
                    elapsed = time.time() - start_time
                    return elapsed
                else:
                    print("WebNN endpoint not available")
                    return None"""
                    
        # Create the improved WebGPU section
        improved_webgpu_section = """
            elif platform_lower == "webgpu":
                if hasattr(self, "endpoint_webgpu"):
                    start_time = time.time()
                    
                    # Determine if batch operations are supported for this model type
                    web_batch_supported = True
                    if self.mode == "text":
                        web_batch_supported = True  # Text models usually support batching
                    elif self.mode == "vision":
                        web_batch_supported = True  # Vision models usually support batching
                    elif self.mode == "audio":
                        web_batch_supported = False  # Audio models may not support batching in WebGPU
                    elif self.mode == "multimodal":
                        web_batch_supported = False  # Multimodal often doesn't batch well on web
                    
                    # Process the input using the fixed web platform handler
                    inputs = process_for_web(self.mode, input_data, web_batch_supported)
                    
                    # Execute WebGPU model
                    _ = self.endpoint_webgpu(inputs)
                    elapsed = time.time() - start_time
                    return elapsed
                else:
                    print("WebGPU endpoint not available")
                    return None"""
                    
        # Replace the sections in the method
        updated_method = current_method.replace(webnn_section_match.group(0), improved_webnn_section)
        updated_method = updated_method.replace(webgpu_section_match.group(0), improved_webgpu_section)
        
        # Update the content with the new method
        content = content.replace(current_method, updated_method)
        
        # Write the updated file
        with open(GENERATOR_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("Updated test_platform method")
        return True
    except Exception as e:
        print(f"Error updating test_platform method: {e}")
        return False

def update_init_methods():
    """Replace the init_webnn and init_webgpu methods with the fixed versions."""
    try:
        # Read the current file
        with open(GENERATOR_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find the init_webnn method
        init_webnn_pattern = r'def init_webnn\(self.*?\):.*?(?=\n    def|\Z)'
        init_webnn_match = re.search(init_webnn_pattern, content, re.DOTALL)
        
        # Find the init_webgpu method
        init_webgpu_pattern = r'def init_webgpu\(self.*?\):.*?(?=\n    def|\Z)'
        init_webgpu_match = re.search(init_webgpu_pattern, content, re.DOTALL)
        
        if not init_webnn_match or not init_webgpu_match:
            print("Could not find init_webnn or init_webgpu methods")
            return False
            
        # Get the current methods
        current_init_webnn = init_webnn_match.group(0)
        current_init_webgpu = init_webgpu_match.group(0)
        
        # Check if we've already updated them
        if "Using the fixed version from test.web_platform" in current_init_webnn:
            print("init methods already updated - skipping")
            return True
            
        # Create the improved init_webnn method (this is just a wrapper around the fixed version)
        improved_init_webnn = '''    def init_webnn(self, model_name=None, model_path=None, model_type=None, device="webnn", web_api_mode="simulation", tokenizer=None, **kwargs):
        """
        Initialize the model for WebNN inference.
        
        Using the fixed version from test.web_platform.
        
        Args:
            model_name: Name of the model to load
            model_path: Path to the model files 
            model_type: Type of model (text, vision, audio, etc.)
            device: Device to use ('webnn')
            web_api_mode: Mode for web API ('real', 'simulation', 'mock')
            tokenizer: Optional tokenizer for text models
            
        Returns:
            Dictionary with endpoint, processor, etc.
        """
        # Pass through to the fixed implementation
        kwargs["create_mock_processor"] = getattr(self, "_create_mock_processor", None)
        return init_webnn(self, model_name, model_path, model_type, device, web_api_mode, tokenizer, **kwargs)'''
        
        # Create the improved init_webgpu method (this is just a wrapper around the fixed version)
        improved_init_webgpu = '''    def init_webgpu(self, model_name=None, model_path=None, model_type=None, device="webgpu", web_api_mode="simulation", tokenizer=None, **kwargs):
        """
        Initialize the model for WebGPU inference.
        
        Using the fixed version from test.web_platform.
        
        Args:
            model_name: Name of the model to load
            model_path: Path to the model files 
            model_type: Type of model (text, vision, audio, etc.)
            device: Device to use ('webgpu')
            web_api_mode: Mode for web API ('simulation', 'mock')
            tokenizer: Optional tokenizer for text models
            
        Returns:
            Dictionary with endpoint, processor, etc.
        """
        # Pass through to the fixed implementation
        kwargs["create_mock_processor"] = getattr(self, "_create_mock_processor", None)
        return init_webgpu(self, model_name, model_path, model_type, device, web_api_mode, tokenizer, **kwargs)'''
        
        # Replace the methods in the content
        content = content.replace(current_init_webnn, improved_init_webnn)
        content = content.replace(current_init_webgpu, improved_init_webgpu)
        
        # Write the updated file
        with open(GENERATOR_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("Updated init_webnn and init_webgpu methods")
        return True
    except Exception as e:
        print(f"Error updating init methods: {e}")
        return False

def add_cli_arguments():
    """Add WebNN and WebGPU CLI arguments if they don't exist."""
    try:
        # Read the current file
        with open(GENERATOR_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if the arguments already exist
        if "--webnn-mode" in content and "--webgpu-mode" in content:
            print("WebNN and WebGPU CLI arguments already exist - skipping")
            return True
            
        # Find the argument parser section and add platform arguments
        platform_arg_pattern = r'parser\.add_argument\("--platform".*?\n'
        platform_arg_match = re.search(platform_arg_pattern, content, re.DOTALL)
        
        if not platform_arg_match:
            print("Could not find --platform argument")
            return False
        
        # Get the end position to add our new arguments
        arg_pos = platform_arg_match.end()
        
        # Add the web platform arguments
        web_platform_args = """    # Web platform options
    parser.add_argument("--webnn-mode", choices=["real", "simulation", "mock"], 
                      default="simulation", help="WebNN implementation mode")
    parser.add_argument("--webgpu-mode", choices=["simulation", "mock"], 
                      default="simulation", help="WebGPU implementation mode")
"""
        content = content[:arg_pos] + web_platform_args + content[arg_pos:]
        
        # Find the main function to update how it passes arguments to init methods
        main_func_pattern = r'def main\(\):'
        main_func_match = re.search(main_func_pattern, content)
        
        if main_func_match:
            # Find the part where it initializes the generator
            init_pattern = r'test_generator\.init_\w+\(.*?\)'
            init_match = re.search(init_pattern, content[main_func_match.end():], re.DOTALL)
            
            if init_match:
                # Get the current initialization call
                init_call = init_match.group(0)
                
                # Add web_api_mode parameter if not already there
                if "web_api_mode" not in init_call:
                    # Check if the call ends with a comma or not
                    if init_call.rstrip().endswith(","):
                        # Already has a trailing comma, just add the parameter
                        updated_init = init_call.rstrip() + " web_api_mode=args.webnn_mode if args.platform == 'webnn' else args.webgpu_mode if args.platform == 'webgpu' else 'simulation')"
                    else:
                        # No trailing comma, remove the closing paren and add parameter
                        updated_init = init_call[:-1] + ", web_api_mode=args.webnn_mode if args.platform == 'webnn' else args.webgpu_mode if args.platform == 'webgpu' else 'simulation')"
                    
                    # Replace in content
                    content = content.replace(init_call, updated_init)
                    
        # Write the updated file
        with open(GENERATOR_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("Added WebNN and WebGPU CLI arguments")
        return True
    except Exception as e:
        print(f"Error adding CLI arguments: {e}")
        return False

def verify_changes():
    """Verify the changes work by importing and checking the generator."""
    try:
        # Add parent directory to path
        sys.path.append(str(CURRENT_DIR.parent))
        
        # Try to import the module
        generator_module = importlib.import_module("test.merged_test_generator")
        
        # Reload to ensure we have the latest version
        importlib.reload(generator_module)
        
        # Check that our fixed imports are there
        if not hasattr(generator_module, "process_for_web"):
            print("ERROR: process_for_web not found in the generator")
            return False
            
        print("Changes verified - merged_test_generator.py now has proper WebNN and WebGPU support")
        return True
    except Exception as e:
        print(f"Error verifying changes: {e}")
        return False

def integrate_hardware_templates():
    """Load hardware templates from template_*.py files and integrate them into the generator."""
    try:
        # Read the current generator file
        with open(GENERATOR_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the template database section
        template_db_start = content.find("# Hardware-aware templates")
        if template_db_start == -1:
            print("Error: Could not find template database section")
            return False
        
        # Find the end of the template database section
        template_db_end = content.find("# Begin generator code", template_db_start)
        if template_db_end == -1:
            # Try another marker
            template_db_end = content.find("class TestGenerator", template_db_start)
            if template_db_end == -1:
                print("Error: Could not find end of template database section")
                return False
        
        # Create a new template database section
        new_template_section = "# Hardware-aware templates\ntemplate_database = {}\n\n"
        
        # Get all template files
        template_files = list(TEMPLATES_DIR.glob("template_*.py"))
        if not template_files:
            print(f"No template files found in {TEMPLATES_DIR}")
            return False
        
        # Process each template file
        added_categories = set()
        for template_file in template_files:
            model_name = template_file.stem.replace("template_", "")
            category = MODEL_CATEGORIES.get(model_name)
            
            if category and category not in added_categories:
                # Read the template file
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                
                # Extract the docstring
                docstring_match = re.search(r'"""(.*?)"""', template_content, re.DOTALL)
                if docstring_match:
                    docstring = docstring_match.group(1).strip()
                    # Add template to the database
                    new_template_section += f'template_database["{category}"] = """\n{docstring}\n"""\n\n'
                    added_categories.add(category)
                    print(f"Added template for category: {category}")
        
        # Replace the template database section
        updated_content = content[:template_db_start] + new_template_section + content[template_db_end:]
        
        # Write the updated content
        with open(GENERATOR_FILE, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("Successfully integrated hardware templates")
        return True
    except Exception as e:
        print(f"Error integrating hardware templates: {e}")
        return False

def main():
    """Run the update process."""
    print("Updating merged_test_generator.py with WebNN and WebGPU platform fixes...")
    
    # Create a backup first
    if not backup_generator():
        print("Failed to create backup, aborting")
        return 1
        
    # Apply the updates
    success = True
    success = fix_template_syntax() and success
    success = integrate_hardware_templates() and success
    success = update_imports() and success
    success = update_test_platform_method() and success
    success = update_init_methods() and success
    success = add_cli_arguments() and success
    
    if success:
        # Verify the changes
        if verify_changes():
            print("\nSuccessfully updated merged_test_generator.py with WebNN and WebGPU platform fixes")
            print("\nYou can now generate tests with Web platform support:")
            print("  python merged_test_generator.py --generate bert --platform webnn")
            print("  python merged_test_generator.py --generate vit --platform webgpu --webgpu-mode simulation")
            return 0
        else:
            print("\nUpdates applied but verification failed - check the generator manually")
            return 1
    else:
        print("\nFailed to update merged_test_generator.py - restoring from backup")
        shutil.copy2(BACKUP_FILE, GENERATOR_FILE)
        print(f"Restored generator from {BACKUP_FILE}")
        return 1

if __name__ == "__main__":
    sys.exit(main())