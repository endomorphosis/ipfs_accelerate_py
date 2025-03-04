#!/usr/bin/env python3
"""
Fix Template Generator

This script fixes the syntax issues in the merged_test_generator.py file
and updates it to properly handle hardware platforms in test generation.

It fixes:
1. Template string syntax issues
2. Loads proper templates from hardware_test_templates
3. Ensures WebNN and WebGPU support is integrated

Usage: python fix_template_generator.py
"""

import os
import re
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
GENERATOR_FILE = CURRENT_DIR / "merged_test_generator.py"
TEMPLATES_DIR = CURRENT_DIR / "hardware_test_templates"
BACKUP_DIR = CURRENT_DIR / "backups"
BACKUP_FILE = BACKUP_DIR / f"merged_test_generator.py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Ensure backup directory exists
BACKUP_DIR.mkdir(exist_ok=True)

# Model categories
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

def create_backup():
    """Create a backup of the original file."""
    try:
        shutil.copy2(GENERATOR_FILE, BACKUP_FILE)
        print(f"Created backup at {BACKUP_FILE}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def fix_template_syntax():
    """Fix template string syntax errors."""
    try:
        with open(GENERATOR_FILE, 'r') as f:
            content = f.read()
        
        # Replace invalid triple-quote syntax
        content = re.sub(r'template_database\["[^"]+"\] = """"""',
                         lambda m: m.group(0).replace('""""""', '"""'),
                         content)
        
        # Fix the docstring within the templates
        content = re.sub(r'(template_database\["[^"]+"\] = """\n.*?)\n"""',
                         lambda m: m.group(1) + '\n"""',
                         content, flags=re.DOTALL)
        
        # Write the fixed content
        with open(GENERATOR_FILE, 'w') as f:
            f.write(content)
            
        print("Fixed template syntax issues")
        return True
    except Exception as e:
        print(f"Error fixing syntax: {e}")
        return False

def load_template_from_file(template_file):
    """Load a template file and extract its docstring."""
    try:
        with open(template_file, 'r') as f:
            content = f.read()
            
        # Extract the docstring
        match = re.search(r'"""(.*?)"""', content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    except Exception as e:
        print(f"Error loading template {template_file}: {e}")
        return None

def update_templates():
    """Update template database with hardware templates."""
    try:
        with open(GENERATOR_FILE, 'r') as f:
            content = f.read()
            
        # Find template database section
        template_section_match = re.search(r'# Hardware-aware templates\s*template_database = \{\}', content)
        if not template_section_match:
            print("Could not find template database initialization")
            return False
            
        template_section_start = template_section_match.start()
        
        # Find the end of template declarations section
        template_section_end = content.find("class", template_section_start)
        if template_section_end == -1:
            # Try another marker
            template_section_end = content.find("def ", template_section_start)
            if template_section_end == -1:
                print("Could not find end of template section")
                return False
                
        # Create new template database section
        new_template_section = "# Hardware-aware templates\ntemplate_database = {}\n\n"
        
        # Process each template file
        template_files = list(TEMPLATES_DIR.glob("template_*.py"))
        if not template_files:
            print(f"No template files found in {TEMPLATES_DIR}")
            return False
            
        # Map model names to categories
        processed_categories = set()
        for template_file in template_files:
            model_name = template_file.stem.replace("template_", "")
            category = MODEL_CATEGORIES.get(model_name)
            
            if not category:
                print(f"Unknown category for {model_name}")
                continue
                
            if category in processed_categories:
                continue
                
            # Load template content
            template_content = load_template_from_file(template_file)
            if not template_content:
                continue
                
            # Add to template database
            new_template_section += f'template_database["{category}"] = """\n{template_content}\n"""\n\n'
            processed_categories.add(category)
            print(f"Added template for {category}")
            
        # Replace the template section
        new_content = content[:template_section_start] + new_template_section + content[template_section_end:]
        
        # Write the updated file
        with open(GENERATOR_FILE, 'w') as f:
            f.write(new_content)
            
        print("Updated templates from hardware_test_templates directory")
        return True
    except Exception as e:
        print(f"Error updating templates: {e}")
        return False

def add_web_platform_support():
    """Add support for WebNN and WebGPU platforms."""
    try:
        with open(GENERATOR_FILE, 'r') as f:
            content = f.read()
            
        # Check if web platform imports exist
        if "from fixed_web_platform import" not in content:
            # Add import near the top
            import_section_end = content.find("# Hardware-aware templates")
            if import_section_end == -1:
                print("Could not find import section")
                return False
                
            # Add web platform imports
            web_imports = """
# Web platform imports
try:
    from fixed_web_platform import web_platform_handler
    from fixed_web_platform.web_platform_handler import WebNNHandler, WebGPUHandler
    HAS_WEB_PLATFORM = True
except ImportError:
    HAS_WEB_PLATFORM = False

"""
            content = content[:import_section_end] + web_imports + content[import_section_end:]
            
        # Create a helper function for web platform detection
        web_detect_function = """
def detect_web_platform(platform_name):
    """Detect if web platform is available."""
    if platform_name.lower() == "webnn":
        # Check if WebNN is available
        try:
            # In a real browser environment, this would check for WebNN API
            # In simulation mode, we assume it's available if the package is
            from fixed_web_platform.web_platform_handler import WebNNHandler
            return True
        except ImportError:
            return False
    elif platform_name.lower() == "webgpu":
        # Check if WebGPU is available
        try:
            # In a real browser environment, this would check for WebGPU API
            # In simulation mode, we assume it's available if the package is
            from fixed_web_platform.web_platform_handler import WebGPUHandler
            return True
        except ImportError:
            return False
    return False
"""
        
        # Add the helper function if it doesn't exist
        if "def detect_web_platform" not in content:
            # Find a good spot to add the function
            class_start = content.find("class TestGenerator")
            if class_start == -1:
                # Alternative: add before the first def
                function_start = content.find("def ")
                if function_start == -1:
                    print("Could not find a place to add web platform detection function")
                    return False
                content = content[:function_start] + web_detect_function + content[function_start:]
            else:
                content = content[:class_start] + web_detect_function + content[class_start:]
                
        # Write the updated file
        with open(GENERATOR_FILE, 'w') as f:
            f.write(content)
            
        print("Added web platform support")
        return True
    except Exception as e:
        print(f"Error adding web platform support: {e}")
        return False

def update_platform_code():
    """Update the platform-specific code in the generator."""
    try:
        with open(GENERATOR_FILE, 'r') as f:
            content = f.read()
            
        # Find the class that handles platform generation
        class_match = re.search(r'class TestGenerator', content)
        if not class_match:
            print("Could not find TestGenerator class")
            return False
            
        # Check if platform handling code exists
        platform_method_match = re.search(r'def generate_for_platform\(self', content)
        if not platform_method_match:
            print("Could not find generate_for_platform method")
            return False
            
        # Improve platform handling to include WebNN and WebGPU
        webnn_webgpu_support = """
    def generate_for_platform(self, model, platform, template):
        """Generate platform-specific code for the given model and platform."""
        if platform.lower() == "webnn":
            # WebNN platform support
            platform_imports = ["# WebNN-specific imports", "from fixed_web_platform.web_platform_handler import WebNNHandler"]
            platform_init = "self.init_webnn()"
            platform_handler = "handler = self.create_webnn_handler()"
            return {
                "imports": platform_imports,
                "init": platform_init,
                "handler": platform_handler,
                "platform_name": "WEBNN"
            }
        elif platform.lower() == "webgpu":
            # WebGPU platform support
            platform_imports = ["# WebGPU-specific imports", "from fixed_web_platform.web_platform_handler import WebGPUHandler"]
            # Add model-specific optimizations
            if self.model_category == "audio":
                platform_imports.append("from fixed_web_platform.webgpu_audio_compute_shaders import optimize_for_firefox")
            elif self.model_category in ["vision_language", "multimodal"]:
                platform_imports.append("from fixed_web_platform.progressive_model_loader import ParallelModelLoader")
            elif self.model_category == "text_generation":
                platform_imports.append("from fixed_web_platform.webgpu_4bit_inference import WebGPU4BitInference")
                
            platform_init = "self.init_webgpu()"
            platform_handler = "handler = self.create_webgpu_handler()"
            return {
                "imports": platform_imports,
                "init": platform_init,
                "handler": platform_handler,
                "platform_name": "WEBGPU"
            }
        """
        
        # Find the existing platform method
        platform_method_start = platform_method_match.start()
        platform_method_end = content.find("def ", platform_method_start + 10)
        if platform_method_end == -1:
            # Try with class end
            platform_method_end = content.find("class ", platform_method_start + 10)
            if platform_method_end == -1:
                print("Could not find end of generate_for_platform method")
                return False
                
        # Replace the existing method with improved version
        content = content[:platform_method_start] + webnn_webgpu_support + content[platform_method_end:]
        
        # Write the updated file
        with open(GENERATOR_FILE, 'w') as f:
            f.write(content)
            
        print("Updated platform code in generator")
        return True
    except Exception as e:
        print(f"Error updating platform code: {e}")
        return False

def main():
    """Main function."""
    print("Fixing template generator...")
    
    # Create backup
    if not create_backup():
        print("Failed to create backup, aborting")
        return 1
        
    # Apply fixes
    success = True
    success = fix_template_syntax() and success
    success = update_templates() and success
    success = add_web_platform_support() and success
    success = update_platform_code() and success
    
    if success:
        print("Successfully fixed template generator")
        return 0
    else:
        print("Failed to fix template generator, restoring backup")
        shutil.copy2(BACKUP_FILE, GENERATOR_FILE)
        return 1

if __name__ == "__main__":
    sys.exit(main())