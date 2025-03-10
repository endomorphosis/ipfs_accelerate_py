#!/usr/bin/env python3
"""
Update Generator Hardware Support

This script updates the generators to fix syntax issues and ensure proper WebNN and WebGPU
platform support for test and implementation templates.

Key enhancements:
1. Fixes syntax errors in template strings (missing triple quotes, improper escaping)
2. Updates WebNN and WebGPU integration in fixed_merged_test_generator.py
3. Updates integrated_skillset_generator.py for better web platform support
4. Ensures proper hardware detection for all platforms
5. Improves cross-platform compatibility across all test templates

This works in conjunction with fix_generator_hardware_support.py but focuses on specific
syntax and template issues that need immediate attention.
"""

import os
import sys
import re
import shutil
from pathlib import Path
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generator_update")

# Constants
TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def fix_template_syntax_errors():
    """Fix syntax errors in template strings in fixed_merged_test_generator.py."""
    generator_file = TEST_DIR / "fixed_merged_test_generator.py"
    
    if not generator_file.exists():
        logger.error(f"Generator file not found: {generator_file}")
        return False
    
    logger.info(f"Fixing template syntax errors in {generator_file}")
    
    # Backup the file
    backup_file = generator_file.with_suffix('.py.backup')
    shutil.copy2(generator_file, backup_file)
    logger.info(f"Created backup at {backup_file}")
    
    try:
        # Read the file
        with open(generator_file, 'r') as f:
            content = f.read()
        
        # 1. Fix audio template (missing triple quotes)
        if 'template_database["audio"] = """"""' in content:
            logger.info("Fixing audio template triple quotes")
            content = content.replace(
                'template_database["audio"] = """"""',
                'template_database["audio"] = """'
            )
        
        # 2. Fix video template (missing triple quotes)
        if 'template_database["video"] = """"""' in content:
            logger.info("Fixing video template triple quotes")
            content = content.replace(
                'template_database["video"] = """"""',
                'template_database["video"] = """'
            )
        
        # 3. Fix template truncation comments
        for template_type in ["text_generation", "vision", "text_embedding", "audio", "video"]:
            pattern = f'template_database\\["{template_type}"\\] = """.*?""\\"\s+# Truncated for readability'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                logger.info(f"Fixing {template_type} template truncation")
                fixed_template = match.group(0).replace('"""  # Truncated', '"""  # Truncated')
                content = content.replace(match.group(0), fixed_template)
        
        # 4. Fix escaped quotes in doc strings
        for docstring_pattern in [
            r'(""".*?def __call__.*?\\\")(.+?)(\\\".*)"""',
            r'(""".*?Return mock output.*?\\\")(.+?)(\\\".*)"""'
        ]:
            matches = re.findall(docstring_pattern, content, re.DOTALL)
            for match in matches:
                full_match = match[0] + match[1] + match[2]
                fixed_match = match[0] + match[1].replace('"', '\\"') + match[2]
                if full_match != fixed_match:
                    logger.info("Fixing escaped quotes in docstring")
                    content = content.replace(full_match, fixed_match)
        
        # 5. Fix WebNN and WebGPU imports
        if "from fixed_web_platform import" not in content:
            logger.info("Adding fixed_web_platform imports")
            web_platform_import = '''
# Import fixed WebNN and WebGPU platform support
try:
    from fixed_web_platform import process_for_web, init_webnn, init_webgpu, create_mock_processors
    WEB_PLATFORM_SUPPORT = True
except ImportError:
    WEB_PLATFORM_SUPPORT = False
    print("WebNN and WebGPU platform support not available - install the fixed_web_platform module")
from unittest.mock import MagicMock
'''
            # Find a good insertion point
            import_section_end = content.find("# Configure logging")
            if import_section_end > 0:
                content = content[:import_section_end] + web_platform_import + content[import_section_end:]
        
        # 6. Update template database with WebNN and WebGPU for all templates
        for template_type in ["text_generation", "text_embedding", "vision", "audio", "video", "vision_language"]:
            template_start = content.find(f'template_database["{template_type}"] = """')
            if template_start > 0:
                # Find the platform list section
                platform_section_start = content.find("- CPU:", template_start)
                if platform_section_start > 0:
                    platform_section_end = content.find("\n\n", platform_section_start)
                    if platform_section_end > 0:
                        # Check if WebNN and WebGPU are already mentioned
                        platform_section = content[platform_section_start:platform_section_end]
                        if "WebNN" not in platform_section or "WebGPU" not in platform_section:
                            # Add WebNN and WebGPU to the platform list
                            updated_section = platform_section
                            if "WebNN" not in platform_section:
                                updated_section += "\n- WebNN: Web Neural Network API (browser)"
                            if "WebGPU" not in platform_section:
                                updated_section += "\n- WebGPU: Web GPU API (browser)"
                            content = content.replace(platform_section, updated_section)
                            logger.info(f"Added WebNN and WebGPU to {template_type} template")
        
        # Write the updated content
        with open(generator_file, 'w') as f:
            f.write(content)
        
        logger.info("Successfully fixed template syntax errors")
        return True
    except Exception as e:
        logger.error(f"Error fixing template syntax: {e}")
        logger.error(traceback.format_exc())
        # Restore from backup
        shutil.copy2(backup_file, generator_file)
        logger.info(f"Restored from backup due to error")
        return False

def update_platform_detection():
    """Update platform detection code for WebNN and WebGPU."""
    generator_file = TEST_DIR / "fixed_merged_test_generator.py"
    
    if not generator_file.exists():
        logger.error(f"Generator file not found: {generator_file}")
        return False
    
    logger.info(f"Updating platform detection for WebNN and WebGPU in {generator_file}")
    
    # Backup the file if not already backed up
    backup_file = generator_file.with_suffix('.py.backup')
    if not backup_file.exists():
        shutil.copy2(generator_file, backup_file)
        logger.info(f"Created backup at {backup_file}")
    
    try:
        # Read the file
        with open(generator_file, 'r') as f:
            content = f.read()
        
        # Find the detect_available_hardware function
        detect_func_start = content.find("def detect_available_hardware()")
        if detect_func_start < 0:
            logger.error("Could not find detect_available_hardware function")
            return False
        
        # Find the WebNN and WebGPU detection section
        webnn_section_start = content.find("# WebNN and WebGPU", detect_func_start)
        if webnn_section_start < 0:
            # No WebNN section found, find a good insertion point before the return
            func_end = content.find("return available_hardware", detect_func_start)
            if func_end > 0:
                # Add WebNN and WebGPU detection code
                web_detection_code = '''
    # WebNN and WebGPU
    try:
        # First check if fixed_web_platform module is available
        if WEB_PLATFORM_SUPPORT:
            available_hardware["webnn"] = True
            available_hardware["webgpu"] = True
            logger.info("WebNN and WebGPU simulation available via fixed_web_platform module")
        else:
            # Check if we're in a browser environment
            try:
                import js
                if hasattr(js, 'navigator'):
                    if hasattr(js.navigator, 'ml'):
                        available_hardware["webnn"] = True
                        logger.info("WebNN API detected in browser environment")
                    else:
                        available_hardware["webnn"] = False

                    if hasattr(js.navigator, 'gpu'):
                        available_hardware["webgpu"] = True
                        logger.info("WebGPU API detected in browser environment")
                    else:
                        available_hardware["webgpu"] = False
                else:
                    available_hardware["webnn"] = False
                    available_hardware["webgpu"] = False
            except ImportError:
                # Not in a browser environment, check environment variables
                available_hardware["webnn"] = os.environ.get("WEBNN_ENABLED", "0") == "1"
                available_hardware["webgpu"] = os.environ.get("WEBGPU_ENABLED", "0") == "1"
                if available_hardware["webnn"] or available_hardware["webgpu"]:
                    logger.info(f"Web platform from environment: WebNN={available_hardware['webnn']}, WebGPU={available_hardware['webgpu']}")
                else:
                    logger.info("WebNN and WebGPU not available")
    except Exception as e:
        available_hardware["webnn"] = False
        available_hardware["webgpu"] = False
        logger.warning(f"Error detecting WebNN and WebGPU: {e}")
'''
                content = content[:func_end] + web_detection_code + content[func_end:]
                logger.info("Added WebNN and WebGPU detection code")
        else:
            # WebNN section exists, update it if needed
            webnn_section_end = content.find("    # Check for NPU", webnn_section_start)
            if webnn_section_end < 0:
                webnn_section_end = content.find("return available_hardware", webnn_section_start)
            
            if webnn_section_end > 0:
                # Check if the section contains WEB_PLATFORM_SUPPORT
                webnn_section = content[webnn_section_start:webnn_section_end]
                if "WEB_PLATFORM_SUPPORT" not in webnn_section:
                    # Update the section to use WEB_PLATFORM_SUPPORT
                    updated_section = '''
    # WebNN and WebGPU - check if fixed_web_platform module is available
    # For WebNN
    if WEB_PLATFORM_SUPPORT:
        available_hardware["webnn"] = True
        available_hardware["webgpu"] = True
        logger.info("WebNN and WebGPU simulation available via fixed_web_platform module")
    else:
        # Check if we're in a browser environment
        try:
            # Check if we're in a browser environment
            import js
            if hasattr(js, 'navigator'):
                if hasattr(js.navigator, 'ml'):
                    logger.info("WebNN API detected in browser environment")
                    available_hardware["webnn"] = True
                if hasattr(js.navigator, 'gpu'):
                    logger.info("WebGPU API detected in browser environment")
                    available_hardware["webgpu"] = True
        except ImportError:
            # Not in a browser environment, use simulation if WEB_PLATFORM_SUPPORT is True
            pass
'''
                    content = content.replace(webnn_section, updated_section)
                    logger.info("Updated WebNN and WebGPU detection code")
        
        # Write the updated content
        with open(generator_file, 'w') as f:
            f.write(content)
        
        logger.info("Successfully updated platform detection")
        return True
    except Exception as e:
        logger.error(f"Error updating platform detection: {e}")
        logger.error(traceback.format_exc())
        return False

def update_integrated_skillset_generator():
    """Update integrated_skillset_generator.py for better web platform support."""
    generator_file = TEST_DIR / "integrated_skillset_generator.py"
    
    if not generator_file.exists():
        logger.error(f"Integrated skillset generator file not found: {generator_file}")
        return False
    
    logger.info(f"Updating web platform support in {generator_file}")
    
    # Backup the file
    backup_file = generator_file.with_suffix('.py.backup')
    if not backup_file.exists():
        shutil.copy2(generator_file, backup_file)
        logger.info(f"Created backup at {backup_file}")
    
    try:
        # Read the file
        with open(generator_file, 'r') as f:
            content = f.read()
        
        # Find and update hardware compatibility matrix for web platforms
        category_compat_start = content.find("category_compatibility = {")
        if category_compat_start > 0:
            category_compat_end = content.find("}", category_compat_start)
            category_compat_end = content.find("}", category_compat_end + 1)
            if category_compat_end > 0:
                # Extract category compatibility section
                category_compat = content[category_compat_start:category_compat_end+1]
                
                # Check each category for WebNN and WebGPU
                for category in ["text_embedding", "text_generation", "vision", "audio", "vision_language", "video"]:
                    category_start = category_compat.find(f'"{category}": {{')
                    if category_start > 0:
                        category_end = category_compat.find("}", category_start)
                        if category_end > 0:
                            # Update the category to include WebNN and WebGPU
                            category_section = category_compat[category_start:category_end+1]
                            if "webnn" not in category_section or "webgpu" not in category_section:
                                updated_section = category_section[:-1]
                                if updated_section[-1] != ",":
                                    updated_section += ","
                                if "webnn" not in category_section:
                                    updated_section += ' "webnn": True,'
                                if "webgpu" not in category_section:
                                    updated_section += ' "webgpu": True'
                                updated_section += " }"
                                content = content.replace(category_section, updated_section)
                                logger.info(f"Updated {category} category to include WebNN and WebGPU")
        
        # Update model_specific_overrides to ensure all key models have WebNN and WebGPU support
        model_overrides_start = content.find("model_specific_overrides = {")
        if model_overrides_start > 0:
            model_overrides_end = content.find("}", model_overrides_start)
            model_overrides_end = content.find("}", model_overrides_end + 1)
            if model_overrides_end > 0:
                # Extract model overrides section
                model_overrides = content[model_overrides_start:model_overrides_end+1]
                
                # Check each key model for WebNN and WebGPU
                key_models = ["bert", "t5", "llama", "clip", "vit", "clap", "whisper", 
                             "wav2vec2", "llava", "llava-next", "qwen2", "detr"]
                
                for model in key_models:
                    model_start = model_overrides.find(f'"{model}": {{')
                    if model_start > 0:
                        model_end = model_overrides.find("}", model_start)
                        if model_end > 0:
                            # Update the model to include WebNN and WebGPU
                            model_section = model_overrides[model_start:model_end+1]
                            if "webnn" not in model_section or "webgpu" not in model_section:
                                updated_section = model_section[:-1]
                                if updated_section[-1] != ",":
                                    updated_section += ","
                                if "webnn" not in model_section:
                                    updated_section += ' "webnn": True,'
                                if "webgpu" not in model_section:
                                    updated_section += ' "webgpu": True'
                                updated_section += " }"
                                content = content.replace(model_section, updated_section)
                                logger.info(f"Updated {model} model override to include WebNN and WebGPU")
        
        # Make sure extract_model_metadata includes hardware_compatibility
        extract_metadata_func = content.find("def extract_model_metadata(")
        if extract_metadata_func > 0:
            # Check if hardware_compatibility is already included
            if "hardware_compatibility" not in content[extract_metadata_func:extract_metadata_func+500]:
                # Add hardware_compatibility to the metadata
                metadata_assignment = content.find("metadata = {", extract_metadata_func)
                if metadata_assignment > 0:
                    metadata_end = content.find("}", metadata_assignment)
                    if metadata_end > 0:
                        # Add hardware_compatibility after primary_task
                        metadata_section = content[metadata_assignment:metadata_end+1]
                        if "primary_task" in metadata_section and "hardware_compatibility" not in metadata_section:
                            primary_task_line = metadata_section.find('"primary_task"')
                            if primary_task_line > 0:
                                line_end = metadata_section.find("\n", primary_task_line)
                                if line_end > 0:
                                    updated_metadata = (
                                        metadata_section[:line_end] + 
                                        ",\n            \"hardware_compatibility\": self.analyze_hardware_compatibility(model_type, test_results)" +
                                        metadata_section[line_end:]
                                    )
                                    content = content.replace(metadata_section, updated_metadata)
                                    logger.info("Added hardware_compatibility to extract_model_metadata")
        
        # Make sure generate_skillset includes web platform support
        generate_skillset_func = content.find("def generate_skillset(")
        if generate_skillset_func > 0:
            # Check if cross_platform parameter is included
            if "cross_platform" not in content[generate_skillset_func:generate_skillset_func+500]:
                # Add cross_platform parameter to function signature
                func_sig_end = content.find(")", generate_skillset_func)
                if func_sig_end > 0:
                    old_sig = content[generate_skillset_func:func_sig_end+1]
                    new_sig = old_sig.replace(")", ", cross_platform: bool = False)")
                    content = content.replace(old_sig, new_sig)
                    logger.info("Added cross_platform parameter to generate_skillset")
            
            # Check if cross_platform is applied in the function
            if "cross_platform and" not in content[generate_skillset_func:generate_skillset_func+3000]:
                # Find where to add the cross_platform logic
                hardware_compat_section = content.find("if hardware_platforms:", generate_skillset_func)
                if hardware_compat_section > 0:
                    # Add the cross_platform logic after the hardware_platforms check
                    cross_platform_logic = '''
            # If cross-platform is requested, ensure all platforms are enabled with real implementations
            if cross_platform and "all" in hardware_platforms:
                # Enable full real implementation support for all platforms
                for platform in HARDWARE_PLATFORMS:
                    # CPU is always True
                    if platform == "cpu":
                        continue
                        
                    # Set to REAL implementation for everything
                    hardware_compat[platform] = "real"
                
                model_metadata["hardware_compatibility"] = hardware_compat
                logger.info(f"Enhanced cross-platform REAL implementation support for {model_type}")
'''
                    # Find a good insertion point within the if hardware_platforms block
                    insertion_point = content.find("if \"all\" not in hardware_platforms:", hardware_compat_section)
                    if insertion_point > 0:
                        # Insert after this check
                        next_block = content.find("            ", insertion_point + 100)
                        if next_block > 0:
                            content = content[:next_block] + cross_platform_logic + content[next_block:]
                            logger.info("Added cross_platform logic to generate_skillset")
        
        # Write the updated content
        with open(generator_file, 'w') as f:
            f.write(content)
        
        logger.info("Successfully updated integrated_skillset_generator.py")
        return True
    except Exception as e:
        logger.error(f"Error updating integrated_skillset_generator.py: {e}")
        logger.error(traceback.format_exc())
        return False

def update_template_hardware_detection():
    """Update template_hardware_detection.py if it exists."""
    template_file = TEST_DIR / "template_hardware_detection.py"
    
    if not template_file.exists():
        logger.info(f"template_hardware_detection.py not found, skipping update")
        return True
    
    logger.info(f"Updating web platform support in {template_file}")
    
    # Backup the file
    backup_file = template_file.with_suffix('.py.backup')
    if not backup_file.exists():
        shutil.copy2(template_file, backup_file)
        logger.info(f"Created backup at {backup_file}")
    
    try:
        # Read the file
        with open(template_file, 'r') as f:
            content = f.read()
        
        # Check if WebNN and WebGPU detection is included in hardware detection code
        detect_hardware_func = content.find("def generate_hardware_detection_code()")
        if detect_hardware_func > 0:
            detect_code = content[detect_hardware_func:content.find("return code", detect_hardware_func)]
            
            if "webnn" not in detect_code.lower() or "webgpu" not in detect_code.lower():
                # Add WebNN and WebGPU detection to the code
                webnn_webgpu_code = '''
    # WebNN and WebGPU
    try:
        # First check if fixed_web_platform module is available
        try:
            import fixed_web_platform
            available_hardware["webnn"] = True
            available_hardware["webgpu"] = True
            logger.info("WebNN and WebGPU simulation available")
        except ImportError:
            # Check if we're in a browser environment
            try:
                import js
                if hasattr(js, 'navigator'):
                    if hasattr(js.navigator, 'ml'):
                        available_hardware["webnn"] = True
                        logger.info("WebNN API detected in browser environment")
                    else:
                        available_hardware["webnn"] = False

                    if hasattr(js.navigator, 'gpu'):
                        available_hardware["webgpu"] = True
                        logger.info("WebGPU API detected in browser environment")
                    else:
                        available_hardware["webgpu"] = False
                else:
                    available_hardware["webnn"] = False
                    available_hardware["webgpu"] = False
            except ImportError:
                # Not in a browser environment, check environment variables
                available_hardware["webnn"] = os.environ.get("WEBNN_ENABLED", "0") == "1"
                available_hardware["webgpu"] = os.environ.get("WEBGPU_ENABLED", "0") == "1"
                if available_hardware["webnn"] or available_hardware["webgpu"]:
                    logger.info(f"Web platform from environment: WebNN={available_hardware['webnn']}, WebGPU={available_hardware['webgpu']}")
                else:
                    logger.info("WebNN and WebGPU not available")
    except Exception as e:
        available_hardware["webnn"] = False
        available_hardware["webgpu"] = False
        logger.warning(f"Error detecting WebNN and WebGPU: {e}")
'''
                # Find the return statement
                return_pos = content.find("return code", detect_hardware_func)
                if return_pos > 0:
                    # Add the WebNN and WebGPU detection code before the return
                    insertion_pos = content.rfind("\n", 0, return_pos)
                    content = content[:insertion_pos] + webnn_webgpu_code + content[insertion_pos:]
                    logger.info("Added WebNN and WebGPU detection to hardware detection code")
        
        # Check if init_webnn and init_webgpu methods are included
        init_methods_func = content.find("def generate_hardware_init_methods(")
        if init_methods_func > 0:
            init_methods_code = content[init_methods_func:content.find("return code", init_methods_func)]
            
            if "def init_webnn" not in init_methods_code or "def init_webgpu" not in init_methods_code:
                # Add WebNN and WebGPU init methods
                webnn_webgpu_init = '''
def init_webnn(self, model_name=None, model_path=None, model_type=None, device="webnn", web_api_mode="simulation", tokenizer=None, **kwargs):
    """Initialize model for WebNN inference."""
    # Check for fixed_web_platform support
    try:
        from fixed_web_platform import init_webnn
        kwargs["create_mock_processor"] = getattr(self, "_create_mock_processor", None)
        return init_webnn(self, model_name, model_path, model_type, device, web_api_mode, tokenizer, **kwargs)
    except ImportError:
        # Fallback implementation
        import asyncio
        handler = lambda x: {"output": "MOCK WEBNN OUTPUT", "implementation_type": "MOCK_WEBNN"}
        return None, None, handler, asyncio.Queue(8), 1

def init_webgpu(self, model_name=None, model_path=None, model_type=None, device="webgpu", web_api_mode="simulation", tokenizer=None, **kwargs):
    """Initialize model for WebGPU inference."""
    # Check for fixed_web_platform support
    try:
        from fixed_web_platform import init_webgpu
        kwargs["create_mock_processor"] = getattr(self, "_create_mock_processor", None)
        return init_webgpu(self, model_name, model_path, model_type, device, web_api_mode, tokenizer, **kwargs)
    except ImportError:
        # Fallback implementation
        import asyncio
        handler = lambda x: {"output": "MOCK WEBGPU OUTPUT", "implementation_type": "MOCK_WEBGPU"}
        return None, None, handler, asyncio.Queue(8), 1
'''
                # Check if there's a conditional block for use_web_platform
                web_platform_check = re.search(r'if use_web_platform:', init_methods_code)
                if web_platform_check:
                    # Find the block end
                    block_start = web_platform_check.end()
                    block_end = init_methods_code.find("return code", block_start)
                    if block_end > 0:
                        # Replace the code inside the conditional block
                        conditional_block = init_methods_code[block_start:block_end]
                        updated_block = f"    code += '''{webnn_webgpu_init}'''\n"
                        content = content.replace(conditional_block, updated_block)
                        logger.info("Updated WebNN and WebGPU init methods in conditional block")
                else:
                    # Add conditional block for use_web_platform
                    return_pos = content.find("return code", init_methods_func)
                    if return_pos > 0:
                        # Add the conditional block before the return
                        insertion_pos = content.rfind("\n", 0, return_pos)
                        conditional_block = f'''
    # Add WebNN and WebGPU methods if requested
    if use_web_platform:
        code += '''{webnn_webgpu_init}'''
'''
                        content = content[:insertion_pos] + conditional_block + content[insertion_pos:]
                        logger.info("Added WebNN and WebGPU init methods with conditional block")
        
        # Check if create_webnn and create_webgpu methods are included
        creation_methods_func = content.find("def generate_creation_methods(")
        if creation_methods_func > 0:
            creation_methods_code = content[creation_methods_func:content.find("return code", creation_methods_func)]
            
            if "create_webnn_" not in creation_methods_code or "create_webgpu_" not in creation_methods_code:
                # Add WebNN and WebGPU creation methods
                model_task_var = "${model_task}" if "${model_task}" in creation_methods_code else "model_task"
                webnn_webgpu_create = f'''
def create_webnn_{model_task_var}_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None):
    """Create a handler function for WebNN inference."""
    # Implementation will be specific to your model type
    pass

def create_webgpu_{model_task_var}_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None):
    """Create a handler function for WebGPU inference."""
    # Implementation will be specific to your model type
    pass
'''
                # Check if there's a conditional block for use_web_platform
                web_platform_check = re.search(r'if use_web_platform:', creation_methods_code)
                if web_platform_check:
                    # Find the block end
                    block_start = web_platform_check.end()
                    block_end = creation_methods_code.find("return code", block_start)
                    if block_end > 0:
                        # Replace the code inside the conditional block
                        conditional_block = creation_methods_code[block_start:block_end]
                        updated_block = f"    code += f\"\"\"{webnn_webgpu_create}\"\"\"\n"
                        content = content.replace(conditional_block, updated_block)
                        logger.info("Updated WebNN and WebGPU creation methods in conditional block")
                else:
                    # Add conditional block for use_web_platform
                    return_pos = content.find("return code", creation_methods_func)
                    if return_pos > 0:
                        # Add the conditional block before the return
                        insertion_pos = content.rfind("\n", 0, return_pos)
                        conditional_block = f'''
    # Add WebNN and WebGPU methods if requested
    if use_web_platform:
        code += f\"\"\"{webnn_webgpu_create}\"\"\"
'''
                        content = content[:insertion_pos] + conditional_block + content[insertion_pos:]
                        logger.info("Added WebNN and WebGPU creation methods with conditional block")
        
        # Write the updated content
        with open(template_file, 'w') as f:
            f.write(content)
        
        logger.info("Successfully updated template_hardware_detection.py")
        return True
    except Exception as e:
        logger.error(f"Error updating template_hardware_detection.py: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run all updates."""
    logger.info("Starting generator hardware support update")
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Fix template syntax errors
    if fix_template_syntax_errors():
        success_count += 1
    
    # Step 2: Update platform detection
    if update_platform_detection():
        success_count += 1
    
    # Step 3: Update integrated skillset generator
    if update_integrated_skillset_generator():
        success_count += 1
    
    # Step 4: Update template hardware detection
    if update_template_hardware_detection():
        success_count += 1
    
    logger.info(f"Completed {success_count}/{total_steps} updates successfully")
    
    if success_count == total_steps:
        logger.info("All updates completed successfully! ✅")
        return 0
    else:
        logger.warning(f"Some updates failed. Check the log for details. ⚠️")
        return 1

if __name__ == "__main__":
    sys.exit(main())