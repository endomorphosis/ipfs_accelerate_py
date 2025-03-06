#\!/usr/bin/env python

# Add detect_capabilities method to platform_detector.py
import re

filepath = 'fixed_web_platform/unified_framework/platform_detector.py'

# Read the file content
with open(filepath, 'r') as f:
    content = f.read()

# Add the detect_capabilities method after the detect_platform method
detect_platform_pattern = r'def detect_platform\(self\)(.*?)def (.*?)\('
detect_platform_match = re.search(detect_platform_pattern, content, re.DOTALL)

if detect_platform_match:
    # Get the indentation
    lines = detect_platform_match.group(1).splitlines()
    indentation = ''
    for line in lines:
        if line.strip():
            indentation = re.match(r'^(\s*)', line).group(1)
            break

    # Create the new method with proper indentation
    new_method = f'''
{indentation}def detect_capabilities(self) -> Dict[str, Any]:
{indentation}    """
{indentation}    Detect platform capabilities and return configuration options.
{indentation}    
{indentation}    Returns:
{indentation}        Dictionary with detected capabilities as configuration options
{indentation}    """
{indentation}    # Get platform info
{indentation}    platform_info = self.detect_platform()
{indentation}    
{indentation}    # Create configuration dictionary
{indentation}    config = {{
{indentation}        "browser": platform_info["browser"]["name"],
{indentation}        "browser_version": platform_info["browser"]["version"],
{indentation}        "webgpu_supported": platform_info.get("features", {{}}).get("webgpu", True),
{indentation}        "webnn_supported": platform_info.get("features", {{}}).get("webnn", True),
{indentation}        "wasm_supported": platform_info.get("features", {{}}).get("wasm", True),
{indentation}        "hardware_platform": platform_info["hardware"].get("platform", "unknown"),
{indentation}        "hardware_memory_gb": platform_info["hardware"].get("memory_gb", 4)
{indentation}    }}
{indentation}    
{indentation}    # Set optimization flags based on capabilities
{indentation}    browser = platform_info["browser"]["name"].lower()
{indentation}    
{indentation}    # Add WebGPU optimization flags
{indentation}    if config["webgpu_supported"]:
{indentation}        config["enable_shader_precompilation"] = True
{indentation}        
{indentation}        # Add model-type specific optimizations
{indentation}        if hasattr(self, "model_type"):
{indentation}            # Enable compute shaders for audio models in Firefox
{indentation}            if browser == "firefox" and self.model_type == "audio":
{indentation}                config["enable_compute_shaders"] = True
{indentation}                config["firefox_audio_optimization"] = True
{indentation}                config["workgroup_size"] = [256, 1, 1]  # Optimized for Firefox
{indentation}            elif self.model_type == "audio":
{indentation}                config["enable_compute_shaders"] = True
{indentation}                config["workgroup_size"] = [128, 2, 1]  # Standard size
{indentation}                
{indentation}            # Enable parallel loading for multimodal models
{indentation}            if self.model_type == "multimodal":
{indentation}                config["enable_parallel_loading"] = True
{indentation}                config["progressive_loading"] = True
{indentation}    
{indentation}    return config
{indentation}
'''

    # Insert the new method
    new_content = re.sub(detect_platform_pattern, 
                         f'def detect_platform(self){detect_platform_match.group(1)}{new_method}def {detect_platform_match.group(2)}(', 
                         content, 
                         flags=re.DOTALL)

    # Write the content back to the file
    with open(filepath, 'w') as f:
        f.write(new_content)
        
    print("Added detect_capabilities method to platform_detector.py")
else:
    print("Could not find detect_platform method in platform_detector.py")
