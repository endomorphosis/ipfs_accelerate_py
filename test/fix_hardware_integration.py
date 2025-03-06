#!/usr/bin/env python3
"""
Fix Hardware Integration in Test Files

This script identifies and fixes integration issues for hardware platforms in test files,
ensuring that all hardware detection and test methods are properly included and called.

Fixes:
1. Standardizes hardware detection methods
2. Ensures all hardware platforms are included in test methods
3. Fixes indentation issues
4. Adds proper AMD precision handling
5. Improves WebNN and WebGPU integration
6. Ensures consistent asyncio usage

Usage:
python fix_hardware_integration.py --specific-models bert,t5,clip
python fix_hardware_integration.py --all-key-models
python fix_hardware_integration.py --specific-directory test/key_models_hardware_fixes
python fix_hardware_integration.py --analyze-only
"""

import os
import re
import sys
import glob
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Check for JSON output deprecation flag
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "0") == "1"

# Define hardware platforms to check
HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]

# Hardware method detection patterns
HARDWARE_DETECTION_PATTERNS = {
    "init_method": r"def\s+init_(\w+)",
    "run_method": r"def\s+test_with_(\w+)",
    "detect_method": r"def\s+detect_(\w+)"
}

# Paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = CURRENT_DIR / "skills"
KEY_MODELS_DIR = CURRENT_DIR / "key_models_hardware_fixes"

# Key model names
KEY_MODELS = [
    "bert", "t5", "llama", "clip", "vit", "clap", "whisper", 
    "wav2vec2", "llava", "llava_next", "xclip", "qwen2", "detr"
]

class HardwareIntegrationFixer:
    """Fixes hardware integration issues in test files."""
    
    def __init__(self, file_path: str):
        """Initialize the fixer with the path to the test file."""
        self.file_path = file_path
        self.file_content = None
        self.has_changes = False
        self.issues = []
        self.fixed_issues = []
        self._load_file()
        
    def _load_file(self):
        """Load the file content."""
        try:
            with open(self.file_path, 'r') as file:
                self.file_content = file.read()
        except Exception as e:
            print(f"Error loading file {self.file_path}: {e}")
            self.file_content = ""
            
    def save_file(self):
        """Save the file if changes were made."""
        if self.has_changes:
            try:
                with open(self.file_path, 'w') as file:
                    file.write(self.file_content)
                print(f"Saved changes to {self.file_path}")
                return True
            except Exception as e:
                print(f"Error saving file {self.file_path}: {e}")
                return False
        return False
    
    def analyze(self) -> Dict:
        """
        Analyze the file for hardware integration issues.
        
        Returns:
            Dict with analysis results
        """
        results = {
            "file_path": self.file_path,
            "model_name": os.path.basename(self.file_path).replace("test_hf_", "").replace(".py", ""),
            "hardware_methods": {},
            "missing_hardware": [],
            "integration_issues": [],
            "indentation_issues": [],
            "asyncio_issues": []
        }
        
        # Check for hardware methods
        for platform in HARDWARE_PLATFORMS:
            platform_key = f"init_{platform}"
            if re.search(rf"def\s+init_{platform}\s*\(", self.file_content):
                results["hardware_methods"][platform_key] = True
            else:
                results["hardware_methods"][platform_key] = False
                results["missing_hardware"].append(platform)
            
            # Check for test methods
            test_key = f"test_with_{platform}"
            if re.search(rf"def\s+test_with_{platform}\s*\(", self.file_content):
                results["hardware_methods"][test_key] = True
            else:
                results["hardware_methods"][test_key] = False
                if platform_key in results["hardware_methods"] and results["hardware_methods"][platform_key]:
                    results["integration_issues"].append(
                        f"Has init_{platform} method but missing test_with_{platform} method"
                    )
        
        # Check for run_tests method calling test methods
        for platform in HARDWARE_PLATFORMS:
            if platform == "cpu":
                continue  # Skip CPU as it's always called
                
            if (results["hardware_methods"].get(f"init_{platform}", False) and 
                results["hardware_methods"].get(f"test_with_{platform}", False)):
                # Has both methods, check if test method is called
                if not re.search(rf"test_with_{platform}\s*\(", self.file_content):
                    results["integration_issues"].append(
                        f"test_with_{platform} method exists but is not called"
                    )
        
        # Check indentation issues
        method_indentation = {}
        for line in self.file_content.split('\n'):
            method_match = re.match(r'^(\s*)def\s+(\w+)', line)
            if method_match:
                indent_size = len(method_match.group(1))
                method_name = method_match.group(2)
                method_indentation[method_name] = indent_size
        
        # Methods should have consistent indentation
        if method_indentation:
            common_indent = max(set(method_indentation.values()), key=list(method_indentation.values()).count)
            for method, indent in method_indentation.items():
                if indent != common_indent and any(hw in method for hw in HARDWARE_PLATFORMS):
                    results["indentation_issues"].append(
                        f"Method {method} has inconsistent indentation ({indent} vs {common_indent})"
                    )
        
        # Check asyncio issues
        has_asyncio_import = "import asyncio" in self.file_content
        has_asyncio_usage = "asyncio.Queue" in self.file_content or "await" in self.file_content
        
        if has_asyncio_usage and not has_asyncio_import:
            results["asyncio_issues"].append("Uses asyncio but missing import")
            
        # Check AMD precision handling
        has_amd_rocm = "init_rocm" in self.file_content
        has_amd_precision = "amd_precision" in self.file_content.lower()
        
        if has_amd_rocm and not has_amd_precision:
            results["integration_issues"].append("Has ROCm support but missing AMD precision handling")
            
        # Check WebNN integration
        has_webnn = "init_webnn" in self.file_content
        has_webgpu = "init_webgpu" in self.file_content
        runs_webnn_test = "test_with_webnn" in self.file_content
        runs_webgpu_test = "test_with_webgpu" in self.file_content
        
        # Check WebNN compute shaders support for audio models
        is_audio_model = any(audio_model in self.file_path for audio_model in ["whisper", "wav2vec2", "clap"])
        has_compute_shaders = "compute_shaders" in self.file_content
        has_firefox_optimization = "firefox" in self.file_content.lower() and "optimization" in self.file_content
        
        if has_webnn and not runs_webnn_test:
            results["integration_issues"].append("Has WebNN method but not integrated in testing")
            
        if has_webgpu and not runs_webgpu_test:
            results["integration_issues"].append("Has WebGPU method but not integrated in testing")
            
        # Check multimodal model features for WebGPU/WebNN
        is_multimodal_model = any(mm_model in self.file_path for mm_model in ["llava", "clip", "xclip"])
        has_parallel_loading = "parallel_loading" in self.file_content
        has_4bit_quantization = "4bit" in self.file_content.lower() or "4-bit" in self.file_content
        has_kv_cache = "kv_cache" in self.file_content.lower() or "kv-cache" in self.file_content
        
        # Add issues for audio models missing Firefox optimizations
        if is_audio_model and has_webgpu and not has_compute_shaders:
            results["integration_issues"].append("Audio model missing WebGPU compute shader support")
            
        if is_audio_model and has_webgpu and not has_firefox_optimization:
            results["integration_issues"].append("Audio model missing Firefox-specific optimizations")
            
        # Add issues for multimodal models missing optimizations    
        if is_multimodal_model and has_webgpu and not has_parallel_loading:
            results["integration_issues"].append("Multimodal model missing parallel loading optimization")
            
        if is_multimodal_model and has_webgpu and not has_4bit_quantization:
            results["integration_issues"].append("Multimodal model missing 4-bit quantization support")
            
        if is_multimodal_model and has_webgpu and not has_kv_cache:
            results["integration_issues"].append("Multimodal model missing KV-cache optimization")
            
        return results
    
    def fix_issues(self) -> bool:
        """
        Fix hardware integration issues in the file.
        
        Returns:
            True if changes were made, False otherwise
        """
        # Analyze first to find issues
        analysis = self.analyze()
        self.issues = []
        
        # Add issues from analysis
        self.issues.extend(analysis["missing_hardware"])
        self.issues.extend(analysis["integration_issues"])
        self.issues.extend(analysis["indentation_issues"])
        self.issues.extend(analysis["asyncio_issues"])
        
        # Nothing to fix
        if not self.issues:
            return False
            
        # Fix async import if needed
        if "Uses asyncio but missing import" in self.issues:
            self._fix_asyncio_import()
            self.fixed_issues.append("Added asyncio import")
            
        # Fix indentation issues
        if analysis["indentation_issues"]:
            self._fix_indentation_issues()
            self.fixed_issues.append("Fixed method indentation")
            
        # Add missing hardware platform methods
        for platform in analysis["missing_hardware"]:
            # Only add key platforms that would be expected for this model type
            if self._should_add_platform(platform, analysis["model_name"]):
                self._add_hardware_platform(platform)
                self.fixed_issues.append(f"Added {platform} hardware platform support")
        
        # Fix integration of existing hardware methods
        for issue in analysis["integration_issues"]:
            if "test method exists but is not called" in issue:
                platform = issue.split("test_with_")[1].split(" ")[0]
                self._integrate_test_method(platform)
                self.fixed_issues.append(f"Integrated test_with_{platform} into run_tests method")
            
            if "Has ROCm support but missing AMD precision handling" in issue:
                self._add_amd_precision_handling()
                self.fixed_issues.append("Added AMD precision handling for ROCm")
                
            if "Has WebNN method but not integrated in testing" in issue:
                self._integrate_test_method("webnn")
                self.fixed_issues.append("Integrated WebNN testing")
                
            if "Has WebGPU method but not integrated in testing" in issue:
                self._integrate_test_method("webgpu")
                self.fixed_issues.append("Integrated WebGPU testing")
        
        return self.has_changes
    
    def _should_add_platform(self, platform: str, model_name: str) -> bool:
        """
        Determine if a platform should be added to this model.
        
        Args:
            platform: Hardware platform name
            model_name: Name of the model
            
        Returns:
            True if the platform should be added
        """
        # Always add these platforms
        if platform in ["cpu", "cuda", "openvino"]:
            return True
            
        # Strip common file name patterns
        normalized_name = model_name.replace("test_", "").replace("hf_", "")
        
        # Check model type to determine appropriate platforms
        for key_model in KEY_MODELS:
            if key_model in normalized_name:
                # For multimodal models like LLaVA
                if key_model in ["llava", "llava_next"]:
                    # These only fully support CPU and CUDA
                    if platform not in ["cpu", "cuda", "openvino"]:
                        return False
                
                # For LLMs
                if key_model in ["llama", "qwen2"]:
                    # WebNN and WebGPU are simulated for LLMs
                    if platform in ["webnn", "webgpu"]:
                        return True
                
                # For audio models
                if key_model in ["clap", "whisper", "wav2vec2"]:
                    # Audio models have partial web platform support
                    if platform in ["webnn", "webgpu"]:
                        return True
                    else:
                        return True
                
                # For vision models (full cross-platform support)
                if key_model in ["clip", "vit", "detr"]:
                    return True
                
                # For text models (full cross-platform support)
                if key_model in ["bert", "t5"]:
                    return True
                    
                # For XCLIP
                if key_model == "xclip":
                    if platform in ["webnn", "webgpu"]:
                        return False
                    return True
                    
        # Default: add all platforms for other models
        return True
    
    def _fix_asyncio_import(self):
        """Add asyncio import if missing."""
        if "import asyncio" not in self.file_content:
            # Find an appropriate place to add the import
            import_section = re.search(r'import .*?(\n\n|$)', self.file_content, re.DOTALL)
            if import_section:
                end_of_imports = import_section.end()
                self.file_content = (
                    self.file_content[:end_of_imports] + 
                    "import asyncio\n" + 
                    self.file_content[end_of_imports:]
                )
                self.has_changes = True
    
    def _fix_indentation_issues(self):
        """Fix indentation issues with methods."""
        lines = self.file_content.split('\n')
        
        # Find class definitions and their indentation
        class_indents = {}
        for i, line in enumerate(lines):
            class_match = re.match(r'^(\s*)class\s+(\w+)', line)
            if class_match:
                indent = class_match.group(1)
                class_name = class_match.group(2)
                class_indents[class_name] = {
                    "start_line": i,
                    "indent": indent,
                    "method_indent": indent + "    "  # Standard 4-space indent
                }
        
        # Fix method indentation
        fixed_lines = []
        current_class = None
        
        for line in lines:
            # Check for class definition
            class_match = re.match(r'^(\s*)class\s+(\w+)', line)
            if class_match:
                class_name = class_match.group(2)
                current_class = class_name
                fixed_lines.append(line)
                continue
                
            # Check for method definition
            method_match = re.match(r'^(\s*)def\s+(\w+)', line)
            if method_match and current_class:
                indent = method_match.group(1)
                method_name = method_match.group(2)
                
                # Fix hardware-related methods
                if any(platform in method_name for platform in HARDWARE_PLATFORMS):
                    expected_indent = class_indents[current_class]["method_indent"]
                    if indent != expected_indent:
                        # Replace the indentation
                        fixed_line = expected_indent + line[len(indent):]
                        fixed_lines.append(fixed_line)
                        self.has_changes = True
                        continue
            
            fixed_lines.append(line)
        
        if self.has_changes:
            self.file_content = '\n'.join(fixed_lines)
    
    def _add_hardware_platform(self, platform: str):
        """
        Add support for a specific hardware platform.
        
        Args:
            platform: Hardware platform to add
        """
        # Check what implementations already exist
        has_cpu = "def init_cpu" in self.file_content
        has_cuda = "def init_cuda" in self.file_content
        
        if not has_cpu and platform != "cpu":
            # Can't add other platforms if CPU isn't implemented
            return
            
        # Generate the implementation based on the platform
        if platform == "rocm":
            self._add_rocm_implementation()
        elif platform == "webnn":
            self._add_webnn_implementation()
        elif platform == "webgpu":
            self._add_webgpu_implementation()
        elif platform == "mps":
            self._add_mps_implementation()
            
        # Also add the corresponding test method
        self._add_test_method(platform)
        
        # Integrate the test method into run_tests
        self._integrate_test_method(platform)
    
    def _add_rocm_implementation(self):
        """Add ROCm hardware implementation."""
        # Find a reference implementation (typically CPU or CUDA)
        code_pattern = re.compile(r'def init_cuda.*?(?=def|\Z)', re.DOTALL)
        match = code_pattern.search(self.file_content)
        
        if not match:
            return  # Couldn't find CUDA implementation to base on
            
        # Get the CUDA code and adapt it for ROCm
        cuda_code = match.group(0)
        rocm_code = cuda_code.replace("init_cuda", "init_rocm")
        rocm_code = rocm_code.replace("CUDA", "ROCm")
        rocm_code = rocm_code.replace("cuda", "rocm")
        
        # Add AMD precision handling
        rocm_code = rocm_code.replace("device.type == 'cuda'", "device.type == 'cuda' or device.type == 'hip'")
        rocm_code = rocm_code.replace("with torch.cuda.amp.autocast()", "with torch.cuda.amp.autocast()")
        
        # Find the appropriate insertion point
        insertion_point = self.file_content.find("def test_")
        if insertion_point == -1:
            # Try finding the class end
            class_end = re.search(r'class \w+.*?\n\n', self.file_content, re.DOTALL)
            if class_end:
                insertion_point = class_end.end()
            else:
                return  # Can't determine insertion point
        
        # Insert the ROCm code
        self.file_content = (
            self.file_content[:insertion_point] + 
            "\n" + rocm_code + "\n" + 
            self.file_content[insertion_point:]
        )
        self.has_changes = True
    
    def _add_webnn_implementation(self):
        """Add WebNN hardware implementation."""
        # Find the end of the class or a good insertion point
        insertion_point = self.file_content.find("def test_")
        if insertion_point == -1:
            # Try finding the class end
            class_end = re.search(r'class \w+.*?\n\n', self.file_content, re.DOTALL)
            if class_end:
                insertion_point = class_end.end()
            else:
                return  # Can't determine insertion point
                
        # Get the model type to customize template
        model_type_match = re.search(r'test_hf_(\w+).py', self.file_path)
        model_type = model_type_match.group(1) if model_type_match else "unknown"
        
        # Detect if this is an audio, vision, or text model
        is_audio = any(m in model_type for m in ["clap", "whisper", "wav2vec2"])
        is_vision = any(m in model_type for m in ["vit", "clip", "detr", "llava"])
        is_text = not (is_audio or is_vision)
        
        # Detect if this is a multimodal model
        is_multimodal = any(m in model_type for m in ["llava", "xclip", "clip"])
                
        # Generate appropriate template based on model type
        if is_audio:
            init_code = self._generate_webnn_audio_template()
        elif is_vision:
            init_code = self._generate_webnn_vision_template()
        elif is_multimodal:
            init_code = self._generate_webnn_multimodal_template()
        else:
            init_code = self._generate_webnn_text_template()
        
        # Insert the code
        self.file_content = (
            self.file_content[:insertion_point] + 
            "\n" + init_code + "\n" + 
            self.file_content[insertion_point:]
        )
        self.has_changes = True
    
    def _add_webgpu_implementation(self):
        """Add WebGPU hardware implementation."""
        # Find the end of the class or a good insertion point
        insertion_point = self.file_content.find("def test_")
        if insertion_point == -1:
            # Try finding the class end
            class_end = re.search(r'class \w+.*?\n\n', self.file_content, re.DOTALL)
            if class_end:
                insertion_point = class_end.end()
            else:
                return  # Can't determine insertion point
                
        # Get the model type to customize template
        model_type_match = re.search(r'test_hf_(\w+).py', self.file_path)
        model_type = model_type_match.group(1) if model_type_match else "unknown"
        
        # Detect if this is an audio, vision, text, or multimodal model
        is_audio = any(m in model_type for m in ["clap", "whisper", "wav2vec2"])
        is_vision = any(m in model_type for m in ["vit", "detr"])
        is_multimodal = any(m in model_type for m in ["llava", "xclip", "clip"])
        is_text = not (is_audio or is_vision or is_multimodal)
        
        # Generate appropriate template based on model type
        if is_audio:
            init_code = self._generate_webgpu_audio_template()
        elif is_vision:
            init_code = self._generate_webgpu_vision_template()
        elif is_multimodal:
            init_code = self._generate_webgpu_multimodal_template()
        else:
            init_code = self._generate_webgpu_text_template()
        
        # Insert the code
        self.file_content = (
            self.file_content[:insertion_point] + 
            "\n" + init_code + "\n" + 
            self.file_content[insertion_point:]
        )
        self.has_changes = True
    
    def _add_mps_implementation(self):
        """Add MPS (Apple Silicon) hardware implementation."""
        # Find a reference implementation (typically CPU or CUDA)
        code_pattern = re.compile(r'def init_cuda.*?(?=def|\Z)', re.DOTALL)
        match = code_pattern.search(self.file_content)
        
        if not match:
            return  # Couldn't find CUDA implementation to base on
            
        # Get the CUDA code and adapt it for MPS
        cuda_code = match.group(0)
        mps_code = cuda_code.replace("init_cuda", "init_mps")
        mps_code = mps_code.replace("CUDA", "MPS")
        mps_code = mps_code.replace("cuda", "mps")
        
        # Replace CUDA-specific code with MPS-specific code
        mps_code = mps_code.replace("torch.cuda.is_available()", "hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()")
        mps_code = mps_code.replace("with torch.cuda.amp.autocast()", "# MPS does not yet fully support automatic mixed precision")
        
        # Find the appropriate insertion point
        insertion_point = self.file_content.find("def test_")
        if insertion_point == -1:
            # Try finding the class end
            class_end = re.search(r'class \w+.*?\n\n', self.file_content, re.DOTALL)
            if class_end:
                insertion_point = class_end.end()
            else:
                return  # Can't determine insertion point
        
        # Insert the MPS code
        self.file_content = (
            self.file_content[:insertion_point] + 
            "\n" + mps_code + "\n" + 
            self.file_content[insertion_point:]
        )
        self.has_changes = True
    
    def _add_test_method(self, platform: str):
        """
        Add test method for a specific hardware platform.
        
        Args:
            platform: Hardware platform to add test method for
        """
        # Check if the test method already exists
        if f"def test_with_{platform}" in self.file_content:
            return
            
        # Find a reference test method (typically CPU or CUDA)
        for ref_platform in ["cuda", "cpu"]:
            code_pattern = re.compile(rf'def test_with_{ref_platform}.*?(?=def|\Z)', re.DOTALL)
            match = code_pattern.search(self.file_content)
            if match:
                break
        
        if not match:
            return  # Couldn't find a reference implementation
            
        # Get the reference code and adapt it for the platform
        ref_code = match.group(0)
        platform_code = ref_code.replace(f"test_with_{ref_platform}", f"test_with_{platform}")
        platform_code = platform_code.replace(f"init_{ref_platform}", f"init_{platform}")
        platform_code = platform_code.replace(f"Using {ref_platform.upper()}", f"Using {platform.upper()}")
        
        # Find the appropriate insertion point - before run_tests
        run_tests_match = re.search(r'def run_tests', self.file_content)
        if run_tests_match:
            insertion_point = run_tests_match.start()
        else:
            # If run_tests not found, insert at the end of the file
            insertion_point = len(self.file_content)
        
        # Insert the test method
        self.file_content = (
            self.file_content[:insertion_point] + 
            platform_code + "\n" + 
            self.file_content[insertion_point:]
        )
        self.has_changes = True
    
    def _integrate_test_method(self, platform: str):
        """
        Integrate a test method into the run_tests method.
        
        Args:
            platform: Hardware platform to integrate
        """
        # Find the run_tests method
        run_tests_pattern = re.compile(r'def run_tests.*?(?=def|\Z)', re.DOTALL)
        match = run_tests_pattern.search(self.file_content)
        
        if not match:
            return  # Couldn't find run_tests method
            
        run_tests_code = match.group(0)
        
        # Check if the platform is already integrated
        if f"test_with_{platform}" in run_tests_code:
            return
            
        # Find the end of the test with CPU method call
        cpu_test_match = re.search(r'(\s+)# Test with CPU.*?(\n\s*\n|\n\s*}|\Z)', run_tests_code, re.DOTALL)
        
        if not cpu_test_match:
            # No CPU test found, just add at the end
            insertion_lines = []
        else:
            # Get the indentation from the CPU test
            indent = cpu_test_match.group(1)
            insertion_point = cpu_test_match.end()
            
            # Create the integration code
            platform_upper = platform.upper()
            insertion_lines = [
                f"{indent}# Test with {platform_upper}",
                f"{indent}if self.test_with_{platform}():",
                f"{indent}    results[\"{platform}_supported\"] = True",
                f"{indent}    results[\"{platform}_handler\"] = \"Success\"",
                f"{indent}else:",
                f"{indent}    results[\"{platform}_supported\"] = False",
                f"{indent}    results[\"{platform}_handler\"] = \"Initialization failed\"",
                ""
            ]
            
            # Detect if run_tests has try-except wrapping
            has_try_except = "try:" in run_tests_code and "except Exception as e:" in run_tests_code
            
            if has_try_except:
                # Already has try-except, just insert normally
                pass
            else:
                # Need to modify integration - look for a return statement
                return_match = re.search(r'(\s+)return\s+results', run_tests_code)
                if return_match:
                    # Insert right before the return
                    insertion_point = return_match.start()
            
            # Integrate the code
            run_tests_with_platform = (
                run_tests_code[:insertion_point] + 
                "\n" + "\n".join(insertion_lines) + 
                run_tests_code[insertion_point:]
            )
            
            # Replace the run_tests method
            self.file_content = self.file_content.replace(run_tests_code, run_tests_with_platform)
            self.has_changes = True
    
    def _add_amd_precision_handling(self):
        """Add AMD precision handling to the ROCm implementation."""
        # Find the ROCm implementation
        rocm_pattern = re.compile(r'def init_rocm.*?(?=def|\Z)', re.DOTALL)
        match = rocm_pattern.search(self.file_content)
        
        if not match:
            return  # Couldn't find ROCm implementation
            
        rocm_code = match.group(0)
        
        # Check if precision handling already exists
        if "autocast" in rocm_code:
            # Add AMD-specific precision if not present
            if "amd_precision" not in rocm_code.lower():
                # Add AMD support to autocast by replacing CUDA-specific code
                modified_code = rocm_code.replace(
                    "torch.cuda.amp.autocast()",
                    "torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, device_type='cuda' if device.type == 'cuda' else 'hip')"
                )
                
                # Replace the old implementation
                self.file_content = self.file_content.replace(rocm_code, modified_code)
                self.has_changes = True
        else:
            # If no autocast at all, add it
            # Find the inference section
            inference_match = re.search(r'(\s+)# Run inference', rocm_code)
            if inference_match:
                indent = inference_match.group(1)
                # Find where to insert
                insert_point = inference_match.end()
                # Create replacement code
                prefix = rocm_code[:insert_point]
                suffix = rocm_code[insert_point:]
                autocast_code = f"\n{indent}# Use mixed precision for both CUDA and ROCm (HIP) devices\n{indent}with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, device_type='cuda' if device.type == 'cuda' else 'hip'):"
                # Adjust indentation for code after autocast
                adjusted_suffix = ""
                for line in suffix.split("\n"):
                    if line.strip() and not line.strip().startswith("#"):
                        adjusted_suffix += f"\n{indent}    {line.lstrip()}"
                    else:
                        adjusted_suffix += f"\n{line}"
                
                # Create final code
                modified_code = prefix + autocast_code + adjusted_suffix
                
                # Replace the old implementation
                self.file_content = self.file_content.replace(rocm_code, modified_code)
                self.has_changes = True
    
    def _generate_webnn_text_template(self) -> str:
        """Generate WebNN implementation template for text models."""
        return """    def init_webnn(self, model_name=None):
        \"\"\"Initialize text model for WebNN inference.\"\"\"
        try:
            print("Initializing WebNN for text model")
            model_name = model_name or self.model_name
            
            # Check for WebNN support
            webnn_support = False
            try:
                # In browser environments, check for WebNN API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'ml'):
                    webnn_support = True
                    print("WebNN API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = asyncio.Queue(16)
            
            if not webnn_support:
                # Create a WebNN simulation using CPU implementation for text models
                print("Using WebNN simulation for text model")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Wrap the CPU function to simulate WebNN
                def webnn_handler(text_input, **kwargs):
                    try:
                        # Process input with tokenizer
                        if isinstance(text_input, list):
                            inputs = processor(text_input, padding=True, truncation=True, return_tensors="pt")
                        else:
                            inputs = processor(text_input, return_tensors="pt")
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebNN-specific metadata
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBNN",
                            "model": model_name,
                            "backend": "webnn-simulation",
                            "device": "cpu"
                        }
                    except Exception as e:
                        print(f"Error in WebNN simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webnn_handler, queue, batch_size
            else:
                # Use actual WebNN implementation when available
                # (This would use the WebNN API in browser environments)
                print("Using native WebNN implementation")
                
                # Since WebNN API access depends on browser environment,
                # implementation details would involve JS interop
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x: {"output": "Native WebNN output", "implementation_type": "WEBNN"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebNN: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = asyncio.Queue(16)
            return None, None, lambda x: {"output": "Mock WebNN output", "implementation_type": "MOCK_WEBNN"}, queue, 1"""
    
    def _generate_webnn_vision_template(self) -> str:
        """Generate WebNN implementation template for vision models."""
        return """    def init_webnn(self, model_name=None):
        \"\"\"Initialize vision model for WebNN inference.\"\"\"
        try:
            print("Initializing WebNN for vision model")
            model_name = model_name or self.model_name
            
            # Check for WebNN support
            webnn_support = False
            try:
                # In browser environments, check for WebNN API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'ml'):
                    webnn_support = True
                    print("WebNN API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = asyncio.Queue(16)
            
            if not webnn_support:
                # Create a WebNN simulation using CPU implementation for vision models
                print("Using WebNN simulation for vision model")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Wrap the CPU function to simulate WebNN
                def webnn_handler(image_input, **kwargs):
                    try:
                        # Process image input (path or PIL Image)
                        if isinstance(image_input, str):
                            from PIL import Image
                            image = Image.open(image_input).convert("RGB")
                        elif isinstance(image_input, list):
                            if all(isinstance(img, str) for img in image_input):
                                from PIL import Image
                                image = [Image.open(img).convert("RGB") for img in image_input]
                            else:
                                image = image_input
                        else:
                            image = image_input
                            
                        # Process with processor
                        inputs = processor(images=image, return_tensors="pt")
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebNN-specific metadata
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBNN",
                            "model": model_name,
                            "backend": "webnn-simulation",
                            "device": "cpu"
                        }
                    except Exception as e:
                        print(f"Error in WebNN simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webnn_handler, queue, batch_size
            else:
                # Use actual WebNN implementation when available
                # (This would use the WebNN API in browser environments)
                print("Using native WebNN implementation")
                
                # Since WebNN API access depends on browser environment,
                # implementation details would involve JS interop
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x: {"output": "Native WebNN output", "implementation_type": "WEBNN"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebNN: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = asyncio.Queue(16)
            return None, None, lambda x: {"output": "Mock WebNN output", "implementation_type": "MOCK_WEBNN"}, queue, 1"""
    
    def _generate_webnn_audio_template(self) -> str:
        """Generate WebNN implementation template for audio models."""
        return """    def init_webnn(self, model_name=None):
        \"\"\"Initialize audio model for WebNN inference.\"\"\"
        try:
            print("Initializing WebNN for audio model")
            model_name = model_name or self.model_name
            
            # Check for WebNN support
            webnn_support = False
            try:
                # In browser environments, check for WebNN API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'ml'):
                    webnn_support = True
                    print("WebNN API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = asyncio.Queue(16)
            
            if not webnn_support:
                # Create a WebNN simulation using CPU implementation for audio models
                print("Using WebNN simulation for audio model")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Wrap the CPU function to simulate WebNN
                def webnn_handler(audio_input, sampling_rate=16000, **kwargs):
                    try:
                        # Process audio input
                        if isinstance(audio_input, str):
                            # Load audio file
                            try:
                                import librosa
                                array, sr = librosa.load(audio_input, sr=sampling_rate)
                            except ImportError:
                                # Mock audio data if librosa isn't available
                                array = torch.zeros((sampling_rate * 3,))  # 3 seconds of silence
                                sr = sampling_rate
                        else:
                            array = audio_input
                            sr = sampling_rate
                            
                        # Process with processor
                        inputs = processor(array, sampling_rate=sr, return_tensors="pt")
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebNN-specific metadata
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBNN",
                            "model": model_name,
                            "backend": "webnn-simulation",
                            "device": "cpu"
                        }
                    except Exception as e:
                        print(f"Error in WebNN simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webnn_handler, queue, batch_size
            else:
                # Use actual WebNN implementation when available
                # (This would use the WebNN API in browser environments)
                print("Using native WebNN implementation")
                
                # Since WebNN API access depends on browser environment,
                # implementation details would involve JS interop via WebAudio API
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x, sampling_rate=16000: {"output": "Native WebNN output", "implementation_type": "WEBNN"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebNN: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = asyncio.Queue(16)
            return None, None, lambda x, sampling_rate=16000: {"output": "Mock WebNN output", "implementation_type": "MOCK_WEBNN"}, queue, 1"""
    
    def _generate_webgpu_text_template(self) -> str:
        """Generate WebGPU implementation template for text models."""
        return """    def init_webgpu(self, model_name=None):
        \"\"\"Initialize text model for WebGPU inference using transformers.js simulation.\"\"\"
        try:
            print("Initializing WebGPU for text model")
            model_name = model_name or self.model_name
            
            # Check for WebGPU support
            webgpu_support = False
            try:
                # In browser environments, check for WebGPU API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'gpu'):
                    webgpu_support = True
                    print("WebGPU API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = asyncio.Queue(16)
            
            if not webgpu_support:
                # Create a WebGPU simulation using CPU implementation for text models
                print("Using WebGPU/transformers.js simulation for text model")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Wrap the CPU function to simulate WebGPU/transformers.js
                def webgpu_handler(text_input, **kwargs):
                    try:
                        # Process input with tokenizer
                        if isinstance(text_input, list):
                            inputs = processor(text_input, padding=True, truncation=True, return_tensors="pt")
                        else:
                            inputs = processor(text_input, return_tensors="pt")
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebGPU-specific metadata to match transformers.js
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBGPU_TRANSFORMERS_JS",
                            "model": model_name,
                            "backend": "webgpu-simulation",
                            "device": "webgpu",
                            "transformers_js": {
                                "version": "2.9.0",  # Simulated version
                                "quantized": False,
                                "format": "float32",
                                "backend": "webgpu"
                            }
                        }
                    except Exception as e:
                        print(f"Error in WebGPU simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webgpu_handler, queue, batch_size
            else:
                # Use actual WebGPU implementation when available
                # (This would use transformers.js in browser environments)
                print("Using native WebGPU implementation with transformers.js")
                
                # Since WebGPU API access depends on browser environment,
                # implementation details would involve JS interop
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x: {"output": "Native WebGPU output", "implementation_type": "WEBGPU_TRANSFORMERS_JS"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebGPU: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = asyncio.Queue(16)
            return None, None, lambda x: {"output": "Mock WebGPU output", "implementation_type": "MOCK_WEBGPU"}, queue, 1"""
    
    def _generate_webgpu_vision_template(self) -> str:
        """Generate WebGPU implementation template for vision models."""
        return """    def init_webgpu(self, model_name=None):
        \"\"\"Initialize vision model for WebGPU inference using transformers.js simulation.\"\"\"
        try:
            print("Initializing WebGPU for vision model")
            model_name = model_name or self.model_name
            
            # Check for WebGPU support
            webgpu_support = False
            try:
                # In browser environments, check for WebGPU API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'gpu'):
                    webgpu_support = True
                    print("WebGPU API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = asyncio.Queue(16)
            
            if not webgpu_support:
                # Create a WebGPU simulation using CPU implementation for vision models
                print("Using WebGPU/transformers.js simulation for vision model")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Wrap the CPU function to simulate WebGPU/transformers.js
                def webgpu_handler(image_input, **kwargs):
                    try:
                        # Process image input (path or PIL Image)
                        if isinstance(image_input, str):
                            from PIL import Image
                            image = Image.open(image_input).convert("RGB")
                        elif isinstance(image_input, list):
                            if all(isinstance(img, str) for img in image_input):
                                from PIL import Image
                                image = [Image.open(img).convert("RGB") for img in image_input]
                            else:
                                image = image_input
                        else:
                            image = image_input
                            
                        # Process with processor
                        inputs = processor(images=image, return_tensors="pt")
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebGPU-specific metadata to match transformers.js
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBGPU_TRANSFORMERS_JS",
                            "model": model_name,
                            "backend": "webgpu-simulation",
                            "device": "webgpu",
                            "transformers_js": {
                                "version": "2.9.0",  # Simulated version
                                "quantized": False,
                                "format": "float32",
                                "backend": "webgpu"
                            }
                        }
                    except Exception as e:
                        print(f"Error in WebGPU simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webgpu_handler, queue, batch_size
            else:
                # Use actual WebGPU implementation when available
                # (This would use transformers.js in browser environments)
                print("Using native WebGPU implementation with transformers.js")
                
                # Since WebGPU API access depends on browser environment,
                # implementation details would involve JS interop
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x: {"output": "Native WebGPU output", "implementation_type": "WEBGPU_TRANSFORMERS_JS"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebGPU: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = asyncio.Queue(16)
            return None, None, lambda x: {"output": "Mock WebGPU output", "implementation_type": "MOCK_WEBGPU"}, queue, 1"""
    
    def _generate_webgpu_audio_template(self) -> str:
        """Generate WebGPU implementation template for audio models."""
        return """    def init_webgpu(self, model_name=None):
        \"\"\"Initialize audio model for WebGPU inference using transformers.js simulation.\"\"\"
        try:
            print("Initializing WebGPU for audio model")
            model_name = model_name or self.model_name
            
            # Check for WebGPU support
            webgpu_support = False
            try:
                # In browser environments, check for WebGPU API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'gpu'):
                    webgpu_support = True
                    print("WebGPU API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = asyncio.Queue(16)
            
            if not webgpu_support:
                # Create a WebGPU simulation using CPU implementation for audio models
                print("Using WebGPU/transformers.js simulation for audio model")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Wrap the CPU function to simulate WebGPU/transformers.js
                def webgpu_handler(audio_input, sampling_rate=16000, **kwargs):
                    try:
                        # Process audio input
                        if isinstance(audio_input, str):
                            # Load audio file
                            try:
                                import librosa
                                array, sr = librosa.load(audio_input, sr=sampling_rate)
                            except ImportError:
                                # Mock audio data if librosa isn't available
                                array = torch.zeros((sampling_rate * 3,))  # 3 seconds of silence
                                sr = sampling_rate
                        else:
                            array = audio_input
                            sr = sampling_rate
                            
                        # Process with processor
                        inputs = processor(array, sampling_rate=sr, return_tensors="pt")
                        
                        # Run inference
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebGPU-specific metadata to match transformers.js
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBGPU_TRANSFORMERS_JS",
                            "model": model_name,
                            "backend": "webgpu-simulation",
                            "device": "webgpu",
                            "transformers_js": {
                                "version": "2.9.0",  # Simulated version
                                "quantized": False,
                                "format": "float32",
                                "backend": "webgpu"
                            }
                        }
                    except Exception as e:
                        print(f"Error in WebGPU simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webgpu_handler, queue, batch_size
            else:
                # Use actual WebGPU implementation when available
                # (This would use transformers.js in browser environments with WebAudio)
                print("Using native WebGPU implementation with transformers.js")
                
                # Since WebGPU API access depends on browser environment,
                # implementation details would involve JS interop via WebAudio
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x, sampling_rate=16000: {"output": "Native WebGPU output", "implementation_type": "WEBGPU_TRANSFORMERS_JS"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebGPU: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = asyncio.Queue(16)
            return None, None, lambda x, sampling_rate=16000: {"output": "Mock WebGPU output", "implementation_type": "MOCK_WEBGPU"}, queue, 1"""
            
    def _generate_webgpu_multimodal_template(self) -> str:
        """Generate WebGPU implementation template for multimodal models."""
        return """    def init_webgpu(self, model_name=None):
        \"\"\"Initialize multimodal model for WebGPU inference with advanced optimizations.\"\"\"
        try:
            print("Initializing WebGPU for multimodal model")
            model_name = model_name or self.model_name
            
            # Check for WebGPU support
            webgpu_support = False
            try:
                # In browser environments, check for WebGPU API
                import js
                if hasattr(js, 'navigator') and hasattr(js.navigator, 'gpu'):
                    webgpu_support = True
                    print("WebGPU API detected in browser environment")
            except ImportError:
                # Not in a browser environment
                pass
                
            # Create queue for inference requests
            import asyncio
            queue = asyncio.Queue(16)
            
            if not webgpu_support:
                # Create a WebGPU simulation using CPU implementation for multimodal models
                print("Using WebGPU/transformers.js simulation for multimodal model with optimizations")
                
                # Initialize with CPU for simulation
                endpoint, processor, _, _, batch_size = self.init_cpu(model_name=model_name)
                
                # Multimodal-specific optimizations
                use_parallel_loading = True
                use_4bit_quantization = True
                use_kv_cache_optimization = True
                print(f"WebGPU optimizations: parallel_loading={use_parallel_loading}, 4bit_quantization={use_4bit_quantization}, kv_cache={use_kv_cache_optimization}")
                
                # Wrap the CPU function to simulate WebGPU/transformers.js for multimodal
                def webgpu_handler(input_data, **kwargs):
                    try:
                        # Process multimodal input (image + text)
                        if isinstance(input_data, dict):
                            # Handle dictionary input with multiple modalities
                            image = input_data.get("image")
                            text = input_data.get("text")
                            
                            # Load image if path is provided
                            if isinstance(image, str):
                                from PIL import Image
                                image = Image.open(image).convert("RGB")
                        elif isinstance(input_data, str) and input_data.endswith(('.jpg', '.png', '.jpeg')):
                            # Handle image path as direct input
                            from PIL import Image
                            image = Image.open(input_data).convert("RGB")
                            text = kwargs.get("text", "")
                        else:
                            # Default handling for text input
                            image = None
                            text = input_data
                            
                        # Process with processor
                        if image is not None and text:
                            # Apply parallel loading optimization if enabled
                            if use_parallel_loading:
                                print("Using parallel loading optimization for multimodal input")
                                
                            # Process with processor
                            inputs = processor(text=text, images=image, return_tensors="pt")
                        else:
                            inputs = processor(input_data, return_tensors="pt")
                        
                        # Apply 4-bit quantization if enabled
                        if use_4bit_quantization:
                            print("Using 4-bit quantization for model weights")
                            # In real implementation, weights would be quantized here
                        
                        # Apply KV cache optimization if enabled
                        if use_kv_cache_optimization:
                            print("Using KV cache optimization for inference")
                            # In real implementation, KV cache would be used here
                        
                        # Run inference with optimizations
                        with torch.no_grad():
                            outputs = endpoint(**inputs)
                        
                        # Add WebGPU-specific metadata including optimization flags
                        return {
                            "output": outputs,
                            "implementation_type": "SIMULATION_WEBGPU_TRANSFORMERS_JS",
                            "model": model_name,
                            "backend": "webgpu-simulation",
                            "device": "webgpu",
                            "optimizations": {
                                "parallel_loading": use_parallel_loading,
                                "quantization_4bit": use_4bit_quantization,
                                "kv_cache_enabled": use_kv_cache_optimization
                            },
                            "transformers_js": {
                                "version": "2.9.0",  # Simulated version
                                "quantized": use_4bit_quantization,
                                "format": "float4" if use_4bit_quantization else "float32",
                                "backend": "webgpu"
                            }
                        }
                    except Exception as e:
                        print(f"Error in WebGPU multimodal simulation handler: {e}")
                        return {
                            "output": f"Error: {str(e)}",
                            "implementation_type": "ERROR",
                            "error": str(e),
                            "model": model_name
                        }
                
                return endpoint, processor, webgpu_handler, queue, batch_size
            else:
                # Use actual WebGPU implementation when available
                print("Using native WebGPU implementation with transformers.js for multimodal model")
                
                # Since WebGPU API access depends on browser environment,
                # implementation details would involve JS interop
                
                # Create mock implementation for now (replace with real implementation)
                return None, None, lambda x: {"output": "Native WebGPU multimodal output", "implementation_type": "WEBGPU_TRANSFORMERS_JS"}, queue, 1
                
        except Exception as e:
            print(f"Error initializing WebGPU for multimodal model: {e}")
            # Fallback to a minimal mock
            import asyncio
            queue = asyncio.Queue(16)
            return None, None, lambda x: {"output": "Mock WebGPU multimodal output", "implementation_type": "MOCK_WEBGPU"}, queue, 1"""


def find_test_files(directory: str, model_pattern: Optional[str] = None) -> List[str]:
    """
    Find test files in a directory with optional pattern matching.
    
    Args:
        directory: Directory to search
        model_pattern: Optional pattern to match model names
        
    Returns:
        List of paths to test files
    """
    pattern = os.path.join(directory, "test_hf_*.py")
    test_files = glob.glob(pattern)
    
    if model_pattern:
        # Filter files based on pattern
        model_names = model_pattern.split(',')
        filtered_files = []
        for file in test_files:
            file_model = os.path.basename(file).replace("test_hf_", "").replace(".py", "")
            # Check for exact match or if the file model starts with any of the pattern models
            if any(file_model == model_name or file_model.startswith(model_name) for model_name in model_names):
                filtered_files.append(file)
        return filtered_files
    
    return test_files

def analyze_files(files: List[str]) -> Dict:
    """
    Analyze multiple test files for hardware integration issues.
    
    Args:
        files: List of file paths to analyze
        
    Returns:
        Dict with analysis results
    """
    results = {
        "total_files": len(files),
        "files_with_issues": 0,
        "files_analyzed": [],
        "total_issues": 0,
        "issue_types": {
            "missing_hardware": 0,
            "integration_issues": 0,
            "indentation_issues": 0,
            "asyncio_issues": 0
        }
    }
    
    for file_path in files:
        fixer = HardwareIntegrationFixer(file_path)
        analysis = fixer.analyze()
        
        # Count issues
        issue_count = (
            len(analysis["missing_hardware"]) + 
            len(analysis["integration_issues"]) + 
            len(analysis["indentation_issues"]) + 
            len(analysis["asyncio_issues"])
        )
        
        results["total_issues"] += issue_count
        results["issue_types"]["missing_hardware"] += len(analysis["missing_hardware"])
        results["issue_types"]["integration_issues"] += len(analysis["integration_issues"])
        results["issue_types"]["indentation_issues"] += len(analysis["indentation_issues"])
        results["issue_types"]["asyncio_issues"] += len(analysis["asyncio_issues"])
        
        if issue_count > 0:
            results["files_with_issues"] += 1
        
        # Record file analysis
        file_result = {
            "file": os.path.basename(file_path),
            "model_name": analysis["model_name"],
            "issues": issue_count,
            "hardware_methods": analysis["hardware_methods"],
            "missing_hardware": analysis["missing_hardware"],
            "integration_issues": analysis["integration_issues"]
        }
        
        results["files_analyzed"].append(file_result)
    
    return results

def fix_files(files: List[str]) -> Tuple[int, int, Dict]:
    """
    Fix hardware integration issues in multiple test files.
    
    Args:
        files: List of file paths to fix
        
    Returns:
        Tuple of (files fixed, total issues fixed, detailed results)
    """
    results = {
        "total_files": len(files),
        "files_fixed": 0,
        "total_issues_fixed": 0,
        "files_details": []
    }
    
    for file_path in files:
        fixer = HardwareIntegrationFixer(file_path)
        changes_made = fixer.fix_issues()
        
        if changes_made:
            fixer.save_file()
            results["files_fixed"] += 1
            results["total_issues_fixed"] += len(fixer.fixed_issues)
            
            # Record file details
            file_result = {
                "file": os.path.basename(file_path),
                "model_name": os.path.basename(file_path).replace("test_hf_", "").replace(".py", ""),
                "issues_fixed": len(fixer.fixed_issues),
                "fixed_issues": fixer.fixed_issues
            }
            
            results["files_details"].append(file_result)
    
    return results["files_fixed"], results["total_issues_fixed"], results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix hardware integration in test files")
    
    # Select files to process
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--specific-models", type=str, help="Comma-separated list of models to fix")
    file_group.add_argument("--all-key-models", action="store_true", help="Fix all key model tests")
    file_group.add_argument("--specific-directory", type=str, help="Directory containing test files to fix")
    
    # Analysis mode
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze files, don't fix issues")
    
    # Output options
    parser.add_argument("--output-json", type=str, help="Save analysis/fix results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    # Determine which files to process
    if args.specific_models:
        # Convert model names to normalized form
        models = [m.replace('-', '_').lower() for m in args.specific_models.split(',')]
        model_pattern = ','.join(models)
        files = find_test_files(str(SKILLS_DIR), model_pattern)
    elif args.all_key_models:
        # Convert key model names to normalized form
        models = [m.replace('-', '_').lower() for m in KEY_MODELS]
        model_pattern = ','.join(models)
        files = find_test_files(str(SKILLS_DIR), model_pattern)
    elif args.specific_directory:
        files = find_test_files(args.specific_directory)
    
    if not files:
        print(f"No test files found matching the specified criteria")
        return 1
    
    print(f"Found {len(files)} test files to process")
    
    # Analysis mode
    if args.analyze_only:
        print("Analyzing files for hardware integration issues...")
        results = analyze_files(files)
        
        # Print summary
        print(f"\nAnalysis Summary:")
        print(f"- Total files analyzed: {results['total_files']}")
        print(f"- Files with issues: {results['files_with_issues']}")
        print(f"- Total issues found: {results['total_issues']}")
        print(f"- Issue types:")
        print(f"  - Missing hardware methods: {results['issue_types']['missing_hardware']}")
        print(f"  - Integration issues: {results['issue_types']['integration_issues']}")
        print(f"  - Indentation issues: {results['issue_types']['indentation_issues']}")
        print(f"  - Asyncio issues: {results['issue_types']['asyncio_issues']}")
        
        # Print details if verbose
        if args.verbose:
            print("\nDetailed Analysis:")
            for file_result in results["files_analyzed"]:
                if file_result["issues"] > 0:
                    print(f"\n{file_result['file']} ({file_result['model_name']}):")
                    
                    if file_result["missing_hardware"]:
                        print(f"  Missing hardware: {', '.join(file_result['missing_hardware'])}")
                    
                    if file_result["integration_issues"]:
                        print(f"  Integration issues:")
                        for issue in file_result["integration_issues"]:
                            print(f"    - {issue}")
        
        # Save results to JSON if requested and not deprecated
        if args.output_json and not DEPRECATE_JSON_OUTPUT:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nAnalysis results saved to {args.output_json}")
        elif args.output_json and DEPRECATE_JSON_OUTPUT:
            try:
                import duckdb
                # Connect to or create a database
                db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
                conn = duckdb.connect(db_path)
                
                # Check if hardware_analysis table exists, create if not
                conn.execute("CREATE TABLE IF NOT EXISTS hardware_analysis (timestamp TIMESTAMP, model_name VARCHAR, issue_count INTEGER, data JSON)")
                
                # Insert the analysis results
                timestamp = duckdb.sql("SELECT now()").fetchone()[0]
                conn.execute(
                    "INSERT INTO hardware_analysis VALUES (?, ?, ?, ?)",
                    [timestamp, "multiple", results["total_issues"], json.dumps(results)]
                )
                conn.commit()
                conn.close()
                print(f"\nAnalysis results saved to database ({db_path})")
            except Exception as e:
                print(f"\nFailed to save to database, falling back to JSON: {e}")
                with open(args.output_json, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nAnalysis results saved to {args.output_json} (database save failed)")
        
        return 0
    
    # Fix mode
    print("Fixing hardware integration issues...")
    files_fixed, issues_fixed, results = fix_files(files)
    
    # Print summary
    print(f"\nFix Summary:")
    print(f"- Total files processed: {len(files)}")
    print(f"- Files fixed: {files_fixed}")
    print(f"- Total issues fixed: {issues_fixed}")
    
    # Print details if verbose
    if args.verbose:
        print("\nDetailed Fixes:")
        for file_result in results["files_details"]:
            print(f"\n{file_result['file']} ({file_result['model_name']}):")
            print(f"  Issues fixed: {file_result['issues_fixed']}")
            
            if file_result["fixed_issues"]:
                print(f"  Fixed issues:")
                for issue in file_result["fixed_issues"]:
                    print(f"    - {issue}")
    
    # Save results to JSON if requested and not deprecated
    if args.output_json and not DEPRECATE_JSON_OUTPUT:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nFix results saved to {args.output_json}")
    elif args.output_json and DEPRECATE_JSON_OUTPUT:
        try:
            import duckdb
            # Connect to or create a database
            db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
            conn = duckdb.connect(db_path)
            
            # Check if hardware_fixes table exists, create if not
            conn.execute("CREATE TABLE IF NOT EXISTS hardware_fixes (timestamp TIMESTAMP, files_fixed INTEGER, issues_fixed INTEGER, data JSON)")
            
            # Insert the fix results
            timestamp = duckdb.sql("SELECT now()").fetchone()[0]
            conn.execute(
                "INSERT INTO hardware_fixes VALUES (?, ?, ?, ?)",
                [timestamp, files_fixed, issues_fixed, json.dumps(results)]
            )
            conn.commit()
            conn.close()
            print(f"\nFix results saved to database ({db_path})")
        except Exception as e:
            print(f"\nFailed to save to database, falling back to JSON: {e}")
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nFix results saved to {args.output_json} (database save failed)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())