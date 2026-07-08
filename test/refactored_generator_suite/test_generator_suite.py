#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced HuggingFace Model Test Generator

This module provides a comprehensive generator for creating hardware-aware test files
for HuggingFace Transformer models. It supports multiple hardware backends including
CPU, CUDA, ROCm (AMD GPU), OpenVINO, Apple MPS, and Qualcomm.

The generator creates test files that:
1. Detect available hardware
2. Select appropriate model implementations
3. Handle graceful degradation
4. Support specialized model architectures
5. Include comprehensive test methods

Usage:
    python test_generator_suite.py --model bert --output ./tests/test_bert.py
    python test_generator_suite.py --batch --architecture encoder-only --output-dir ./generated_tests/
"""

import os
import sys
import json
import time
import logging
import argparse
import importlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_generator_suite")

# Model architecture types
ARCHITECTURE_TYPES = [
    "encoder-only",
    "decoder-only",
    "encoder-decoder",
    "vision",
    "vision-text",
    "speech",
    "multimodal",
    "rag",
    "diffusion",
    "time-series",
    "graph",
    "object-detection",
    "mixture-of-experts",
    "state-space",
    "text-to-image",
    "protein-folding",
    "video-processing",
]

# Task types
TASK_TYPES = [
    "text-generation",
    "fill-mask",
    "text-classification",
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "feature-extraction",
    "image-classification",
    "image-segmentation",
    "object-detection",
    "audio-classification",
    "audio-to-text",
    "text-to-audio",
    "text-to-image",
    "image-to-text",
    "time-series-forecasting",
    "graph-classification",
    "node-classification",
    "protein-structure-prediction",
    "video-classification",
]

# Hardware backends
HARDWARE_BACKENDS = [
    "cpu",
    "cuda",
    "rocm",
    "openvino",
    "apple",
    "qualcomm",
]


class TestGeneratorSuite:
    """Generator for HuggingFace model tests with comprehensive hardware support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the test generator.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Import required modules if not already imported
        self._import_dependencies()
        
        # Load model registry
        self.model_registry = self._load_model_registry()
        
        # Load template registry
        self.template_registry = self._load_template_registry()
        
        # Load hardware detector
        self.hardware_detector = self._load_hardware_detector()
        
        # Initialize stats
        self.stats = {
            "generated": 0,
            "failed": 0,
            "skipped": 0,
            "models_by_architecture": {},
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            Configuration dictionary.
        """
        default_config = {
            "output_dir": "./generated_tests",
            "template_dir": "./templates",
            "model_registry_path": "./model_selection/registry.py",
            "default_model_type": "bert",
            "default_architecture": "encoder-only",
            "default_task": "fill-mask",
            "fix_syntax": True,
            "validate_output": True,
            "test_generated_files": False,
            "logging": {
                "level": "INFO",
                "file": None
            },
            "hardware": {
                "detect_automatically": True,
                "mock_unavailable": True,
                "hardware_priority": ["cuda", "rocm", "mps", "openvino", "cpu"]
            }
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    user_config = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    user_config = yaml.safe_load(f)
                else:
                    self.logger.warning(f"Unsupported config format: {config_path}")
                    return default_config
                
            # Merge configs
            merged_config = default_config.copy()
            self._deep_update(merged_config, user_config)
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return default_config
    
    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """Deep update dictionary d with values from dictionary u.
        
        Args:
            d: Dictionary to update.
            u: Dictionary with new values.
            
        Returns:
            Updated dictionary.
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def _import_dependencies(self) -> None:
        """Import required modules dynamically."""
        try:
            # Add modules to sys.path if needed
            repo_root = Path(__file__).parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            
            # Try to import core modules
            self.generator_core = importlib.import_module("generator_core")
            self.registry_module = importlib.import_module("generator_core.registry")
            self.model_generator_module = importlib.import_module("generators.model_generator")
            self.syntax_module = importlib.import_module("syntax.validator")
            
            self.logger.info("Successfully imported core dependencies")
        except ImportError as e:
            self.logger.error(f"Failed to import dependencies: {e}")
            raise
    
    def _load_model_registry(self) -> Any:
        """Load the model registry.
        
        Returns:
            Model registry instance.
        """
        try:
            # Try to import from the configured path
            registry_path = self.config.get("model_registry_path", "./model_selection/registry.py")
            
            if isinstance(registry_path, str) and registry_path.endswith('.py'):
                # Convert to module path
                module_path = registry_path.replace('/', '.').replace('.py', '')
                registry_module = importlib.import_module(module_path)
                registry_class = getattr(registry_module, "ModelRegistry", None)
                
                if registry_class:
                    return registry_class()
            
            # Fallback to registry in generator_core
            return self.registry_module.ComponentRegistry()
            
        except Exception as e:
            self.logger.error(f"Failed to load model registry: {e}")
            return self.registry_module.ComponentRegistry()
    
    def _load_template_registry(self) -> Dict[str, Any]:
        """Load the template registry.
        
        Returns:
            Template registry dictionary.
        """
        templates = {}
        template_dir = Path(self.config.get("template_dir", "./templates"))
        
        try:
            # Register base templates
            base_path = template_dir / "base.py"
            if base_path.exists():
                templates["base"] = self._load_template(base_path)
            
            # Register all templates in the directory
            for template_file in template_dir.glob("*.py"):
                if template_file.name == "base.py" or template_file.name == "__init__.py":
                    continue
                
                # Extract template name without extension
                template_name = template_file.stem
                
                # Add to registry
                templates[template_name] = self._load_template(template_file)
                
                # Also add with canonical architecture name if this is an architecture template
                for arch in ARCHITECTURE_TYPES:
                    arch_filename = arch.replace('-', '_')
                    if template_name == arch_filename or template_name == f"{arch_filename}_template":
                        templates[arch] = templates[template_name]
                        break
            
            # Special handling for reference template
            reference_path = template_dir / "hf_reference_template.py"
            if reference_path.exists():
                templates["reference"] = self._load_template(reference_path)
                
            # Check for specialized templates that we're interested in
            specialized_templates = [
                "text_to_image_template", 
                "protein_folding_template", 
                "video_processing_template"
            ]
            
            for specialized_template in specialized_templates:
                template_path = template_dir / f"{specialized_template}.py"
                if template_path.exists():
                    self.logger.info(f"Found specialized template: {specialized_template}")
                    templates[specialized_template] = self._load_template(template_path)
                    
                    # Map the specialized template to its canonical architecture name
                    if specialized_template == "text_to_image_template":
                        templates["text-to-image"] = templates[specialized_template]
                    elif specialized_template == "protein_folding_template":
                        templates["protein-folding"] = templates[specialized_template]
                    elif specialized_template == "video_processing_template":
                        templates["video-processing"] = templates[specialized_template]
                
            self.logger.info(f"Loaded {len(templates)} templates")
            return templates
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")
            return {}
    
    def _load_template(self, template_path: Path) -> str:
        """Load a template from file.
        
        Args:
            template_path: Path to template file.
            
        Returns:
            Template content as string.
        """
        with open(template_path, 'r') as f:
            return f.read()
    
    def _load_hardware_detector(self) -> Any:
        """Load the hardware detector.
        
        Returns:
            Hardware detector instance.
        """
        try:
            # Try to import hardware detector
            hardware_module = importlib.import_module("hardware.hardware_detection")
            detector_class = getattr(hardware_module, "HardwareDetector", None)
            
            if detector_class:
                return detector_class()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load hardware detector: {e}")
            return None
    
    def detect_hardware(self) -> Dict[str, Dict[str, Any]]:
        """Detect available hardware.
        
        Returns:
            Dictionary of hardware capabilities.
        """
        hardware_info = {
            "cpu": {"available": True, "name": "CPU", "memory": None},
            "cuda": {"available": False, "name": None, "memory": None},
            "rocm": {"available": False, "name": None, "memory": None},
            "mps": {"available": False, "name": None, "memory": None},
            "openvino": {"available": False, "name": None, "memory": None},
            "qualcomm": {"available": False, "name": None, "memory": None},
        }
        
        # Use hardware detector if available
        if self.hardware_detector:
            try:
                detected_info = self.hardware_detector.detect_all()
                if detected_info:
                    # Merge with our default structure
                    for hw_type, hw_info in detected_info.items():
                        if hw_type in hardware_info:
                            hardware_info[hw_type].update(hw_info)
                    
                    return hardware_info
            except Exception as e:
                self.logger.error(f"Error using hardware detector: {e}")
        
        # Manual detection as fallback
        try:
            # Try to import torch for CUDA/ROCm detection
            import torch
            
            # Check for CUDA
            if torch.cuda.is_available():
                cuda_device_name = torch.cuda.get_device_name(0)
                cuda_device_mem = torch.cuda.get_device_properties(0).total_memory
                
                # Check if this is actually ROCm (AMD GPU)
                if "AMD" in cuda_device_name or "Radeon" in cuda_device_name:
                    hardware_info["rocm"]["available"] = True
                    hardware_info["rocm"]["name"] = cuda_device_name
                    hardware_info["rocm"]["memory"] = cuda_device_mem
                else:
                    hardware_info["cuda"]["available"] = True
                    hardware_info["cuda"]["name"] = cuda_device_name
                    hardware_info["cuda"]["memory"] = cuda_device_mem
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                hardware_info["mps"]["available"] = True
                hardware_info["mps"]["name"] = "Apple Silicon MPS"
            
            # Check for OpenVINO
            try:
                import openvino
                hardware_info["openvino"]["available"] = True
                hardware_info["openvino"]["name"] = f"OpenVINO {openvino.__version__}"
            except ImportError:
                pass
            
            # Check for Qualcomm
            try:
                import importlib.util
                has_qnn = importlib.util.find_spec("qnn_wrapper") is not None
                has_qti = importlib.util.find_spec("qti.aisw.dlc_utils") is not None
                has_qualcomm_env = "QUALCOMM_SDK" in os.environ
                
                if has_qnn or has_qti or has_qualcomm_env:
                    hardware_info["qualcomm"]["available"] = True
                    hardware_info["qualcomm"]["name"] = "Qualcomm AI Engine"
            except ImportError:
                pass
                
        except ImportError:
            self.logger.warning("PyTorch not available, cannot detect CUDA/ROCm/MPS")
        except Exception as e:
            self.logger.error(f"Error in hardware detection: {e}")
        
        return hardware_info
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model type.
        
        Args:
            model_type: Model type (bert, gpt2, t5, etc.).
            
        Returns:
            Dictionary with model information.
        """
        # Try to get from registry
        if hasattr(self.model_registry, "get_model_info"):
            model_info = self.model_registry.get_model_info(model_type)
            if model_info:
                return model_info
        
        # Fallback to built-in model info
        architecture_mapping = {
            "bert": "encoder-only",
            "roberta": "encoder-only",
            "distilbert": "encoder-only",
            "albert": "encoder-only",
            "electra": "encoder-only",
            "gpt2": "decoder-only",
            "gpt-neo": "decoder-only",
            "gpt-j": "decoder-only",
            "llama": "decoder-only",
            "t5": "encoder-decoder",
            "bart": "encoder-decoder",
            "mbart": "encoder-decoder",
            "vit": "vision",
            "deit": "vision",
            "clip": "vision-text",
            "wav2vec2": "speech",
            "whisper": "speech",
            # Specialized architectures
            "stable-diffusion": "text-to-image",
            "latent-diffusion": "text-to-image",
            "kandinsky": "text-to-image",
            "dalle": "text-to-image",
            "sdxl": "text-to-image",
            "esm": "protein-folding",
            "esm2": "protein-folding",
            "esmfold": "protein-folding",
            "prot-bert": "protein-folding",
            "videomae": "video-processing",
            "vivit": "video-processing",
            "timesformer": "video-processing",
        }
        
        task_mapping = {
            "bert": "fill-mask",
            "roberta": "fill-mask",
            "gpt2": "text-generation",
            "llama": "text-generation",
            "t5": "translation",
            "bart": "summarization",
            "vit": "image-classification",
            "clip": "image-to-text",
            "wav2vec2": "audio-to-text",
            "whisper": "audio-to-text",
            "stable-diffusion": "text-to-image",
            "esm": "protein-structure-prediction",
            "videomae": "video-classification",
        }
        
        architecture = architecture_mapping.get(model_type, self.config.get("default_architecture", "encoder-only"))
        task = task_mapping.get(model_type, self.config.get("default_task", "fill-mask"))
        
        return {
            "id": model_type,
            "name": model_type,
            "architecture": architecture,
            "task": task,
            "default_model": f"{model_type}-base" if model_type != "gpt2" else "gpt2",
            "hidden_size": 768,  # Default for most base models
            "automodel_class": f"AutoModelFor{task.replace('-', '')}"
        }
    
    def get_template_for_model(self, model_info: Dict[str, Any]) -> str:
        """Get the appropriate template for a model.
        
        Args:
            model_info: Dictionary with model information.
            
        Returns:
            Template content as string.
        """
        architecture = model_info.get("architecture", "encoder-only")
        
        # Map normalized architecture name to template file name
        template_mapping = {
            "encoder-only": "encoder_only",
            "decoder-only": "decoder_only",
            "encoder-decoder": "encoder_decoder",
            "vision": "vision",
            "vision-encoder-text-decoder": "vision_text",
            "speech": "speech",
            "multimodal": "multimodal",
            "text-to-image": "text_to_image",
            "protein-folding": "protein_folding",
            "video-processing": "video_processing",
            "diffusion": "diffusion_model",
            "mixture-of-experts": "moe_model",
            "state-space": "ssm_model",
            "rag": "rag_model"
        }
        
        # Get template filename
        template_name = template_mapping.get(architecture, architecture.replace("-", "_"))
        
        # Try the specific architecture template (with and without _template suffix)
        if f"{template_name}" in self.template_registry:
            return self.template_registry[f"{template_name}"]
        elif f"{template_name}_template" in self.template_registry:
            return self.template_registry[f"{template_name}_template"]
        
        # Try the specific architecture legacy template format
        if architecture in self.template_registry:
            return self.template_registry[architecture]
            
        # Fallback to reference template
        if "reference" in self.template_registry:
            return self.template_registry["reference"]
        elif "hf_reference_template" in self.template_registry:
            return self.template_registry["hf_reference_template"]
        
        # Last resort fallback
        self.logger.warning(f"No template found for architecture {architecture}, using generic template")
        return self.template_registry.get("base", "# Could not find appropriate template for this model")
    
    def render_template(self, template: str, context: Dict[str, Any]) -> str:
        """Fill a template with values from context.
        
        Args:
            template: Template string.
            context: Dictionary with values to fill.
            
        Returns:
            Rendered template.
        """
        # Simple string replacement for placeholders
        result = template
        
        # Replace placeholders with values from context
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                # Special handling for model_type to ensure no hyphens in class names
                if key == "model_type" and "-" in str(value):
                    # Create a sanitized version with underscores instead of hyphens
                    sanitized_value = str(value).replace("-", "_")
                    # Replace class and function names with sanitized version
                    result = re.sub(r'class\s+hf_{' + key + '}', f'class hf_{sanitized_value}', result)
                    result = re.sub(r'def\s+hf_{' + key + '}', f'def hf_{sanitized_value}', result)
                    result = result.replace(f"hf_{{{key}}} test", f"hf_{sanitized_value} test")
                    # But preserve original model name for other occurrences like display text
                    result = result.replace(placeholder, str(value))
                else:
                    result = result.replace(placeholder, str(value))
        
        # Additional sanitization for any hyphens that might have been missed
        result = re.sub(r'class\s+hf_([a-zA-Z0-9_-]+):', lambda m: f'class hf_{m.group(1).replace("-", "_")}:', result)
        
        # Fix empty return statements
        result = re.sub(r'(\s+)return\s*$', r'\1return {}', result)
        result = re.sub(r'(\s+)return\s*#\s*Code will be generated here', 
                        r'\1return {"success": True, "device": device, "hardware": hardware_label}', result)
        
        # Fix missing imports (for AutoModel classes)
        if "AutoModel" in result and "from transformers import AutoModel" not in result:
            auto_models = re.findall(r'(\bAutoModel\w*)', result)
            auto_models = list(set(auto_models))  # Remove duplicates
            
            if auto_models:
                import_line = f"from transformers import {', '.join(auto_models)}"
                # Add after other imports
                import_section_end = result.find("import time") + len("import time")
                if import_section_end > 0:
                    result = result[:import_section_end] + "\n" + import_line + result[import_section_end:]
        
        # Fix any specialized imports based on architecture
        if "diffusers." in result and "import diffusers" not in result:
            if "import torch" in result:
                # Add after torch import
                torch_import = result.find("import torch")
                import_end = result.find("\n", torch_import)
                result = result[:import_end+1] + "import diffusers\n" + result[import_end+1:]
            else:
                # Add at top of imports
                result = "import diffusers\n" + result
        
        return result
    
    def build_context(self, model_type: str, model_info: Dict[str, Any], 
                     hardware_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build context for template rendering.
        
        Args:
            model_type: Model type (bert, gpt2, t5, etc.).
            model_info: Dictionary with model information.
            hardware_info: Dictionary with hardware capabilities.
            
        Returns:
            Context dictionary for template rendering.
        """
        # Basic context
        context = {
            "model_type": model_type,
            "model_type_upper": model_type.upper(),
            "automodel_class": model_info.get("automodel_class", "AutoModel"),
            "task_type": model_info.get("task", "fill-mask"),
            "task_class": model_info.get("task", "fill-mask").replace("-", "").capitalize(),
            "hidden_size": model_info.get("hidden_size", 768),
            "test_input": "This is a test input for the model.",
        }
        
        # Add model description based on architecture
        architecture = model_info.get("architecture", "encoder-only")
        architecture_descriptions = {
            "encoder-only": f"The {model_type} model is an encoder-only Transformer that can be used for tasks like masked language modeling, sequence classification, and token classification.",
            "decoder-only": f"The {model_type} model is a decoder-only Transformer that can be used for autoregressive text generation, causal language modeling, and sequence completion.",
            "encoder-decoder": f"The {model_type} model is an encoder-decoder Transformer that can be used for sequence-to-sequence tasks like translation, summarization, and question answering.",
            "vision": f"The {model_type} model is a vision Transformer that can be used for image classification, object detection, and other vision tasks.",
            "vision-text": f"The {model_type} model is a multimodal vision-text Transformer that can be used for tasks like image captioning, visual question answering, and cross-modal retrieval.",
            "speech": f"The {model_type} model is a speech Transformer that can be used for tasks like speech recognition, speech-to-text, and audio classification.",
            "text-to-image": f"The {model_type} model is a text-to-image diffusion model that can generate images from text prompts.",
            "protein-folding": f"The {model_type} model is a protein language model that can predict protein structure and properties from amino acid sequences.",
            "video-processing": f"The {model_type} model is a video processing Transformer that can be used for video classification, action recognition, and temporal analysis.",
        }
        context["model_description"] = architecture_descriptions.get(
            architecture,
            f"The {model_type} model is a Transformer model for NLP and ML tasks."
        )
        
        # Add specific test inputs based on task
        task_inputs = {
            "fill-mask": "The capital of France is [MASK].",
            "text-generation": "Once upon a time in a land far away,",
            "translation": "Hello, how are you?",
            "summarization": "The researchers at DeepMind published a new paper on artificial intelligence. The paper describes a novel approach to reinforcement learning that outperforms previous methods on a range of tasks.",
            "question-answering": "What is the capital of France?",
            "image-classification": "<image_input>",
            "image-segmentation": "<image_input>",
            "audio-to-text": "<audio_input>",
            "text-to-image": "A beautiful sunset over the mountains",
            "protein-structure-prediction": "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ",
            "video-classification": "<video_input>"
        }
        context["test_input"] = task_inputs.get(
            model_info.get("task", "fill-mask"), 
            "This is a test input for the model."
        )
        
        # Add hardware availability flags
        context.update({
            "has_cuda": hardware_info.get("cuda", {}).get("available", False),
            "has_rocm": hardware_info.get("rocm", {}).get("available", False),
            "has_mps": hardware_info.get("mps", {}).get("available", False),
            "has_openvino": hardware_info.get("openvino", {}).get("available", False),
            "has_qualcomm": hardware_info.get("qualcomm", {}).get("available", False),
        })
        
        return context
    
    def validate_syntax(self, content: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax in generated content.
        
        Args:
            content: Generated content to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            # Try to compile to check syntax
            compile(content, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def fix_syntax(self, content: str) -> str:
        """Fix common syntax issues in generated content.
        
        Args:
            content: Content to fix.
            
        Returns:
            Fixed content.
        """
        # Check if we have a syntax module
        if hasattr(self, "syntax_module") and hasattr(self.syntax_module, "fix_syntax"):
            try:
                return self.syntax_module.fix_syntax(content)
            except Exception as e:
                self.logger.error(f"Error using syntax fixer: {e}")
        
        # Simple fixes
        fixed = content
        
        # Fix unbalanced quotes
        if fixed.count('"""') % 2 != 0:
            fixed += '\n"""'
        if fixed.count("'''") % 2 != 0:
            fixed += "\n'''"
            
        # Fix unbalanced parentheses, brackets, braces
        for open_char, close_char in [('(', ')'), ('[', ']'), ('{', '}')]:
            diff = fixed.count(open_char) - fixed.count(close_char)
            if diff > 0:
                fixed += close_char * diff
        
        # Replace problematic constructs
        fixed = fixed.replace('""""', '"""')
        fixed = fixed.replace("''''", "'''")
        
        # Fix hyphenated class names
        class_pattern = re.compile(r'class\s+hf_([a-zA-Z0-9_-]+):')
        match = class_pattern.search(fixed)
        if match and '-' in match.group(1):
            original_name = match.group(1)
            fixed_name = original_name.replace('-', '_')
            fixed = fixed.replace(f"hf_{original_name}", f"hf_{fixed_name}")
            fixed = fixed.replace(f"hf_{original_name} test", f"hf_{fixed_name} test")
        
        # Fix incomplete return statements
        fixed = re.sub(r'(\s+)return\s*$', r'\1return {}', fixed)
        fixed = re.sub(r'(\s+)return\s*#\s*Code will be generated here', 
                    r'\1return {"success": True, "device": device, "hardware": hardware_label}', fixed)
        
        # Fix missing imports based on content
        if "AutoModel" in fixed and "from transformers import AutoModel" not in fixed:
            auto_models = re.findall(r'(\bAutoModel\w*)', fixed)
            auto_models = list(set(auto_models))  # Remove duplicates
            if auto_models:
                import_line = f"from transformers import {', '.join(auto_models)}"
                
                # Find a good place to insert the import
                if "import" in fixed:
                    import_section_end = max(fixed.find("import time") + len("import time"), 
                                           fixed.find("import torch") + len("import torch"),
                                           fixed.find("import os") + len("import os"))
                    if import_section_end > len("import os"):
                        fixed = fixed[:import_section_end] + "\n" + import_line + fixed[import_section_end:]
                else:
                    # Put at the top
                    fixed = import_line + "\n" + fixed
        
        # Add diffusers import for text-to-image models
        if "diffusers." in fixed and "import diffusers" not in fixed:
            if "import torch" in fixed:
                # Add after torch import
                torch_import = fixed.find("import torch")
                import_end = fixed.find("\n", torch_import)
                fixed = fixed[:import_end+1] + "import diffusers\n" + fixed[import_end+1:]
            else:
                # Add at top of imports
                fixed = "import diffusers\n" + fixed
        
        return fixed
    
    def generate_test(self, model_type: str, output_path: Optional[str] = None, 
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a test file for a specific model type.
        
        Args:
            model_type: Model type (bert, gpt2, t5, etc.).
            output_path: Optional output file path.
            options: Optional dictionary with generation options.
            
        Returns:
            Dictionary with generation results.
        """
        start_time = time.time()
        options = options or {}
        
        self.logger.info(f"Generating test for model type: {model_type}")
        
        try:
            # Get model information
            model_info = self.get_model_info(model_type)
            if not model_info:
                error_msg = f"Cannot find model info for {model_type}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "model_type": model_type,
                    "duration": time.time() - start_time
                }
            
            # Check specifically for text-to-image, protein-folding, and video-processing models
            if '-' in model_type:
                # Look for specific architecture detection in model name
                if model_type.startswith('stable-') or model_type.endswith('-diffusion'):
                    model_info["architecture"] = "text-to-image"
                elif model_type.startswith('esm') or model_type.startswith('prot-'):
                    model_info["architecture"] = "protein-folding"
                elif model_type.startswith('video') or model_type.endswith('-video'):
                    model_info["architecture"] = "video-processing"
            
            # Get hardware information
            hardware_info = self.detect_hardware()
            
            # Get template
            template = self.get_template_for_model(model_info)
            if not template:
                error_msg = f"Cannot find template for {model_type}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "model_type": model_type,
                    "duration": time.time() - start_time
                }
            
            # Build context
            context = self.build_context(model_type, model_info, hardware_info)
            
            # Render template
            content = self.render_template(template, context)
            
            # Validate and fix syntax if needed
            if self.config.get("fix_syntax", True):
                is_valid, error = self.validate_syntax(content)
                if not is_valid:
                    self.logger.warning(f"Syntax error in generated content: {error}")
                    content = self.fix_syntax(content)
                    
                    # Validate again after fixing
                    is_valid, error = self.validate_syntax(content)
                    if not is_valid:
                        self.logger.error(f"Failed to fix syntax: {error}")
            
            # Determine output path
            if not output_path:
                output_dir = options.get("output_dir", self.config.get("output_dir", "./generated_tests"))
                sanitized_model_type = model_type.replace('-', '_')
                output_path = os.path.join(output_dir, f"test_hf_{sanitized_model_type}.py")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Write output
            with open(output_path, 'w') as f:
                f.write(content)
            
            # Update stats
            self.stats["generated"] += 1
            architecture = model_info.get("architecture", "unknown")
            if architecture not in self.stats["models_by_architecture"]:
                self.stats["models_by_architecture"][architecture] = 0
            self.stats["models_by_architecture"][architecture] += 1
            
            self.logger.info(f"Generated test file: {output_path}")
            
            return {
                "success": True,
                "model_type": model_type,
                "architecture": model_info.get("architecture", "unknown"),
                "output_file": output_path,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.exception(f"Error generating test for {model_type}")
            self.stats["failed"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "model_type": model_type,
                "duration": time.time() - start_time
            }
    
    def generate_batch(self, model_types: List[str], output_dir: Optional[str] = None,
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate test files for multiple model types.
        
        Args:
            model_types: List of model types to generate.
            output_dir: Optional output directory.
            options: Optional dictionary with generation options.
            
        Returns:
            Dictionary with batch generation results.
        """
        start_time = time.time()
        options = options or {}
        
        if output_dir:
            options["output_dir"] = output_dir
            
        # Generate test for each model type
        results = []
        success_count = 0
        error_count = 0
        
        for model_type in model_types:
            # Generate test file
            result = self.generate_test(model_type, None, options)
            results.append(result)
            
            if result["success"]:
                success_count += 1
            else:
                error_count += 1
        
        return {
            "success": error_count == 0,
            "results": results,
            "total_count": len(model_types),
            "success_count": success_count,
            "error_count": error_count,
            "duration": time.time() - start_time,
            "stats": self.stats
        }
    
    def generate_report(self, result: Dict[str, Any], report_path: Optional[str] = None) -> None:
        """Generate a summary report of test generation.
        
        Args:
            result: Generation result dictionary.
            report_path: Optional path to write the report.
        """
        # Create report
        is_batch = "results" in result
        
        if is_batch:
            # Batch report
            total_count = result.get("total_count", 0)
            success_count = result.get("success_count", 0)
            error_count = result.get("error_count", 0)
            duration = result.get("duration", 0)
            
            report = f"""# Test Generation Report

## Summary
- Total models: {total_count}
- Successfully generated: {success_count}
- Failed: {error_count}
- Success rate: {(success_count / total_count * 100) if total_count > 0 else 0:.2f}%
- Duration: {duration:.2f} seconds

## Models by Architecture
"""
            
            for arch, count in result.get("stats", {}).get("models_by_architecture", {}).items():
                report += f"- {arch}: {count}\n"
            
            report += "\n## Details\n\n"
            
            # Add successful models
            report += "\n### Successfully Generated Models\n\n"
            for item in result.get("results", []):
                if item.get("success", False):
                    model_type = item.get("model_type", "unknown")
                    architecture = item.get("architecture", "unknown")
                    output_file = item.get("output_file", "unknown")
                    report += f"- **{model_type}** ({architecture}): {output_file}\n"
            
            # Add failed models
            report += "\n### Failed Models\n\n"
            for item in result.get("results", []):
                if not item.get("success", False):
                    model_type = item.get("model_type", "unknown")
                    error = item.get("error", "Unknown error")
                    report += f"- **{model_type}**: {error}\n"
        
        else:
            # Single model report
            model_type = result.get("model_type", "unknown")
            success = result.get("success", False)
            duration = result.get("duration", 0)
            
            report = f"""# Test Generation Report for {model_type}

## Summary
- Model: {model_type}
- Success: {"Yes" if success else "No"}
- Duration: {duration:.2f} seconds

## Details
"""
            
            if success:
                architecture = result.get("architecture", "unknown")
                output_file = result.get("output_file", "unknown")
                report += f"- Architecture: {architecture}\n"
                report += f"- Output file: {output_file}\n"
            else:
                error = result.get("error", "Unknown error")
                report += f"- Error: {error}\n"
        
        # Write or print report
        if report_path:
            with open(report_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report written to: {report_path}")
        else:
            print("\n" + report)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="HuggingFace Model Test Generator Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument("--model", "-m", dest="model_type",
                        help="Model type to generate test for (bert, gpt2, t5, etc.)")
    parser.add_argument("--output", "-o", dest="output_file",
                        help="Output file path")
    parser.add_argument("--config", "-c", dest="config_file",
                        help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    # Batch generation
    parser.add_argument("--batch", action="store_true",
                        help="Enable batch generation mode")
    parser.add_argument("--batch-file", dest="batch_file",
                        help="File containing list of models to generate")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Output directory for generated files")
    parser.add_argument("--architecture", choices=ARCHITECTURE_TYPES,
                        help="Generate tests for models with this architecture")
    parser.add_argument("--task", choices=TASK_TYPES,
                        help="Generate tests for models supporting this task")
    
    # Output options
    parser.add_argument("--report", action="store_true",
                        help="Generate a summary report")
    parser.add_argument("--report-file", dest="report_file",
                        help="Report file path")
    parser.add_argument("--json", action="store_true",
                        help="Output results in JSON format")
    
    # Hardware options
    parser.add_argument("--hardware", choices=HARDWARE_BACKENDS,
                        help="Specify target hardware backend (for testing)")
    
    return parser.parse_args()


def load_batch_models(args: argparse.Namespace, generator: TestGeneratorSuite) -> List[str]:
    """Load models for batch generation.
    
    Args:
        args: Parsed command-line arguments.
        generator: Test generator instance.
        
    Returns:
        List of model types to generate.
    """
    models = []
    
    # Load from batch file if provided
    if args.batch_file:
        try:
            with open(args.batch_file, 'r') as f:
                if args.batch_file.endswith('.json'):
                    # JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        models = data
                    elif isinstance(data, dict) and "models" in data:
                        models = data["models"]
                else:
                    # Plain text, one model per line
                    models = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading batch file: {e}")
    
    # Filter by architecture if provided
    elif args.architecture:
        try:
            # Get models for this architecture from registry
            if hasattr(generator.model_registry, "get_models_by_architecture"):
                arch_models = generator.model_registry.get_models_by_architecture(args.architecture)
                if arch_models:
                    models = arch_models
        except Exception as e:
            logger.error(f"Error getting models by architecture: {e}")
    
    # If still empty, use some default models
    if not models:
        models = [
            "bert", "roberta", "distilbert", "gpt2", "t5", "bart", 
            "vit", "wav2vec2", "clip", "albert", "electra"
        ]
    
    return models


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code.
    """
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create generator
    generator = TestGeneratorSuite(args.config_file)
    
    # Generate tests
    if args.batch or args.batch_file or args.architecture:
        # Batch generation
        models = load_batch_models(args, generator)
        logger.info(f"Generating tests for {len(models)} models...")
        
        result = generator.generate_batch(models, args.output_dir)
        
        # Output results
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Generated {result['success_count']} of {result['total_count']} tests")
            print(f"Duration: {result['duration']:.2f} seconds")
        
        # Generate report if requested
        if args.report or args.report_file:
            generator.generate_report(result, args.report_file)
        
        return 0 if result["success"] else 1
    
    elif args.model_type:
        # Single model generation
        logger.info(f"Generating test for model: {args.model_type}")
        
        result = generator.generate_test(args.model_type, args.output_file)
        
        # Output results
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print(f"Successfully generated test for {args.model_type}")
                print(f"Output: {result['output_file']}")
            else:
                print(f"Failed to generate test for {args.model_type}: {result['error']}")
        
        # Generate report if requested
        if args.report or args.report_file:
            generator.generate_report(result, args.report_file)
        
        return 0 if result["success"] else 1
    
    else:
        # No model specified
        logger.error("No model type specified. Use --model or --batch.")
        return 1


if __name__ == "__main__":
    sys.exit(main())