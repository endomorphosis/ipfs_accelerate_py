#!/usr/bin/env python3
"""
Improved Skillset Generator

This module is an enhanced version of the integrated_skillset_generator.py with:

1. Standardized hardware detection using improved_hardware_detection module
2. Properly integrated database storage using database_integration module
3. Fixed duplicated code and inconsistent error handling
4. Added improved cross-platform test generation
5. Better error handling for thread pool execution

Usage:
    python improved_skillset_generator.py --model bert
    python improved_skillset_generator.py --all --cross-platform
    python improved_skillset_generator.py --family text-embedding
    python improved_skillset_generator.py --model bert --hardware cuda,webgpu
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import traceback
import importlib.util
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if improvements module is in the path
if importlib.util.find_spec("improvements") is None:
    # Add the parent directory to the path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import improved hardware detection and database modules
try:
    from improvements.improved_hardware_detection import (
        detect_all_hardware, 
        apply_web_platform_optimizations,
        get_hardware_compatibility_matrix,
        HAS_CUDA,
        HAS_ROCM,
        HAS_MPS,
        HAS_OPENVINO,
        HAS_WEBNN,
        HAS_WEBGPU,
        HARDWARE_PLATFORMS,
        KEY_MODEL_HARDWARE_MATRIX
    )
    HAS_HARDWARE_MODULE = True
except ImportError:
    logger.warning("Could not import hardware detection module, using local implementation")
    HAS_HARDWARE_MODULE = False

try:
    from improvements.database_integration import (
        DUCKDB_AVAILABLE,
        DEPRECATE_JSON_OUTPUT,
        get_or_create_test_run,
        get_or_create_model,
        store_test_result,
        store_implementation_metadata,
        complete_test_run
    )
    HAS_DATABASE_MODULE = True
except ImportError:
    logger.warning("Could not import database integration module, using local implementation")
    HAS_DATABASE_MODULE = False
    # Set environment variable flag
    DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

# Create a fallback hardware detection function if the module is not available
if not HAS_HARDWARE_MODULE:
    def detect_hardware():
        """Simple hardware detection fallback"""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
            
        return {
            "cpu": {"detected": True},
            "cuda": {"detected": has_cuda},
            "rocm": {"detected": False},
            "mps": {"detected": False},
            "openvino": {"detected": False},
            "webnn": {"detected": False},
            "webgpu": {"detected": False},
        }
    
    # Use fallback detection
    detect_all_hardware = detect_hardware
    
    # Define fallback variables
    HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]
    
    # Create a fallback compatibility matrix
    def get_hardware_compatibility_matrix():
        """Fallback hardware compatibility matrix"""
        # Default compatibility for all models
        default_compat = {
            "cpu": "REAL",
            "cuda": "REAL",
            "openvino": "REAL",
            "mps": "REAL",
            "rocm": "REAL",
            "webnn": "REAL",
            "webgpu": "REAL"
        }
        
        # Build matrix with defaults
        compatibility_matrix = {
            "bert": default_compat,
            "t5": default_compat,
            "gpt2": default_compat,
            "vit": default_compat,
            "clip": default_compat,
            # Add other models as needed
        }
        
        return compatibility_matrix
    
    KEY_MODEL_HARDWARE_MATRIX = get_hardware_compatibility_matrix()

# Output directory for generated skillsets
OUTPUT_DIR = Path("./generated_skillsets")

class SkillsetGenerator:
    """
    Enhanced skillset generator that creates implementation files based on model types,
    with comprehensive hardware detection and database integration.
    """
    
    def __init__(self):
        """Initialize the skillset generator"""
        self.hw_capabilities = detect_all_hardware()
        self.output_dir = OUTPUT_DIR
        self.model_registry = self._load_model_registry()
        
    def _load_model_registry(self):
        """Load model registry containing available models and their families"""
        # This would normally load from a centralized registry
        # but for this example, we'll use a simple dictionary
        families = {
            "text_embedding": ["bert", "roberta", "albert", "distilbert"],
            "text_generation": ["t5", "gpt2", "llama", "opt", "falcon"],
            "vision": ["vit", "resnet", "convnext", "deit"],
            "vision_text": ["clip", "blip"],
            "audio": ["whisper", "wav2vec2", "clap"],
            "multimodal": ["llava", "llava-next", "xclip"],
        }
        
        # Create a registry with task information
        registry = {}
        for family, models in families.items():
            for model in models:
                if family == "text_embedding":
                    task = "embedding"
                elif family == "text_generation":
                    task = "generation"
                elif family == "vision":
                    task = "classification"
                elif family == "vision_text" or family == "multimodal":
                    task = "multimodal"
                elif family == "audio":
                    task = "transcription"
                else:
                    task = "general"
                    
                registry[model] = {
                    "family": family,
                    "task": task
                }
        
        return registry
    
    def determine_hardware_compatibility(self, model_type: str) -> Dict[str, str]:
        """
        Determine hardware compatibility for a model type.
        
        Args:
            model_type: Type of model (bert, t5, etc.)
            
        Returns:
            Dict mapping hardware platforms to compatibility types (REAL, SIMULATION, False)
        """
        # Standardize model type
        model_type = model_type.lower().split("-")[0]
        
        # Check if model type is in the hardware compatibility matrix
        if model_type in KEY_MODEL_HARDWARE_MATRIX:
            # Use predefined compatibility
            compatibility = KEY_MODEL_HARDWARE_MATRIX[model_type]
        else:
            # Determine model family
            family = self.model_registry.get(model_type, {}).get("family", "unknown")
            
            # Use family-based compatibility
            if family == "text_embedding" or family == "text_generation":
                compatibility = {
                    "cpu": "REAL",
                    "cuda": "REAL", 
                    "openvino": "REAL", 
                    "mps": "REAL", 
                    "rocm": "REAL",
                    "webnn": "REAL", 
                    "webgpu": "REAL"
                }
            elif family == "vision":
                compatibility = {
                    "cpu": "REAL",
                    "cuda": "REAL", 
                    "openvino": "REAL", 
                    "mps": "REAL", 
                    "rocm": "REAL",
                    "webnn": "REAL", 
                    "webgpu": "REAL"
                }
            elif family == "audio":
                compatibility = {
                    "cpu": "REAL",
                    "cuda": "REAL", 
                    "openvino": "REAL", 
                    "mps": "REAL", 
                    "rocm": "REAL",
                    "webnn": "SIMULATION", 
                    "webgpu": "SIMULATION"
                }
            elif family == "vision_text" or family == "multimodal":
                compatibility = {
                    "cpu": "REAL",
                    "cuda": "REAL", 
                    "openvino": "SIMULATION", 
                    "mps": "SIMULATION", 
                    "rocm": "SIMULATION",
                    "webnn": "SIMULATION", 
                    "webgpu": "SIMULATION"
                }
            else:
                # Default compatibility
                compatibility = {
                    "cpu": "REAL",
                    "cuda": "REAL", 
                    "openvino": "REAL", 
                    "mps": "REAL", 
                    "rocm": "REAL",
                    "webnn": "REAL", 
                    "webgpu": "REAL"
                }
        
        # Override based on actual hardware availability
        hw_capabilities = self.hw_capabilities
        
        for platform in HARDWARE_PLATFORMS:
            # If hardware is not detected, mark it as False regardless of compatibility
            if not hw_capabilities.get(platform, {}).get("detected", False):
                # For CPU, always keep as REAL since it's always available
                if platform != "cpu":
                    compatibility[platform] = False
        
        return compatibility
    
    def get_skillset_template(self, model_type: str, hardware_compatibility: Dict[str, str]) -> str:
        """
        Get a skillset implementation template for the given model type with hardware support.
        
        Args:
            model_type: Type of model (bert, t5, etc.)
            hardware_compatibility: Dict mapping hardware platforms to compatibility types
            
        Returns:
            Skillset implementation template string
        """
        # Standardized imports
        imports = """#!/usr/bin/env python3
\"\"\"
{model_type_cap} Model Implementation

This module provides the implementation for the {model_type_cap} model with
cross-platform hardware support.
\"\"\"

import os
import logging
import importlib.util
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
        
        # Hardware detection code
        hw_detection = """
# Hardware detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    from unittest.mock import MagicMock
    torch = MagicMock()
    logger.warning("torch not available, using mock")

# Initialize hardware capability flags
HAS_CUDA = False
HAS_ROCM = False
HAS_MPS = False
HAS_OPENVINO = False
HAS_WEBNN = False
HAS_WEBGPU = False

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    
    # ROCm detection
    if HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
    
    # Apple MPS detection
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        HAS_MPS = torch.mps.is_available()

# OpenVINO detection
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None

# WebNN detection (browser API)
HAS_WEBNN = (
    importlib.util.find_spec("webnn") is not None or 
    importlib.util.find_spec("webnn_js") is not None or
    "WEBNN_AVAILABLE" in os.environ or
    "WEBNN_SIMULATION" in os.environ
)

# WebGPU detection (browser API)
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    importlib.util.find_spec("wgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

def detect_hardware() -> Dict[str, bool]:
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {
        "cpu": True,
        "cuda": HAS_CUDA,
        "rocm": HAS_ROCM,
        "mps": HAS_MPS,
        "openvino": HAS_OPENVINO,
        "webnn": HAS_WEBNN,
        "webgpu": HAS_WEBGPU
    }
    return capabilities

# Web Platform Optimizations
def apply_web_platform_optimizations(platform: str = "webgpu") -> Dict[str, bool]:
    \"\"\"Apply web platform optimizations based on environment settings.\"\"\"
    optimizations = {
        "compute_shaders": False,
        "parallel_loading": False,
        "shader_precompile": False
    }
    
    # Check for optimization environment flags
    if os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1":
        optimizations["compute_shaders"] = True
    
    if os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1":
        optimizations["parallel_loading"] = True
        
    if os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1":
        optimizations["shader_precompile"] = True
        
    if os.environ.get("WEB_ALL_OPTIMIZATIONS", "0") == "1":
        optimizations = {"compute_shaders": True, "parallel_loading": True, "shader_precompile": True}
    
    return optimizations
"""
        
        # Skillset implementation
        implementation = """
class {model_type_cap}Implementation:
    \"\"\"Implementation of the {model_type_cap} model with cross-platform hardware support.\"\"\"
    
    def __init__(self, model_name: str = None, **kwargs):
        \"\"\"
        Initialize the {model_type_cap} implementation.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional keyword arguments
        \"\"\"
        self.model_name = model_name or "{model_type}"
        self.hardware = detect_hardware()
        self.model = None
        self.backend = None
        self.select_hardware()
        
    def select_hardware(self) -> str:
        \"\"\"
        Select the best available hardware backend based on capabilities.
        
        Returns:
            Name of the selected backend
        \"\"\"
        # Default to CPU
        self.backend = "cpu"
        
        # Check for CUDA
        if self.hardware["cuda"] and {cuda_compat!r}:
            self.backend = "cuda"
        # Check for ROCm (AMD)
        elif self.hardware["rocm"] and {rocm_compat!r}:
            self.backend = "rocm"
        # Check for MPS (Apple)
        elif self.hardware["mps"] and {mps_compat!r}:
            self.backend = "mps"
        # Check for OpenVINO
        elif self.hardware["openvino"] and {openvino_compat!r}:
            self.backend = "openvino"
        # Check for WebGPU
        elif self.hardware["webgpu"] and {webgpu_compat!r}:
            self.backend = "webgpu"
        # Check for WebNN
        elif self.hardware["webnn"] and {webnn_compat!r}:
            self.backend = "webnn"
            
        # Log selection
        if self.backend != "cpu":
            logger.info(f"Selected hardware backend: {{self.backend}}")
            
        return self.backend
    
    def load_model(self) -> None:
        \"\"\"Load the model based on the selected hardware backend.\"\"\"
        if self.model is not None:
            return
            
        try:
            if self.backend == "cuda":
                self._load_cuda_model()
            elif self.backend == "rocm":
                self._load_rocm_model()
            elif self.backend == "mps":
                self._load_mps_model()
            elif self.backend == "openvino":
                self._load_openvino_model()
            elif self.backend == "webgpu":
                self._load_webgpu_model()
            elif self.backend == "webnn":
                self._load_webnn_model()
            else:
                # Default to CPU
                self._load_cpu_model()
                
            logger.info(f"Loaded {self.model_name} model on {{self.backend}}")
        except Exception as e:
            logger.error(f"Error loading model: {{e}}")
            # Fallback to CPU
            self.backend = "cpu"
            self._load_cpu_model()
    
    def _load_cpu_model(self) -> None:
        \"\"\"Load model on CPU.\"\"\"
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Error loading CPU model: {{e}}")
            raise
    
    def _load_cuda_model(self) -> None:
        \"\"\"Load model on CUDA.\"\"\"
        if not HAS_TORCH or not HAS_CUDA:
            logger.warning("CUDA not available, falling back to CPU")
            self._load_cpu_model()
            return
            
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).cuda()
        except Exception as e:
            logger.error(f"Error loading CUDA model: {{e}}")
            # Fallback to CPU
            self._load_cpu_model()
    
    def _load_rocm_model(self) -> None:
        \"\"\"Load model on ROCm.\"\"\"
        if not HAS_TORCH or not HAS_ROCM:
            logger.warning("ROCm not available, falling back to CPU")
            self._load_cpu_model()
            return
            
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).cuda()
        except Exception as e:
            logger.error(f"Error loading ROCm model: {{e}}")
            # Fallback to CPU
            self._load_cpu_model()
    
    def _load_mps_model(self) -> None:
        \"\"\"Load model on MPS (Apple Silicon).\"\"\"
        if not HAS_TORCH or not HAS_MPS:
            logger.warning("MPS not available, falling back to CPU")
            self._load_cpu_model()
            return
            
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to("mps")
        except Exception as e:
            logger.error(f"Error loading MPS model: {{e}}")
            # Fallback to CPU
            self._load_cpu_model()
    
    def _load_openvino_model(self) -> None:
        \"\"\"Load model with OpenVINO.\"\"\"
        if not HAS_OPENVINO:
            logger.warning("OpenVINO not available, falling back to CPU")
            self._load_cpu_model()
            return
            
        try:
            from transformers import AutoModel, AutoTokenizer
            from openvino.runtime import Core
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # First load the PyTorch model
            model = AutoModel.from_pretrained(self.model_name)
            
            # Convert to ONNX in memory
            import io
            import torch.onnx
            
            onnx_buffer = io.BytesIO()
            sample_input = self.tokenizer("Sample text", return_tensors="pt")
            torch.onnx.export(
                model,
                tuple(sample_input.values()),
                onnx_buffer,
                input_names=list(sample_input.keys()),
                output_names=["last_hidden_state"],
                opset_version=12,
                do_constant_folding=True
            )
            
            # Load with OpenVINO
            ie = Core()
            onnx_model = onnx_buffer.getvalue()
            ov_model = ie.read_model(model=onnx_model, weights=onnx_model)
            compiled_model = ie.compile_model(ov_model, "CPU")
            
            self.model = compiled_model
        except Exception as e:
            logger.error(f"Error loading OpenVINO model: {{e}}")
            # Fallback to CPU
            self._load_cpu_model()
    
    def _load_webgpu_model(self) -> None:
        \"\"\"Load model with WebGPU.\"\"\"
        if not HAS_WEBGPU:
            logger.warning("WebGPU not available, falling back to CPU")
            self._load_cpu_model()
            return
            
        try:
            # Apply optimizations
            optimizations = apply_web_platform_optimizations("webgpu")
            
            # Check if we're using real or simulated WebGPU
            if "WEBGPU_SIMULATION" in os.environ:
                # Simulated implementation
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("Using simulated WebGPU implementation")
            else:
                # Real WebGPU implementation (depends on browser environment)
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Load with transformers.js in browser environment
                # This is a placeholder for the real implementation
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info(f"Loaded WebGPU model with optimizations: {{optimizations}}")
        except Exception as e:
            logger.error(f"Error loading WebGPU model: {{e}}")
            # Fallback to CPU
            self._load_cpu_model()
    
    def _load_webnn_model(self) -> None:
        \"\"\"Load model with WebNN.\"\"\"
        if not HAS_WEBNN:
            logger.warning("WebNN not available, falling back to CPU")
            self._load_cpu_model()
            return
            
        try:
            # Check if we're using real or simulated WebNN
            if "WEBNN_SIMULATION" in os.environ:
                # Simulated implementation
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("Using simulated WebNN implementation")
            else:
                # Real WebNN implementation (depends on browser environment)
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Load with transformers.js in browser environment
                # This is a placeholder for the real implementation
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("Loaded WebNN model")
        except Exception as e:
            logger.error(f"Error loading WebNN model: {{e}}")
            # Fallback to CPU
            self._load_cpu_model()
    
    def infer(self, inputs: Union[str, List[str]]) -> Dict[str, Any]:
        \"\"\"
        Run inference with the model.
        
        Args:
            inputs: Input text or list of texts
            
        Returns:
            Dict containing the model outputs
        \"\"\"
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
            
        # Process inputs
        if isinstance(inputs, str):
            inputs = [inputs]
            
        try:
            # Tokenize inputs
            if hasattr(self, 'tokenizer'):
                encoded_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                
                # Move to appropriate device if using PyTorch
                if self.backend in ["cuda", "rocm", "mps"] and hasattr(encoded_inputs, "to"):
                    device = "cuda" if self.backend in ["cuda", "rocm"] else "mps"
                    encoded_inputs = {{k: v.to(device) for k, v in encoded_inputs.items()}}
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(**encoded_inputs)
                
                # Format results
                results = {{
                    "last_hidden_state": outputs.last_hidden_state.cpu().numpy().tolist() if hasattr(outputs, "last_hidden_state") else None,
                    "backend": self.backend
                }}
            else:
                # Generic fallback (e.g., for OpenVINO)
                results = {{
                    "outputs": None,
                    "backend": self.backend,
                    "error": "No tokenizer available"
                }}
                
            return results
        except Exception as e:
            logger.error(f"Error during inference: {{e}}")
            return {{"error": str(e), "backend": self.backend}}
    
    @classmethod
    def get_supported_hardware(cls) -> Dict[str, str]:
        \"\"\"
        Get information about supported hardware platforms.
        
        Returns:
            Dict mapping hardware platforms to support status (REAL, SIMULATION, False)
        \"\"\"
        return {{
            "cpu": "REAL",
            "cuda": {cuda_compat!r},
            "rocm": {rocm_compat!r},
            "mps": {mps_compat!r},
            "openvino": {openvino_compat!r},
            "webnn": {webnn_compat!r},
            "webgpu": {webgpu_compat!r}
        }}

# Instantiate the implementation for direct use
default_implementation = {model_type_cap}Implementation()

# Convenience functions
def load():
    \"\"\"Load the model.\"\"\"
    default_implementation.load_model()
    return default_implementation

def infer(inputs):
    \"\"\"Run inference with the model.\"\"\"
    return default_implementation.infer(inputs)
"""
        
        # Format templates
        model_type_cap = model_type.capitalize()
        
        # Format compatibility values
        cuda_compat = hardware_compatibility.get("cuda", "REAL")
        rocm_compat = hardware_compatibility.get("rocm", "REAL")
        mps_compat = hardware_compatibility.get("mps", "REAL")
        openvino_compat = hardware_compatibility.get("openvino", "REAL")
        webnn_compat = hardware_compatibility.get("webnn", "REAL")
        webgpu_compat = hardware_compatibility.get("webgpu", "REAL")
        
        formatted_imports = imports.format(model_type_cap=model_type_cap)
        formatted_implementation = implementation.format(
            model_type=model_type,
            model_type_cap=model_type_cap,
            cuda_compat=cuda_compat,
            rocm_compat=rocm_compat,
            mps_compat=mps_compat,
            openvino_compat=openvino_compat,
            webnn_compat=webnn_compat,
            webgpu_compat=webgpu_compat
        )
        
        # Combine all parts
        template = formatted_imports + hw_detection + formatted_implementation
        
        return template
    
    def generate_skillset(self, model_type: str, hardware_platforms: List[str] = None, 
                         cross_platform: bool = False) -> Optional[Path]:
        """
        Generate a skillset implementation file for the given model type.
        
        Args:
            model_type: Type of model (bert, t5, etc.)
            hardware_platforms: List of hardware platforms to support
            cross_platform: Whether to generate implementations for all platforms
            
        Returns:
            Path to the generated implementation file
        """
        # Standardize model type
        model_type = model_type.lower()
        
        # Check if model type is in registry
        if model_type not in self.model_registry:
            logger.warning(f"Model type '{model_type}' not found in registry")
            return None
        
        # Determine hardware compatibility
        hardware_compatibility = self.determine_hardware_compatibility(model_type)
        
        # Filter platforms based on arguments
        if cross_platform:
            # Use all platforms with their compatibility
            pass
        elif hardware_platforms:
            # Filter to specified platforms
            for platform in list(hardware_compatibility.keys()):
                if platform != "cpu" and platform not in hardware_platforms:
                    hardware_compatibility[platform] = False
        else:
            # Default to CPU and any available GPU (CUDA, ROCm, MPS)
            for platform in list(hardware_compatibility.keys()):
                if platform not in ["cpu", "cuda", "rocm", "mps"]:
                    hardware_compatibility[platform] = False
        
        logger.info(f"Generating skillset for {model_type} with compatibility: {hardware_compatibility}")
        
        # Get the skillset template
        template = self.get_skillset_template(model_type, hardware_compatibility)
        
        # Prepare output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate file path
        file_path = self.output_dir / f"hf_{model_type.lower()}.py"
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(template)
        
        # Store metadata in database if available
        if HAS_DATABASE_MODULE:
            # Create or get a test run
            run_id = get_or_create_test_run(
                test_name=f"generate_skillset_{model_type.lower()}",
                test_type="skillset_generation",
                metadata={
                    "model_type": model_type,
                    "hardware_compatibility": hardware_compatibility,
                    "cross_platform": cross_platform
                }
            )
            
            # Get or create model
            model_id = get_or_create_model(
                model_name=model_type,
                model_family=model_type.split("-")[0],
                model_type=self.model_registry.get(model_type, {}).get("family"),
                task=self.model_registry.get(model_type, {}).get("task")
            )
            
            # Store implementation metadata
            store_implementation_metadata(
                model_type=model_type,
                file_path=str(file_path),
                generation_date=datetime.datetime.now(),
                model_category=self.model_registry.get(model_type, {}).get("family"),
                hardware_support=hardware_compatibility,
                primary_task=self.model_registry.get(model_type, {}).get("task"),
                cross_platform=cross_platform
            )
            
            # Store test result
            store_test_result(
                run_id=run_id,
                test_name=f"generate_skillset_{model_type.lower()}",
                status="PASS",
                model_id=model_id,
                metadata={
                    "hardware_compatibility": hardware_compatibility,
                    "file_path": str(file_path)
                }
            )
            
            # Complete test run
            complete_test_run(run_id)
        
        logger.info(f"Generated skillset file: {file_path}")
        return file_path
    
    def generate_skillsets_batch(self, model_types: List[str], 
                               hardware_platforms: List[str] = None,
                               cross_platform: bool = False,
                               max_workers: int = 5) -> List[Path]:
        """
        Generate skillset implementation files for multiple model types in parallel.
        
        Args:
            model_types: List of model types
            hardware_platforms: List of hardware platforms to support
            cross_platform: Whether to generate implementations for all platforms
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of paths to generated implementation files
        """
        results = []
        failed_models = []
        
        # Use thread pool to generate skillsets in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dict mapping futures to their models
            future_to_model = {}
            for model_type in model_types:
                future = executor.submit(
                    self.generate_skillset,
                    model_type,
                    hardware_platforms,
                    cross_platform
                )
                future_to_model[future] = model_type
            
            # Process results as they complete
            for future in as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"Successfully generated skillset for {model_type}")
                    else:
                        failed_models.append(model_type)
                        logger.error(f"Failed to generate skillset for {model_type}")
                except Exception as e:
                    failed_models.append(model_type)
                    logger.error(f"Exception generating skillset for {model_type}: {e}")
                    logger.debug(traceback.format_exc())
        
        # Log summary
        logger.info(f"Generated {len(results)} skillset files, {len(failed_models)} failures")
        if failed_models:
            logger.info(f"Failed models: {', '.join(failed_models)}")
        
        return results
    
    def generate_family(self, family: str, 
                      hardware_platforms: List[str] = None,
                      cross_platform: bool = False,
                      max_workers: int = 5) -> List[Path]:
        """
        Generate skillset implementation files for all models in a family.
        
        Args:
            family: Model family (text_embedding, text_generation, etc.)
            hardware_platforms: List of hardware platforms to support
            cross_platform: Whether to generate implementations for all platforms
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of paths to generated implementation files
        """
        # Normalize family name
        family = family.lower().replace("-", "_")
        
        # Find models in this family
        model_types = []
        for model_type, info in self.model_registry.items():
            if info.get("family", "").lower() == family:
                model_types.append(model_type)
        
        if not model_types:
            logger.warning(f"No models found for family '{family}'")
            return []
        
        logger.info(f"Generating skillsets for {len(model_types)} models in family '{family}'")
        
        # Generate skillsets for all models in the family
        return self.generate_skillsets_batch(
            model_types,
            hardware_platforms,
            cross_platform,
            max_workers
        )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate model skillset implementation files")
    
    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Generate skillset for a specific model")
    group.add_argument("--family", type=str, help="Generate skillsets for a model family")
    group.add_argument("--all", action="store_true", help="Generate skillsets for all models in registry")
    
    # Hardware platforms
    parser.add_argument("--hardware", type=str, help="Comma-separated list of hardware platforms to include")
    parser.add_argument("--cross-platform", action="store_true", help="Generate implementations for all hardware platforms")
    
    # Output options
    parser.add_argument("--output-dir", type=str, help="Output directory for generated implementations")
    
    # Parallel processing
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of parallel workers")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create skillset generator
    generator = SkillsetGenerator()
    
    # Set output directory if provided
    if args.output_dir:
        generator.output_dir = Path(args.output_dir)
    
    # Parse hardware platforms
    hardware_platforms = None
    if args.hardware:
        hardware_platforms = args.hardware.split(",")
        if "all" in hardware_platforms:
            hardware_platforms = HARDWARE_PLATFORMS
    
    # Generate skillsets based on arguments
    if args.model:
        # Generate a single skillset
        generator.generate_skillset(
            args.model,
            hardware_platforms,
            args.cross_platform
        )
    elif args.family:
        # Generate skillsets for a family
        generator.generate_family(
            args.family,
            hardware_platforms,
            args.cross_platform,
            args.max_workers
        )
    elif args.all:
        # Generate skillsets for all models in registry
        model_types = list(generator.model_registry.keys())
        generator.generate_skillsets_batch(
            model_types,
            hardware_platforms,
            args.cross_platform,
            args.max_workers
        )

if __name__ == "__main__":
    main()