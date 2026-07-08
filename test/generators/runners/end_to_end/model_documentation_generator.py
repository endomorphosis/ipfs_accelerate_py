#!/usr/bin/env python3
"""
Model Documentation Generator for End-to-End Testing Framework

This module generates Markdown documentation for models, explaining the implementation
details, expected behavior, and usage patterns. It integrates with the template system
to use documentation templates from the template database when available.
"""

import os
import sys
import re
import json
import logging
import inspect
import uuid
import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import project utilities (assuming they exist)
try:
    from simple_utils import setup_logging
except ImportError:
    # Define a simple setup_logging function if the import fails
    def setup_logging(logger, level=logging.INFO):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

# Try importing template system components
try:
    from template_database import TemplateDatabase
    from template_renderer import TemplateRenderer
    HAS_TEMPLATE_SYSTEM = True
except ImportError:
    HAS_TEMPLATE_SYSTEM = False

# Set up logging
logger = logging.getLogger(__name__)
setup_logging(logger)

class ModelDocGenerator:
    """Generates comprehensive documentation for model implementations."""
    
    def __init__(self, model_name: str, hardware: str, 
                 skill_path: str, test_path: str, benchmark_path: str,
                 expected_results_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 template_db_path: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize the model documentation generator.
        
        Args:
            model_name: Name of the model being documented
            hardware: Hardware platform the model is running on
            skill_path: Path to the generated skill file
            test_path: Path to the generated test file
            benchmark_path: Path to the generated benchmark file
            expected_results_path: Path to expected results file (optional)
            output_dir: Directory to save the documentation (optional)
            template_db_path: Path to template database (optional)
            verbose: Whether to output verbose logs
        """
        self.model_name = model_name
        self.hardware = hardware
        self.skill_path = skill_path
        self.test_path = test_path
        self.benchmark_path = benchmark_path
        self.expected_results_path = expected_results_path
        self.template_db_path = template_db_path
        
        if output_dir:
            self.output_dir = output_dir
        else:
            # Default to a 'docs' directory next to the script
            self.output_dir = os.path.join(os.path.dirname(script_dir), "model_documentation")
        
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
        # Initialize template system if available
        self.use_templates = False
        if HAS_TEMPLATE_SYSTEM and self.template_db_path and os.path.exists(self.template_db_path):
            try:
                self.template_db = TemplateDatabase(self.template_db_path, verbose=self.verbose)
                self.template_renderer = TemplateRenderer(db_path=self.template_db_path, verbose=self.verbose)
                self.use_templates = True
                logger.debug("Initialized template system for documentation generation")
            except Exception as e:
                logger.warning(f"Failed to initialize template system: {e}")
                self.use_templates = False
    
    def extract_docstrings(self, file_path: str) -> Dict[str, str]:
        """
        Extract docstrings from Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary mapping function/class names to their docstrings
        """
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            
            # Use regex to extract docstrings
            # This is a simple implementation - a real one would use AST or similar
            docstring_map = {}
            
            # Extract module docstring
            module_match = re.search(r'^"""(.*?)"""', file_content, re.DOTALL)
            if module_match:
                docstring_map["module"] = module_match.group(1).strip()
            
            # Extract class docstrings
            class_matches = re.finditer(r'class\s+(\w+).*?:(?:\s+"""(.*?)""")?', file_content, re.DOTALL)
            for match in class_matches:
                class_name = match.group(1)
                docstring = match.group(2)
                if docstring:
                    docstring_map[class_name] = docstring.strip()
            
            # Extract method docstrings
            method_matches = re.finditer(r'def\s+(\w+).*?:(?:\s+"""(.*?)""")?', file_content, re.DOTALL)
            for match in method_matches:
                method_name = match.group(1)
                docstring = match.group(2)
                if docstring:
                    docstring_map[method_name] = docstring.strip()
            
            return docstring_map
            
        except Exception as e:
            logger.error(f"Error extracting docstrings from {file_path}: {str(e)}")
            return {}
    
    def extract_key_code_snippets(self, file_path: str) -> Dict[str, str]:
        """
        Extract key code snippets from Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary mapping snippet names to code
        """
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            
            # Extract snippets based on file type
            snippets = {}
            
            if "skill" in os.path.basename(file_path):
                # Extract relevant parts from skill file
                
                # Extract class definition
                class_match = re.search(r'class\s+\w+.*?(?=\n\n|\Z)', file_content, re.DOTALL)
                if class_match:
                    snippets["class_definition"] = class_match.group(0)
                
                # Extract setup method
                setup_match = re.search(r'def\s+setup.*?(?=\n    def|\Z)', file_content, re.DOTALL)
                if setup_match:
                    snippets["setup_method"] = setup_match.group(0)
                
                # Extract run method
                run_match = re.search(r'def\s+run.*?(?=\n    def|\Z)', file_content, re.DOTALL)
                if run_match:
                    snippets["run_method"] = run_match.group(0)
            
            elif "test" in os.path.basename(file_path):
                # Extract relevant parts from test file
                
                # Extract test class
                test_class_match = re.search(r'class\s+Test\w+.*?(?=\n\nif|\Z)', file_content, re.DOTALL)
                if test_class_match:
                    snippets["test_class"] = test_class_match.group(0)
                
                # Extract test methods
                test_methods = re.finditer(r'def\s+test_\w+.*?(?=\n    def|\n\n|\Z)', file_content, re.DOTALL)
                for i, match in enumerate(test_methods):
                    snippets[f"test_method_{i+1}"] = match.group(0)
            
            elif "benchmark" in os.path.basename(file_path):
                # Extract relevant parts from benchmark file
                
                # Extract benchmark function
                benchmark_match = re.search(r'def\s+benchmark.*?(?=\n\ndef|\n\nif|\Z)', file_content, re.DOTALL)
                if benchmark_match:
                    snippets["benchmark_function"] = benchmark_match.group(0)
                
                # Extract main execution block
                main_match = re.search(r'if\s+__name__\s*==\s*"__main__".*', file_content, re.DOTALL)
                if main_match:
                    snippets["main_execution"] = main_match.group(0)
            
            return snippets
            
        except Exception as e:
            logger.error(f"Error extracting code snippets from {file_path}: {str(e)}")
            return {}
    
    def load_expected_results(self) -> Dict[str, Any]:
        """
        Load expected results from file.
        
        Returns:
            Dictionary with expected results or empty dict if file not found
        """
        if not self.expected_results_path or not os.path.exists(self.expected_results_path):
            logger.warning(f"No expected results found at {self.expected_results_path}")
            return {}
        
        try:
            with open(self.expected_results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading expected results: {str(e)}")
            return {}
    
    def generate_documentation(self) -> str:
        """
        Generate comprehensive Markdown documentation for the model.
        
        Returns:
            Path to the generated documentation file
        """
        logger.info(f"Generating documentation for {self.model_name} on {self.hardware}...")
        
        # Create output directory if it doesn't exist
        model_doc_dir = os.path.join(self.output_dir, self.model_name.replace('/', '_'))
        os.makedirs(model_doc_dir, exist_ok=True)
        
        # Generate documentation file path
        doc_path = os.path.join(model_doc_dir, f"{self.model_name.replace('/', '_')}_{self.hardware}_docs.md")
        
        # Use template system if available
        if self.use_templates:
            try:
                # Determine model family from model name
                model_family = self.template_db.get_model_family(self.model_name)
                if not model_family:
                    logger.warning(f"Could not determine model family for {self.model_name}, falling back to manual generation")
                    return self._generate_documentation_manual(doc_path)
                
                # Extract information from files
                skill_docstrings = self.extract_docstrings(self.skill_path)
                test_docstrings = self.extract_docstrings(self.test_path)
                benchmark_docstrings = self.extract_docstrings(self.benchmark_path)
                
                skill_snippets = self.extract_key_code_snippets(self.skill_path)
                test_snippets = self.extract_key_code_snippets(self.test_path)
                benchmark_snippets = self.extract_key_code_snippets(self.benchmark_path)
                
                expected_results = self.load_expected_results()
                
                # Extract the model class name from the skill file
                model_class_name = self._extract_class_name(self.skill_path)
                
                # Extract API details for better documentation
                api_details = self._extract_api_details(self.skill_path)
                
                # Extract model architecture details
                model_architecture = self._get_model_architecture_description(model_family)
                
                # Extract hardware capability details
                hardware_capabilities = self._get_hardware_capability_details(self.hardware)
                
                # Get model-specific features and use cases
                model_specific_features = self._get_model_specific_features(model_family)
                model_common_use_cases = self._get_model_common_use_cases(model_family)
                
                # Format features and use cases for better markdown rendering
                formatted_features = "\n".join(model_specific_features)
                formatted_use_cases = "\n".join(model_common_use_cases)
                
                # Create variables for template rendering
                variables = {
                    "model_name": self.model_name,
                    "model_family": model_family,
                    "hardware_type": self.hardware,
                    "timestamp": os.environ.get('DATE', datetime.datetime.now().isoformat()),
                    "user": os.environ.get('USER', 'auto-generated'),
                    "test_id": str(uuid.uuid4()),
                    "model_class_name": model_class_name,
                    
                    # Code snippets
                    "class_definition": skill_snippets.get("class_definition", "# No class definition found"),
                    "setup_method": skill_snippets.get("setup_method", "# No setup method found"),
                    "run_method": skill_snippets.get("run_method", "# No run method found"),
                    "test_class": test_snippets.get("test_class", "# No test class found"),
                    "benchmark_function": benchmark_snippets.get("benchmark_function", "# No benchmark function found"),
                    "main_execution": benchmark_snippets.get("main_execution", "# No main execution block found"),
                    
                    # Expected results
                    "expected_results_json": json.dumps(expected_results, indent=2) if expected_results else "{}",
                    "expected_results_available": bool(expected_results),
                    
                    # Hardware-specific information
                    "hardware_specific_notes": self._get_hardware_specific_notes(self.hardware),
                    "hardware_capabilities": hardware_capabilities,
                    
                    # Model architecture details
                    "model_architecture": model_architecture,
                    
                    # API documentation
                    "api_methods": api_details.get("methods", {}),
                    "api_method_count": len(api_details.get("methods", {})),
                    
                    # Module docstring
                    "module_docstring": skill_docstrings.get("module", ""),
                    "class_docstring": skill_docstrings.get(model_class_name, ""),
                    
                    # Model type-specific details
                    "model_specific_features": model_specific_features,
                    "formatted_model_specific_features": formatted_features,
                    "model_common_use_cases": model_common_use_cases,
                    "formatted_model_common_use_cases": formatted_use_cases,
                    
                    # Usage example
                    "usage_example": self._generate_usage_example(model_class_name, model_family)
                }
                
                # Add test method snippets
                test_methods = [k for k in test_snippets.keys() if k.startswith("test_method_")]
                test_methods_content = ""
                for method_key in test_methods:
                    test_methods_content += f"```python\n{test_snippets[method_key]}\n```\n\n"
                variables["test_methods_content"] = test_methods_content
                
                # Generate formatted API documentation
                api_docs = self._format_api_documentation(api_details)
                variables["formatted_api_docs"] = api_docs
                
                # Render documentation template
                try:
                    logger.debug("Rendering documentation template with template system")
                    rendered_doc = self.template_renderer.render_template(
                        model_name=self.model_name,
                        template_type="documentation",
                        hardware_platform=self.hardware,
                        variables=variables
                    )
                    
                    # Write to file
                    with open(doc_path, 'w') as f:
                        f.write(rendered_doc)
                        
                    logger.info(f"Documentation generated with template: {doc_path}")
                    return doc_path
                    
                except Exception as e:
                    logger.warning(f"Error rendering documentation template: {e}")
                    logger.info("Falling back to manual documentation generation")
                    return self._generate_documentation_manual(doc_path)
                    
            except Exception as e:
                logger.warning(f"Error using template system for documentation: {e}")
                logger.info("Falling back to manual documentation generation")
                return self._generate_documentation_manual(doc_path)
        else:
            # No template system available, use manual generation
            return self._generate_documentation_manual(doc_path)
            
    def _extract_class_name(self, file_path: str) -> str:
        """Extract class name from a Python file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Find the class definition
            import re
            class_match = re.search(r'class\s+(\w+)', content)
            if class_match:
                return class_match.group(1)
            return "ModelSkill"  # Default if not found
            
        except Exception as e:
            logger.error(f"Error extracting class name: {str(e)}")
            return "ModelSkill"  # Default if exception
            
    def _extract_api_details(self, file_path: str) -> Dict[str, Any]:
        """Extract API details from a Python file."""
        # Find the class name and methods in the file
        methods = {}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract class name
            import re
            class_match = re.search(r'class\s+([\w_]+)', content)
            class_name = class_match.group(1) if class_match else "Unknown"
                
            # Extract methods
            method_matches = re.finditer(r'def\s+([\w_]+)\s*\((self(?:,\s*[^)]+)*)\)(?:\s*->\s*([^:]+))?\s*:(.*?)(?=\n\s*def|\n\s*$|\n\s*class|$)', content, re.DOTALL)
            
            for match in method_matches:
                method_name = match.group(1)
                params_str = match.group(2)
                return_type = match.group(3)
                method_body = match.group(4).strip()
                
                # Skip private methods
                if method_name.startswith('_'):
                    continue
                    
                # Extract docstring
                docstring_match = re.search(r'"""(.*?)"""', method_body, re.DOTALL)
                if not docstring_match:
                    docstring_match = re.search(r"'''(.*?)'''", method_body, re.DOTALL)
                    
                docstring = docstring_match.group(1).strip() if docstring_match else ""
                
                # Parse parameters
                params = {}
                if params_str:
                    params_list = [p.strip() for p in params_str.split(',')]
                    for param in params_list[1:]:  # Skip 'self'
                        if ':' in param:
                            param_name, param_type = [p.strip() for p in param.split(':', 1)]
                            if '=' in param_type:
                                param_type, default = [p.strip() for p in param_type.split('=', 1)]
                            params[param_name] = {
                                "type": param_type,
                                "description": self._extract_param_description(param_name, docstring)
                            }
                        else:
                            param_name = param
                            if '=' in param_name:
                                param_name = param_name.split('=', 1)[0].strip()
                            params[param_name] = {
                                "type": "Any",
                                "description": self._extract_param_description(param_name, docstring)
                            }
                
                # Parse return type
                return_desc = self._extract_return_description(docstring)
                returns = {
                    "type": return_type.strip() if return_type else "Any",
                    "description": return_desc if return_desc else "Return value"
                }
                
                # Store method info
                methods[method_name] = {
                    "docstring": docstring,
                    "parameters": params,
                    "returns": returns
                }
                
            return {
                "class_name": class_name,
                "methods": methods
            }
            
        except Exception as e:
            logger.error(f"Error extracting API details: {str(e)}")
            return {
                "class_name": "Unknown",
                "methods": {}
            }
            
    def _extract_param_description(self, param_name: str, docstring: str) -> str:
        """Extract parameter description from docstring."""
        if not docstring:
            return f"Parameter '{param_name}'"
            
        # Look for parameter in docstring (supporting both numpy and google docstring formats)
        import re
        
        # Try Google style first (Args: param_name: description)
        param_pattern = rf"Args:.*?{param_name}\s*:\s*(.*?)(?:\n\s*\w+\s*:|$)"
        param_match = re.search(param_pattern, docstring, re.DOTALL)
        
        if param_match:
            return param_match.group(1).strip()
            
        # Try Numpy style (Parameters: param_name : type, description)
        param_pattern = rf"Parameters.*?{param_name}\s*:.*?\n\s+(.*?)(?:\n\s*\w+\s*:|$)"
        param_match = re.search(param_pattern, docstring, re.DOTALL)
        
        if param_match:
            return param_match.group(1).strip()
            
        return f"Parameter '{param_name}'"
        
    def _extract_return_description(self, docstring: str) -> str:
        """Extract return description from docstring."""
        if not docstring:
            return "Return value"
            
        # Look for return description in docstring
        import re
        
        # Try Google style (Returns: description)
        return_pattern = r"Returns:.*?\n\s+(.*?)(?:\n\s*\w+\s*:|$)"
        return_match = re.search(return_pattern, docstring, re.DOTALL)
        
        if return_match:
            return return_match.group(1).strip()
            
        # Try Numpy style (Returns: type, description)
        return_pattern = r"Returns.*?\n\s+(.*?)(?:\n\s*\w+\s*:|$)"
        return_match = re.search(return_pattern, docstring, re.DOTALL)
        
        if return_match:
            return return_match.group(1).strip()
            
        return "Return value"
        
    def _format_api_documentation(self, api_details: Dict[str, Any]) -> str:
        """Format API documentation for markdown."""
        if not api_details or 'methods' not in api_details:
            return "No API documentation available."
            
        # Create markdown for API docs
        md = ""
        
        for method_name, method_info in api_details.get('methods', {}).items():
            md += f"### `{method_name}`\n\n"
            
            # Add docstring
            if 'docstring' in method_info and method_info['docstring']:
                md += f"{method_info['docstring']}\n\n"
                
            # Add parameters
            if 'parameters' in method_info and method_info['parameters']:
                md += "**Parameters**:\n\n"
                for param, param_info in method_info['parameters'].items():
                    param_type = param_info.get('type', 'Any')
                    param_desc = param_info.get('description', 'No description available.')
                    md += f"- `{param}` (`{param_type}`): {param_desc}\n"
                md += "\n"
                
            # Add return value
            if 'returns' in method_info:
                returns_type = method_info['returns'].get('type', 'Any')
                returns_desc = method_info['returns'].get('description', 'No description available.')
                md += f"**Returns**: `{returns_type}`: {returns_desc}\n\n"
                
        return md
    
    def _get_model_architecture_description(self, model_family: str) -> str:
        """Get model architecture description based on model family."""
        if model_family == "text_embedding":
            return """This text embedding model uses a Transformer-based architecture:

1. **Embedding Layer**: Converts token IDs into embeddings, includes token, position, and (optionally) segment embeddings
2. **Transformer Encoder**: Multiple layers of self-attention and feed-forward networks
3. **Pooling Layer**: Creates a single vector representation from token embeddings, using strategies like:
   - CLS token pooling: uses the first [CLS] token's embedding
   - Mean pooling: averages all token embeddings
   - Max pooling: takes element-wise maximum across token embeddings

The embedding output is typically a fixed-size vector (768 dimensions for base models) that captures the semantic meaning of the input text."""
        elif model_family == "text_generation":
            return """This text generation model uses a Transformer-based architecture:

1. **Embedding Layer**: Converts token IDs into embeddings with position information
2. **Transformer Layers**: Multiple layers combining:
   - Self-attention mechanisms (allowing the model to focus on different parts of the input)
   - Feed-forward neural networks
3. **Language Modeling Head**: Projects hidden states to vocabulary distribution

The model is typically trained with a causal language modeling objective, meaning it predicts the next token based on all previous tokens. During inference, it generates text autoregressively by repeatedly sampling from the output distribution and feeding the selected token back as input."""
        elif model_family == "vision":
            return """This vision model uses a Transformer-based architecture adapted for images:

1. **Patch Embedding**: Divides the input image into patches and projects each to a fixed-size embedding
2. **Position Embedding**: Adds position information to retain spatial relationships
3. **Transformer Encoder**: Multiple layers of self-attention and feed-forward networks
4. **Output Layer**: Context-dependent per task:
   - Class token ([CLS]) for image classification
   - All patch embeddings for dense tasks like segmentation
   - Projection layer for embedding tasks

The model processes images as a sequence of patch tokens, similar to how text transformers process word tokens. Vision models excel at tasks like image classification, object detection, and visual feature extraction. Their self-attention mechanism allows them to focus on different parts of the image with varying importance."""
        elif model_family == "audio":
            return """This audio model uses a specialized architecture for processing audio inputs:

1. **Feature Extraction**: Converts raw audio waveforms to spectrograms or other frequency-domain representations
2. **Encoder**: Processes audio features, typically using:
   - Convolutional layers to capture local patterns
   - Transformer layers for long-range dependencies
3. **Decoder** (for speech-to-text models): Generates text output from encoded audio representations
4. **Task-Specific Heads**: Specialized output layers for tasks like:
   - Speech recognition (text output)
   - Audio classification
   - Audio embedding generation

The model is designed to handle variable-length audio inputs and extract meaningful patterns from temporal audio signals. Audio models can process different types of sound including speech, music, and environmental audio. They are optimized to handle the unique challenges of audio data such as time-varying patterns and frequency dynamics."""
        elif model_family == "multimodal":
            return """This multimodal model processes multiple types of inputs (e.g., text and images) together:

1. **Specialized Encoders**: Separate encoders for different modality types:
   - Text Encoder: Processes text inputs with transformer architecture
   - Vision Encoder: Processes image inputs with vision transformer
2. **Cross-Modal Fusion**: Mechanisms to combine information across modalities:
   - Early fusion: Combining raw inputs before encoding
   - Late fusion: Combining encoded representations
   - Attention mechanisms: Allowing modalities to attend to each other
3. **Unified Representation**: Creation of joint embedding space for both modalities
4. **Task-Specific Heads**: Output layers specialized for tasks like:
   - Visual question answering
   - Image-text retrieval
   - Multimodal classification

The model architecture aligns representations from different modalities to enable reasoning across them. This allows the model to understand relationships between different types of data, such as relating images to their textual descriptions or answering questions about visual content. Multimodal models are particularly powerful for tasks that require integrating information across different sensory domains."""
        else:
            return f"""This model's specific architecture is based on the {model_family} family:

1. **Input Processing**: Takes {model_family.replace('_', ' ')} inputs and converts them to model representations
2. **Model Backbone**: Uses a transformer-based architecture to process inputs
3. **Output Layer**: Produces appropriate outputs for the model's primary task

The model follows standard practices for {model_family.replace('_', ' ')} models with potential model-specific enhancements."""
    
    def _get_hardware_capability_details(self, hardware: str) -> Dict[str, Any]:
        """Get detailed hardware capability information."""
        if hardware == "cpu":
            return {
                "description": "Standard CPU implementation using PyTorch",
                "optimizations": [
                    "Multi-threading for batch processing",
                    "SIMD instructions (AVX, SSE) where available",
                    "Memory-efficient operations",
                    "Operation fusion for better cache utilization"
                ],
                "limitations": [
                    "Lower throughput compared to specialized hardware",
                    "Higher latency for large batch sizes",
                    "Memory bandwidth limitations for large models"
                ],
                "best_for": [
                    "Development and testing",
                    "Small batch sizes",
                    "Deployment on standard servers without GPUs",
                    "Environments where power efficiency isn't critical"
                ],
                "performance_characteristics": {
                    "typical_throughput_factor": 1.0,  # Baseline
                    "memory_usage": "Moderate",
                    "power_efficiency": "Low"
                }
            }
        elif hardware == "cuda":
            return {
                "description": "NVIDIA GPU implementation using CUDA and PyTorch",
                "optimizations": [
                    "Tensor core acceleration for matrix operations",
                    "Mixed-precision (FP16) computation",
                    "CUDA kernel fusion",
                    "Parallel execution across CUDA cores",
                    "Optimized memory access patterns"
                ],
                "limitations": [
                    "Requires NVIDIA GPU hardware",
                    "CUDA driver and toolkit dependencies",
                    "Limited by GPU memory capacity",
                    "Higher power consumption"
                ],
                "best_for": [
                    "High-throughput batch processing",
                    "Training and fine-tuning",
                    "Production deployments requiring maximum performance",
                    "Environments with dedicated GPU hardware"
                ],
                "performance_characteristics": {
                    "typical_throughput_factor": 10.0,  # 10x CPU
                    "memory_usage": "High",
                    "power_efficiency": "Moderate"
                }
            }
        elif hardware == "webgpu":
            return {
                "description": "Browser-based GPU acceleration using WebGPU API",
                "optimizations": [
                    "GPU acceleration in browser environment",
                    "Optimized shader programs for neural network operations",
                    "Compute shader optimizations for matrix multiply",
                    "Browser compute capability detection",
                    "Optimized data transfer between CPU and GPU"
                ],
                "limitations": [
                    "Requires browser with WebGPU support",
                    "Limited by browser-imposed memory constraints",
                    "Performance varies by browser implementation",
                    "Some operations may fall back to CPU"
                ],
                "best_for": [
                    "Client-side inference in web applications",
                    "Privacy-preserving local inference",
                    "Interactive web demos",
                    "Edge deployment without specialized hardware"
                ],
                "performance_characteristics": {
                    "typical_throughput_factor": 3.0,  # 3x CPU
                    "memory_usage": "Moderate",
                    "power_efficiency": "Moderate-High"
                }
            }
        else:
            return {
                "description": f"Implementation for {hardware} hardware platform",
                "optimizations": [
                    f"Platform-specific optimizations for {hardware}",
                    "Specialized memory management",
                    "Hardware-aware operation scheduling"
                ],
                "limitations": [
                    f"Requires {hardware} hardware support",
                    "May have platform-specific dependencies"
                ],
                "best_for": [
                    f"Deployments targeting {hardware} hardware",
                    "Specific use cases where this hardware excels"
                ],
                "performance_characteristics": {
                    "typical_throughput_factor": 2.0,  # Generic estimate
                    "memory_usage": "Varies",
                    "power_efficiency": "Varies"
                }
            }
            
    def _get_model_specific_features(self, model_family: str) -> List[str]:
        """Get model-specific features based on model family."""
        if model_family == "text_embedding":
            return [
                "**Semantic Text Representation**: Creates vector representations that capture meaning rather than just keywords",
                "**Fixed-Dimension Embeddings**: Produces consistent vector sizes (typically 768 dimensions for base models)",
                "**Contextual Understanding**: Captures word meaning based on surrounding context",
                "**Bidirectional Context**: Processes text in both directions for comprehensive understanding",
                "**Pooling Strategies**: Supports CLS token, mean, and max pooling for different use cases",
                "**Cross-Lingual Capabilities**: Many models can map similar meanings across languages",
                "**Token-Level Features**: Can generate embeddings for individual tokens or whole sequences",
                "**Similarity Metrics**: Optimized for cosine similarity comparison between embeddings",
                "**Hierarchical Representations**: Captures both token-level and sequence-level information",
                "**Fine-Tuning Capability**: Adaptable to domain-specific embedding tasks"
            ]
        elif model_family == "text_generation":
            return [
                "**Autoregressive Generation**: Generates text one token at a time, conditioned on previous tokens",
                "**Controllable Parameters**: Adjustable temperature, top-k, and top-p sampling for varied outputs",
                "**Context Window**: Typical context length of 1K-8K tokens (model dependent)",
                "**Prompt Engineering**: Supports various prompting techniques for task specification",
                "**In-Context Learning**: Can follow examples provided in the context window",
                "**Task Adaptation**: Capable of adapting to various text generation tasks through prompting",
                "**Memory Efficiency**: Optimized KV-cache for efficient token generation",
                "**Beam Search**: Support for beam search and sampling strategies",
                "**Prefix Conditioning**: Can condition generation on specific prefixes",
                "**Early Stopping**: Configurable stopping criteria for generation"
            ]
        elif model_family == "vision":
            return [
                "**Patch-Based Processing**: Processes images as sequences of patches",
                "**Visual Feature Extraction**: Creates rich representations of visual content",
                "**Resolution Flexibility**: Handles different input image resolutions",
                "**Global Context**: Self-attention mechanism captures relationships between all image patches",
                "**Pre-trained Visual Knowledge**: Utilizes knowledge from pre-training on large image datasets",
                "**Position-Aware Processing**: Maintains spatial relationships between image regions",
                "**Transfer Learning**: Easily adaptable to downstream vision tasks",
                "**Multi-Scale Feature Maps**: Captures both fine and coarse visual features",
                "**Attention Visualization**: Supports visualization of regions the model focuses on",
                "**Class Token**: Uses special classification token for image-level tasks"
            ]
        elif model_family == "audio":
            return [
                "**Spectrogram Processing**: Converts audio waveforms to time-frequency representations",
                "**Automatic Speech Recognition**: Transcribes spoken content to text",
                "**Variable-Length Processing**: Handles audio inputs of different durations",
                "**Noise Robustness**: Many models are trained to be robust to environmental noise",
                "**Multi-Lingual Support**: Many models support multiple languages (model dependent)",
                "**Speaker-Invariant Features**: Works across different speakers and accents",
                "**Temporal Feature Extraction**: Captures time-dependent patterns in audio",
                "**Audio Classification**: Can classify audio into predefined categories",
                "**Sequence Modeling**: Models temporal dependencies in audio signals",
                "**Acoustic Feature Learning**: Learns meaningful features from raw or processed audio"
            ]
        elif model_family == "multimodal":
            return [
                "**Cross-Modal Understanding**: Processes and relates information across different modalities",
                "**Joint Embedding Space**: Creates unified representations for different input types",
                "**Modal Alignment**: Aligns representations from different modalities (e.g., text and images)",
                "**Zero-Shot Transfer**: Applies knowledge across modalities without specific training",
                "**Multi-Task Capability**: Handles various tasks involving multiple input types",
                "**Contrastive Learning**: Many models use contrastive objectives for alignment",
                "**Modal Fusion**: Combines information from different modalities at various levels",
                "**Attention Mechanisms**: Uses cross-attention to relate elements across modalities",
                "**Modal Grounding**: Grounds concepts in one modality using another",
                "**Flexible Input Handling**: Processes various combinations of input types"
            ]
        else:
            return [
                f"**Specialized {model_family.replace('_', ' ').title()} Processing**: Optimized for {model_family.replace('_', ' ')} data",
                "**Transformer-Based Architecture**: Uses attention mechanisms for capturing relationships",
                "**Fine-Tuning Capability**: Adaptable to specialized downstream tasks",
                "**Pre-Trained Feature Extraction**: Leverages knowledge from pre-training",
                "**Domain-Specific Optimizations**: Includes optimizations specific to this model family",
                "**Transfer Learning Support**: Applies learned knowledge to new but related tasks",
                "**Contextual Understanding**: Processes inputs in context rather than in isolation",
                "**Scalable Architecture**: Scales with model size for improved performance"
            ]
            
    def _get_model_common_use_cases(self, model_family: str) -> List[str]:
        """Get common use cases based on model family."""
        if model_family == "text_embedding":
            return [
                "**Semantic Search**: Finding documents based on meaning rather than keywords",
                "**Information Retrieval**: Retrieving relevant documents from large collections",
                "**Document Clustering**: Grouping similar documents based on content",
                "**Text Similarity**: Measuring semantic similarity between texts",
                "**Recommendation Systems**: Suggesting content based on embedding similarity",
                "**Classification Features**: Providing features for text classification models",
                "**Question Answering Systems**: Finding relevant passages for questions",
                "**Anomaly Detection**: Identifying outlier texts that differ semantically",
                "**Content-Based Filtering**: Filtering content based on semantic properties",
                "**Cross-Lingual Applications**: Finding similar content across languages"
            ]
        elif model_family == "text_generation":
            return [
                "**Content Creation**: Generating articles, stories, and creative content",
                "**Code Generation**: Writing and completing code in various languages",
                "**Conversational AI**: Powering chatbots and conversational agents",
                "**Text Summarization**: Creating concise summaries of longer documents",
                "**Question Answering**: Generating answers to queries based on knowledge",
                "**Translation**: Converting text between languages (with appropriate models)",
                "**Text Completion**: Suggesting completions for partially written text",
                "**Data Augmentation**: Generating variations of text for training data",
                "**Personalized Content**: Creating customized content for specific audiences",
                "**Report Generation**: Producing structured reports from data or specifications"
            ]
        elif model_family == "vision":
            return [
                "**Image Classification**: Categorizing images into classes",
                "**Visual Search**: Finding similar images based on content",
                "**Object Recognition**: Identifying objects within images",
                "**Feature Extraction**: Providing image features for downstream tasks",
                "**Image Tagging**: Automatically tagging images with relevant labels",
                "**Visual Quality Assessment**: Evaluating image quality",
                "**Image Retrieval**: Finding relevant images from large collections",
                "**Zero-Shot Classification**: Classifying into categories not seen during training",
                "**Transfer Learning**: Using pre-trained visual knowledge for new tasks",
                "**Visual Representation Learning**: Learning rich image representations"
            ]
        elif model_family == "audio":
            return [
                "**Speech Recognition**: Converting spoken language to text",
                "**Audio Transcription**: Creating text transcripts from audio recordings",
                "**Meeting Transcription**: Generating text records of meetings",
                "**Voice Command Systems**: Powering voice-activated commands",
                "**Audio Classification**: Categorizing sounds into predefined classes",
                "**Subtitle Generation**: Creating subtitles for videos automatically",
                "**Audio Search**: Finding specific content in audio recordings",
                "**Voice Assistants**: Powering conversational voice interfaces",
                "**Language Identification**: Determining the language being spoken",
                "**Acoustic Event Detection**: Identifying specific sounds in audio streams"
            ]
        elif model_family == "multimodal":
            return [
                "**Visual Question Answering**: Answering questions about images",
                "**Image Captioning**: Generating text descriptions for images",
                "**Cross-Modal Retrieval**: Finding matching content across modalities",
                "**Multimodal Classification**: Classifying using multiple input types",
                "**Image-Text Matching**: Determining relevance between images and text",
                "**Multimodal Search**: Searching using combinations of text, image, etc.",
                "**Visual Grounding**: Locating objects in images based on text descriptions",
                "**Document Understanding**: Processing documents with text and images",
                "**Product Search**: Finding products based on images and descriptions",
                "**Multimodal Content Creation**: Creating content that combines multiple modalities"
            ]
        else:
            return [
                f"**{model_family.replace('_', ' ').title()} Specific Tasks**: Specialized for {model_family.replace('_', ' ')} domain",
                "**Feature Extraction**: Creating rich features for downstream applications",
                "**Transfer Learning**: Applying pre-trained knowledge to new tasks",
                "**Domain-Specific Analysis**: Analyzing data specific to this domain",
                "**Data Understanding**: Creating meaningful representations of input data",
                "**Predictive Modeling**: Building predictive models within this domain",
                "**Classification**: Categorizing inputs into predefined classes",
                "**Pattern Recognition**: Identifying patterns specific to this domain"
            ]
            
    def _generate_usage_example(self, model_class_name: str, model_family: str) -> str:
        """Generate a usage example based on model family and class name."""
        if model_family == "text_embedding":
            return f"""
```python
# Import the skill class
from {self.model_name.replace('-', '_').replace('/', '_')}_{self.hardware}_skill import {model_class_name}

# Create an instance
skill = {model_class_name}()

# Set up the model
success = skill.setup()
if not success:
    raise RuntimeError("Failed to set up the model")

# Run embedding on a single input
result = skill.run("This is a sample text for embedding generation")

# Access the embeddings
embeddings = result["embeddings"]
print(f"Generated embeddings with shape: {embeddings.shape}")

# Run embedding on a batch of inputs
batch_result = skill.run([
    "First sample text",
    "Second sample text",
    "Third sample text"
])

batch_embeddings = batch_result["embeddings"]
print(f"Generated batch embeddings with shape: {batch_embeddings.shape}")

# Calculate similarity between embeddings
from scipy.spatial.distance import cosine
similarity = 1 - cosine(batch_embeddings[0], batch_embeddings[1])
print(f"Cosine similarity between first and second embeddings: {similarity:.4f}")

# Clean up resources
skill.cleanup()
```"""
        elif model_family == "text_generation":
            return f"""
```python
# Import the skill class
from {self.model_name.replace('-', '_').replace('/', '_')}_{self.hardware}_skill import {model_class_name}

# Create an instance
skill = {model_class_name}()

# Set up the model
success = skill.setup()
if not success:
    raise RuntimeError("Failed to set up the model")

# Run text generation with a prompt
result = skill.run(
    "Write a short poem about artificial intelligence",
    max_length=100,
    temperature=0.7,
    top_p=0.9
)

# Access the generated text
generated_text = result["generated_text"]
print(f"Generated text:\\n{generated_text}")

# Try with different parameters for creative vs deterministic outputs
creative_result = skill.run(
    "Write a short story beginning with: Once upon a time in Silicon Valley",
    max_length=200,
    temperature=0.9,  # Higher temperature for more creative output
    top_p=0.95
)

deterministic_result = skill.run(
    "Write a short story beginning with: Once upon a time in Silicon Valley",
    max_length=200,
    temperature=0.1,  # Lower temperature for more deterministic output
    top_p=0.8
)

# Clean up resources
skill.cleanup()
```"""
        elif model_family == "vision":
            return f"""
```python
# Import the skill class
from {self.model_name.replace('-', '_').replace('/', '_')}_{self.hardware}_skill import {model_class_name}

# Create an instance
skill = {model_class_name}()

# Set up the model
success = skill.setup()
if not success:
    raise RuntimeError("Failed to set up the model")

# Load and preprocess an image
from PIL import Image
import numpy as np

image = Image.open("sample_image.jpg")
# Resize image to the model's expected input size
image = image.resize((224, 224))
# Convert to numpy array and normalize
image_array = np.array(image) / 255.0

# Run the model
result = skill.run(image_array)

# Access the output
features = result["features"]
print(f"Extracted image features with shape: {features.shape}")

# For classification models, get predicted class
if "logits" in result:
    logits = result["logits"]
    predicted_class = np.argmax(logits)
    print(f"Predicted class: {predicted_class}")
    
# Process a batch of images
image_batch = [
    np.array(Image.open("image1.jpg").resize((224, 224))) / 255.0,
    np.array(Image.open("image2.jpg").resize((224, 224))) / 255.0
]
batch_result = skill.run(image_batch)

# Clean up resources
skill.cleanup()
```"""
        elif model_family == "audio":
            return f"""
```python
# Import the skill class
from {self.model_name.replace('-', '_').replace('/', '_')}_{self.hardware}_skill import {model_class_name}

# Create an instance
skill = {model_class_name}()

# Set up the model
success = skill.setup()
if not success:
    raise RuntimeError("Failed to set up the model")

# Load audio data
import soundfile as sf
audio_data, sample_rate = sf.read("sample_audio.wav")

# Ensure audio is in the right format (models typically expect mono audio)
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0]  # Take first channel if stereo

# Run the model for transcription
result = skill.run(
    audio_data,
    sample_rate=sample_rate,
    return_timestamps=True  # Optional, get word-level timestamps
)

# Access the transcription results
transcription = result["text"]
print(f"Transcribed text: {transcription}")

# If timestamps were requested
if "timestamps" in result:
    timestamps = result["timestamps"]
    print("Word-level timestamps:")
    for word, time in timestamps:
        print(f"  {word}: {time[0]:.2f}s - {time[1]:.2f}s")

# Clean up resources
skill.cleanup()
```"""
        elif model_family == "multimodal":
            return f"""
```python
# Import the skill class
from {self.model_name.replace('-', '_').replace('/', '_')}_{self.hardware}_skill import {model_class_name}

# Create an instance
skill = {model_class_name}()

# Set up the model
success = skill.setup()
if not success:
    raise RuntimeError("Failed to set up the model")

# Load image
from PIL import Image
import numpy as np
image = Image.open("sample_image.jpg")
image = image.resize((224, 224))  # Resize to expected dimensions
image_array = np.array(image) / 255.0

# For image-text models like CLIP
text_queries = [
    "a photo of a cat",
    "a photo of a dog",
    "a landscape photo",
    "a photo of food"
]

# Run multimodal processing
result = skill.run({
    "image": image_array,
    "text": text_queries
})

# Access similarity scores between image and text
if "similarity_scores" in result:
    similarity_scores = result["similarity_scores"]
    print("Similarity scores:")
    for query, score in zip(text_queries, similarity_scores):
        print(f"  {query}: {score:.4f}")
    
    # Get best matching description
    best_match_idx = np.argmax(similarity_scores)
    print(f"Best match: {text_queries[best_match_idx]}")

# For visual question answering models
vqa_result = skill.run({
    "image": image_array,
    "question": "What is shown in this image?"
})

if "answer" in vqa_result:
    print(f"Answer: {vqa_result['answer']}")

# Clean up resources
skill.cleanup()
```"""
        else:
            return f"""
```python
# Import the skill class
from {self.model_name.replace('-', '_').replace('/', '_')}_{self.hardware}_skill import {model_class_name}

# Create an instance
skill = {model_class_name}()

# Set up the model
success = skill.setup()
if not success:
    raise RuntimeError("Failed to set up the model")

# Prepare input data appropriate for this model type
input_data = "Sample input for this model type"

# Run the model
result = skill.run(input_data)

# Process the output
print(f"Model output: {result}")

# Clean up resources
skill.cleanup()
```"""
            
    def _get_hardware_specific_notes(self, hardware: str) -> str:
        """
        Get hardware-specific notes for documentation.
        
        Args:
            hardware: Hardware platform
            
        Returns:
            Hardware-specific notes as a string
        """
        if hardware == "cpu":
            return """- **Standard CPU Implementation**: Optimized for general CPU execution
- **Multi-threading Support**: Uses PyTorch's multi-threading for parallel processing
- **SIMD Instructions**: Leverages AVX, SSE where available for vector operations
- **Memory-Efficient Operations**: Optimized for host memory access patterns
- **Portability**: Works on virtually any system with compatible Python environment
- **Typical Use Cases**: Development, testing, small batch processing, systems without GPUs
- **Performance Characteristics**: Balanced performance, limited by CPU cores and memory bandwidth"""
        elif hardware == "cuda":
            return """- **NVIDIA GPU Optimization**: Specifically tuned for NVIDIA GPUs using CUDA
- **Tensor Core Acceleration**: Uses Tensor Cores for mixed-precision matrix operations (on supported GPUs)
- **Parallel Execution**: Leverages thousands of CUDA cores for highly parallel computation
- **Optimized Memory Access**: Efficient GPU memory usage patterns with coalesced memory access
- **Requirements**: CUDA toolkit and compatible NVIDIA drivers
- **Best For**: Training and high-throughput inference on NVIDIA hardware
- **Performance Characteristics**: Highest throughput with large batch sizes, significantly faster than CPU"""
        elif hardware == "rocm":
            return """- **AMD GPU Optimization**: Specifically optimized for AMD GPUs using ROCm
- **Matrix Acceleration**: Uses AMD matrix cores for accelerated operations where available
- **HIP Programming Model**: Uses HIP (Heterogeneous-Compute Interface for Portability)
- **Environment Variables**: May require ROCm-specific environment variables for optimal performance
- **Requirements**: ROCm installation and compatible AMD hardware
- **Best For**: Training and inference on AMD hardware
- **Performance Characteristics**: High throughput with good performance/cost ratio"""
        elif hardware == "mps":
            return """- **Apple Silicon Optimization**: Specifically optimized for Apple M-series chips
- **Metal Performance Shaders**: Uses Apple's MPS for hardware acceleration
- **Unified Memory Architecture**: Takes advantage of Apple's unified memory between CPU and GPU
- **Power Efficiency**: Optimized for battery life and thermal constraints
- **Requirements**: macOS and Apple Silicon hardware (M1/M2/M3 series)
- **Best For**: Deployment on Apple devices, development on MacBooks
- **Performance Characteristics**: Excellent performance/watt ratio, lower peak performance than discrete GPUs"""
        elif hardware == "openvino":
            return """- **Intel Hardware Optimization**: Tuned specifically for Intel CPUs, GPUs, and accelerators
- **Model Compilation**: Uses OpenVINO's model compiler for platform-specific optimizations
- **Inference Engine**: Leverages OpenVINO Inference Engine for optimal execution
- **Quantization Support**: Int8 quantization for improved performance
- **Requirements**: OpenVINO Runtime installation and compatible Intel hardware
- **Best For**: Deployment on Intel hardware, especially edge devices
- **Performance Characteristics**: Optimized for inference with great CPU utilization"""
        elif hardware == "qnn":
            return """- **Qualcomm AI Engine Optimization**: Specifically tuned for Qualcomm Snapdragon processors
- **Hardware Accelerator Usage**: Leverages Hexagon DSP, Adreno GPU, and Kryo CPU
- **Specialized Kernels**: Uses optimized kernels for neural network operations
- **Power Efficiency**: Designed for mobile battery constraints
- **Requirements**: Qualcomm AI Engine SDK and compatible Snapdragon device
- **Best For**: Mobile deployment, edge devices, and battery-constrained applications
- **Performance Characteristics**: Excellent performance/watt ratio, optimized for Snapdragon"""
        elif hardware == "webnn":
            return """- **Web Neural Network API**: Uses the W3C WebNN API standard for web-based neural computation
- **Browser Acceleration**: Leverages hardware acceleration through the browser
- **Cross-Platform**: Works on different operating systems through the browser
- **Fallback Mechanisms**: Falls back to WebAssembly when WebNN API isn't supported
- **Browser Compatibility**: Best performance on Edge (best WebNN support) and Chrome
- **Best For**: Client-side inference in web applications
- **Performance Characteristics**: Good performance for web deployment, varies by browser implementation"""
        elif hardware == "webgpu":
            return """- **Web GPU API**: Uses the WebGPU API for GPU acceleration in browsers
- **Compute Shader Support**: Leverages compute shaders for neural network operations
- **Browser Optimization**: Specific optimizations for different browsers (Firefox optimal for audio models)
- **Shader Precompilation**: Supports shader precompilation for faster startup
- **Parallel Model Loading**: Optimized loading of model components in parallel
- **Memory Management**: Careful management of GPU memory within browser constraints
- **Requirements**: Modern browser with WebGPU API support
- **Best For**: Client-side inference in web applications requiring GPU acceleration
- **Performance Characteristics**: Best GPU-accelerated performance in browser environments"""
        elif hardware == "samsung":
            return """- **Samsung NPU Optimization**: Specifically tuned for Samsung Neural Processing Units
- **One UI Integration**: Optimized for Samsung's Android implementation
- **Power Efficiency**: Highly optimized for battery-constrained mobile devices
- **Model Compression**: Support for optimized model formats for NPU execution
- **Requirements**: One UI 5.0+ and compatible Samsung devices
- **Best For**: Samsung mobile and edge devices
- **Performance Characteristics**: Excellent performance/watt ratio, optimized for Samsung hardware"""
        else:
            return f"""- **{hardware.title()} Implementation**: Specialized implementation for {hardware} hardware
- **Platform-Specific Optimizations**: Tuned for optimal performance on {hardware} hardware
- **Performance Characteristics**: Varies based on specific {hardware} hardware capabilities"""
    
    def _generate_documentation_manual(self, doc_path: str) -> str:
        """
        Generate documentation manually (without using templates).
        
        Args:
            doc_path: Path where documentation should be written
            
        Returns:
            Path to the generated documentation file
        """
        logger.info("Generating documentation manually (without templates)")
        
        # Extract information from files
        skill_docstrings = self.extract_docstrings(self.skill_path)
        test_docstrings = self.extract_docstrings(self.test_path)
        benchmark_docstrings = self.extract_docstrings(self.benchmark_path)
        
        skill_snippets = self.extract_key_code_snippets(self.skill_path)
        test_snippets = self.extract_key_code_snippets(self.test_path)
        benchmark_snippets = self.extract_key_code_snippets(self.benchmark_path)
        
        expected_results = self.load_expected_results()
        
        # Generate documentation file
        with open(doc_path, 'w') as f:
            f.write(f"# {self.model_name} Implementation on {self.hardware.upper()}\n\n")
            
            # Overview section
            f.write("## Overview\n\n")
            f.write(f"This document describes the implementation and testing of the {self.model_name} model ")
            f.write(f"on {self.hardware} hardware. It includes details about the skill implementation, ")
            f.write("test cases, benchmarking methodology, and expected results.\n\n")
            
            # Model information
            f.write("## Model Information\n\n")
            f.write(f"- **Model**: {self.model_name}\n")
            f.write(f"- **Hardware**: {self.hardware}\n")
            
            if expected_results:
                # Add performance metrics if available
                if "metrics" in expected_results:
                    f.write("- **Performance Metrics**:\n")
                    metrics = expected_results["metrics"]
                    for metric_name, metric_value in metrics.items():
                        f.write(f"  - {metric_name}: {metric_value}\n")
            
            f.write("\n")
            
            # Skill implementation
            f.write("## Skill Implementation\n\n")
            f.write("The skill implementation is responsible for loading and running the model.\n\n")
            
            if "class_definition" in skill_snippets:
                f.write("### Class Definition\n\n")
                f.write("```python\n" + skill_snippets["class_definition"] + "\n```\n\n")
            
            if "setup_method" in skill_snippets:
                f.write("### Setup Method\n\n")
                f.write("```python\n" + skill_snippets["setup_method"] + "\n```\n\n")
            
            if "run_method" in skill_snippets:
                f.write("### Run Method\n\n")
                f.write("```python\n" + skill_snippets["run_method"] + "\n```\n\n")
            
            # Test implementation
            f.write("## Test Implementation\n\n")
            f.write("The test implementation validates that the model produces correct outputs.\n\n")
            
            if "test_class" in test_snippets:
                f.write("### Test Class\n\n")
                f.write("```python\n" + test_snippets["test_class"] + "\n```\n\n")
            
            # Find all test methods
            test_methods = [k for k in test_snippets.keys() if k.startswith("test_method_")]
            if test_methods:
                f.write("### Test Methods\n\n")
                for method_key in test_methods:
                    f.write("```python\n" + test_snippets[method_key] + "\n```\n\n")
            
            # Benchmark implementation
            f.write("## Benchmark Implementation\n\n")
            f.write("The benchmark measures the performance of the model on this hardware.\n\n")
            
            if "benchmark_function" in benchmark_snippets:
                f.write("### Benchmark Function\n\n")
                f.write("```python\n" + benchmark_snippets["benchmark_function"] + "\n```\n\n")
            
            if "main_execution" in benchmark_snippets:
                f.write("### Execution\n\n")
                f.write("```python\n" + benchmark_snippets["main_execution"] + "\n```\n\n")
            
            # Expected results
            f.write("## Expected Results\n\n")
            
            if expected_results:
                f.write("The model should produce outputs matching these expected results:\n\n")
                f.write("```json\n" + json.dumps(expected_results, indent=2) + "\n```\n\n")
                
                # Add specific input/output examples if available
                if "input" in expected_results and "output" in expected_results:
                    f.write("### Input/Output Example\n\n")
                    f.write("**Input:**\n")
                    f.write("```json\n" + json.dumps(expected_results["input"], indent=2) + "\n```\n\n")
                    f.write("**Expected Output:**\n")
                    f.write("```json\n" + json.dumps(expected_results["output"], indent=2) + "\n```\n\n")
            else:
                f.write("No expected results are available yet. Run the tests and update the expected results.\n\n")
            
            # Hardware-specific notes
            f.write("## Hardware-Specific Notes\n\n")
            f.write(self._get_hardware_specific_notes(self.hardware))
            f.write("\n\n")
            
            # Implementation history
            f.write("## Implementation History\n\n")
            f.write("- Initial implementation: AUTO-GENERATED\n")
            f.write(f"- Last updated: {os.environ.get('USER', 'unknown')}, {os.environ.get('DATE', 'auto-generated')}\n\n")
        
        logger.info(f"Documentation generated manually: {doc_path}")
        return doc_path


def generate_model_documentation(model_name: str, hardware: str, 
                                skill_path: str, test_path: str, benchmark_path: str,
                                expected_results_path: Optional[str] = None,
                                output_dir: Optional[str] = None,
                                template_db_path: Optional[str] = None) -> str:
    """
    Generate documentation for a model implementation.
    
    Args:
        model_name: Name of the model
        hardware: Hardware platform
        skill_path: Path to skill implementation
        test_path: Path to test implementation
        benchmark_path: Path to benchmark implementation
        expected_results_path: Path to expected results file (optional)
        output_dir: Output directory for documentation (optional)
        template_db_path: Path to template database (optional)
        
    Returns:
        Path to the generated documentation file
    """
    generator = ModelDocGenerator(
        model_name=model_name,
        hardware=hardware,
        skill_path=skill_path,
        test_path=test_path,
        benchmark_path=benchmark_path,
        expected_results_path=expected_results_path,
        output_dir=output_dir,
        template_db_path=template_db_path
    )
    
    return generator.generate_documentation()


if __name__ == "__main__":
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(description="Generate model documentation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--hardware", required=True, help="Hardware platform")
    parser.add_argument("--skill-path", required=True, help="Path to skill implementation")
    parser.add_argument("--test-path", required=True, help="Path to test implementation")
    parser.add_argument("--benchmark-path", required=True, help="Path to benchmark implementation")
    parser.add_argument("--expected-results", help="Path to expected results file")
    parser.add_argument("--output-dir", help="Output directory for documentation")
    parser.add_argument("--template-db-path", help="Path to template database")
    parser.add_argument("--use-templates", action="store_true", help="Use templates for documentation generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Set environment variables for timestamp
    os.environ['DATE'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Determine template DB path
    template_db_path = None
    if args.use_templates:
        if args.template_db_path:
            template_db_path = args.template_db_path
        else:
            # Try default locations
            default_paths = [
                os.path.join(script_dir, "template_database.duckdb"),
                os.path.join(script_dir, "test_template_db.duckdb"),
                os.path.join(test_dir, "template_db.duckdb"),
                os.path.join(test_dir, "template_database.duckdb")
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    template_db_path = path
                    logger.info(f"Using template database at {path}")
                    break
                    
            if not template_db_path:
                logger.warning("No template database found. Using manual documentation generation.")
    
    doc_path = generate_model_documentation(
        model_name=args.model,
        hardware=args.hardware,
        skill_path=args.skill_path,
        test_path=args.test_path,
        benchmark_path=args.benchmark_path,
        expected_results_path=args.expected_results,
        output_dir=args.output_dir,
        template_db_path=template_db_path
    )
    
    print(f"Documentation generated: {doc_path}")