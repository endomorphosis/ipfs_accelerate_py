#!/usr/bin/env python3
"""
Template composer for IPFS Accelerate Python.

This module provides functionality to compose hardware, architecture, and 
pipeline templates to generate complete model implementations.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from templates.base_hardware import BaseHardwareTemplate
from templates.base_architecture import BaseArchitectureTemplate
from templates.base_pipeline import BasePipelineTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemplateComposer:
    """
    Composer for generating model implementation files by combining
    hardware, architecture, and pipeline templates.
    """
    
    def __init__(self, 
                 hardware_templates: Dict[str, BaseHardwareTemplate],
                 architecture_templates: Dict[str, BaseArchitectureTemplate],
                 pipeline_templates: Dict[str, BasePipelineTemplate],
                 output_dir: str):
        """
        Initialize the template composer.
        
        Args:
            hardware_templates: Dictionary mapping hardware types to templates
            architecture_templates: Dictionary mapping architecture types to templates
            pipeline_templates: Dictionary mapping pipeline types to templates
            output_dir: Directory for generated files
        """
        self.hardware_templates = hardware_templates
        self.architecture_templates = architecture_templates
        self.pipeline_templates = pipeline_templates
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def select_hardware_template(self, hardware_type: str) -> BaseHardwareTemplate:
        """
        Select a hardware template based on type.
        
        Args:
            hardware_type: The hardware type
            
        Returns:
            The hardware template
        """
        if hardware_type in self.hardware_templates:
            return self.hardware_templates[hardware_type]
        else:
            logger.warning(f"Unknown hardware type: {hardware_type}, using CPU as fallback")
            return self.hardware_templates["cpu"]
    
    def select_architecture_template(self, arch_type: str) -> BaseArchitectureTemplate:
        """
        Select an architecture template based on type.
        
        Args:
            arch_type: The architecture type
            
        Returns:
            The architecture template
        """
        if arch_type in self.architecture_templates:
            return self.architecture_templates[arch_type]
        else:
            logger.warning(f"Unknown architecture type: {arch_type}, using encoder-only as fallback")
            return self.architecture_templates["encoder-only"]
    
    def select_pipeline_template(self, pipeline_type: str) -> BasePipelineTemplate:
        """
        Select a pipeline template based on type.
        
        Args:
            pipeline_type: The pipeline type
            
        Returns:
            The pipeline template
        """
        if pipeline_type in self.pipeline_templates:
            return self.pipeline_templates[pipeline_type]
        else:
            logger.warning(f"Unknown pipeline type: {pipeline_type}, using text as fallback")
            return self.pipeline_templates["text"]
    
    def select_templates_for_model(self, 
                                  model_name: str, 
                                  arch_type: str,
                                  hardware_types: List[str]) -> Tuple[BaseArchitectureTemplate, 
                                                                     List[BaseHardwareTemplate], 
                                                                     BasePipelineTemplate]:
        """
        Select appropriate templates for a model.
        
        Args:
            model_name: The model name
            arch_type: The architecture type
            hardware_types: List of hardware types to include
            
        Returns:
            Tuple of (architecture_template, list of hardware_templates, pipeline_template)
        """
        # Select architecture template
        arch_template = self.select_architecture_template(arch_type)
        
        # Determine appropriate pipeline type based on architecture
        if arch_type in ["encoder-only", "decoder-only", "encoder-decoder"]:
            pipeline_type = "text"
        elif arch_type in ["vision"]:
            pipeline_type = "image"
        elif arch_type in ["vision-encoder-text-decoder"]:
            pipeline_type = "vision-text"  # Use dedicated vision-text pipeline
        elif arch_type in ["speech"]:
            pipeline_type = "audio"  # Use dedicated audio pipeline
        elif arch_type in ["multimodal"]:
            pipeline_type = "multimodal"  # Use dedicated multimodal pipeline
        elif arch_type in ["diffusion", "vae", "sam"]:
            pipeline_type = "diffusion"  # Use dedicated diffusion pipeline
        elif arch_type in ["mixture-of-experts", "moe", "sparse"]:
            pipeline_type = "moe"  # Use dedicated MoE pipeline
        elif arch_type in ["state-space", "mamba", "rwkv", "linear-attention", "recurrent"]:
            pipeline_type = "state-space"  # Use dedicated State-Space pipeline
        elif arch_type in ["rag", "retrieval-augmented-generation", "retrieval-augmented"]:
            pipeline_type = "rag"  # Use dedicated RAG pipeline
        else:
            pipeline_type = "text"  # Default to text
        
        # Select pipeline template
        pipeline_template = self.select_pipeline_template(pipeline_type)
        
        # Select hardware templates and filter for compatibility
        hardware_templates = []
        for hw_type in hardware_types:
            hw_template = self.select_hardware_template(hw_type)
            if hw_template.is_compatible_with_architecture(arch_type):
                hardware_templates.append(hw_template)
            else:
                logger.warning(f"Hardware {hw_type} is not compatible with architecture {arch_type}")
                fallback_hw = hw_template.get_fallback_hardware()
                fallback_template = self.select_hardware_template(fallback_hw)
                hardware_templates.append(fallback_template)
                
        return arch_template, hardware_templates, pipeline_template
    
    def generate_model_implementation(self,
                                     model_name: str,
                                     arch_type: str,
                                     hardware_types: List[str],
                                     force: bool = False) -> Tuple[bool, str]:
        """
        Generate a model implementation file.
        
        Args:
            model_name: The model name (for reference and documentation)
            arch_type: The architecture type (used for the filename)
            hardware_types: List of hardware types to include
            force: Whether to overwrite existing files
            
        Returns:
            Tuple of (success, output file path)
        """
        # Generate output file path using architecture type rather than model name
        # Architecture type better represents what the file implements
        arch_type_filename = arch_type.replace('-', '_')
        output_file = os.path.join(self.output_dir, f"hf_{arch_type_filename}.py")
            
        logger.info(f"Output file will be: {output_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(output_file) and not force:
            logger.warning(f"File already exists: {output_file}, use force=True to overwrite")
            return False, output_file
        
        # Select templates
        arch_template, hardware_templates, pipeline_template = self.select_templates_for_model(
            model_name, arch_type, hardware_types
        )
        
        # Get task type from architecture template
        task_type = arch_template.default_task_type
        
        # Begin generating content
        logger.info(f"Generating implementation for {model_name} ({arch_type}) on {', '.join(hardware_types)}")
        
        # Generate imports
        imports = self._generate_imports(arch_template, hardware_templates, pipeline_template)
        
        # Generate class definition
        class_def = self._generate_class_definition(model_name, arch_template)
        
        # Generate init method
        init_method = self._generate_init_method(model_name, arch_template)
        
        # Generate hardware initialization methods
        hardware_init_methods = self._generate_hardware_init_methods(
            model_name, arch_template, hardware_templates, pipeline_template, task_type
        )
        
        # Generate handler methods
        handler_methods = self._generate_handler_methods(
            model_name, arch_template, hardware_templates, pipeline_template, task_type
        )
        
        # Generate utilities
        utilities = self._generate_utilities(arch_template, pipeline_template)
        
        # Generate mock implementations
        mock_impls = self._generate_mock_implementations(
            model_name, arch_template, hardware_templates, pipeline_template, task_type
        )
        
        # Combine all sections
        content = f"{imports}\n\n{class_def}\n\n{init_method}\n\n{utilities}\n\n{mock_impls}\n\n{hardware_init_methods}\n\n{handler_methods}"
        
        # Write to file
        try:
            with open(output_file, 'w') as f:
                f.write(content)
            logger.info(f"Successfully wrote implementation to {output_file}")
            return True, output_file
        except Exception as e:
            logger.error(f"Error writing to {output_file}: {e}")
            return False, output_file
    
    def _generate_imports(self, 
                         arch_template: BaseArchitectureTemplate,
                         hardware_templates: List[BaseHardwareTemplate],
                         pipeline_template: BasePipelineTemplate) -> str:
        """Generate import statements."""
        imports = "#!/usr/bin/env python3\n"
        imports += "import asyncio\n"
        imports += "import os\n"
        imports += "import json\n"
        imports += "import time\n"
        imports += "from typing import Dict, List, Any, Tuple, Optional, Union\n\n"
        
        # Add architecture-specific imports
        
        # Add hardware-specific imports
        for hw_template in hardware_templates:
            imports += f"# {hw_template.hardware_name} imports\n"
            imports += hw_template.get_import_statements() + "\n"
        
        # Add pipeline-specific imports
        imports += f"# {pipeline_template.pipeline_type} pipeline imports\n"
        imports += pipeline_template.get_import_statements() + "\n"
        
        return imports
    
    def _generate_class_definition(self, 
                                  model_name: str,
                                  arch_template: BaseArchitectureTemplate) -> str:
        """Generate class definition."""
        # Use architecture type for class name, not model name
        arch_type = arch_template.architecture_type
        class_name = arch_type.replace('-', '_')
        return f"""class hf_{class_name}:
    \"\"\"HuggingFace {arch_template.architecture_name} implementation for {model_name.upper()}.
    
    This class provides standardized interfaces for working with {arch_template.architecture_name} models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    
    {arch_template.model_description}
    \"\"\"
"""
    
    def _generate_init_method(self, 
                             model_name: str,
                             arch_template: BaseArchitectureTemplate) -> str:
        """Generate initialization method."""
        return f"""    def __init__(self, resources=None, metadata=None):
        \"\"\"Initialize the {arch_template.architecture_name} model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        \"\"\"
        self.resources = resources
        self.metadata = metadata
        
        # Handler creation methods
        {self._generate_handler_refs(arch_template)}
        
        # Initialization methods
        self.init = self.init
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        
        # Test methods
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None  # Qualcomm SNPE utils
        return None
        
    def init(self):        
        if "torch" not in list(self.resources.keys()):
            import torch
            self.torch = torch
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.transformers = transformers
        else:
            self.transformers = self.resources["transformers"]
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]

        return None"""
    
    def _generate_handler_refs(self, arch_template: BaseArchitectureTemplate) -> str:
        """Generate handler references for all task types."""
        handler_refs = ""
        for task_type in arch_template.supported_task_types:
            handler_refs += f"self.create_cpu_{task_type}_endpoint_handler = self.create_cpu_{task_type}_endpoint_handler\n        "
            handler_refs += f"self.create_cuda_{task_type}_endpoint_handler = self.create_cuda_{task_type}_endpoint_handler\n        "
            handler_refs += f"self.create_openvino_{task_type}_endpoint_handler = self.create_openvino_{task_type}_endpoint_handler\n        "
            handler_refs += f"self.create_apple_{task_type}_endpoint_handler = self.create_apple_{task_type}_endpoint_handler\n        "
            handler_refs += f"self.create_qualcomm_{task_type}_endpoint_handler = self.create_qualcomm_{task_type}_endpoint_handler\n        "
        return handler_refs
    
    def _generate_hardware_init_methods(self,
                                       model_name: str,
                                       arch_template: BaseArchitectureTemplate,
                                       hardware_templates: List[BaseHardwareTemplate],
                                       pipeline_template: BasePipelineTemplate,
                                       task_type: str) -> str:
        """Generate hardware initialization methods."""
        init_methods = f"""    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        \"\"\"Test function to validate endpoint functionality.
        
        Args:
            endpoint_model: The model name or path
            endpoint_handler: The handler function
            endpoint_label: The hardware label
            tokenizer: The tokenizer
            
        Returns:
            Boolean indicating test success
        \"\"\"
        test_input = "{arch_template.test_input}"
        timestamp1 = time.time()
        test_batch = None
        
        # Get tokens for length calculation
        tokens = tokenizer(test_input)["input_ids"]
        len_tokens = len(tokens)
        
        try:
            # Run the model
            test_batch = endpoint_handler(test_input)
            print(test_batch)
            print("hf_{model_name} test passed")
        except Exception as e:
            print(e)
            print("hf_{model_name} test failed")
            return False
            
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {{elapsed_time}}")
        print(f"tokens: {{len_tokens}}")
        print(f"tokens per second: {{tokens_per_second}}")
        
        # Clean up memory
        with self.torch.no_grad():
            if "cuda" in dir(self.torch):
                self.torch.cuda.empty_cache()
        return True\n\n"""
        
        # Generate init methods for each hardware type
        for hw_template in hardware_templates:
            hw_type = hw_template.hardware_type
            hw_name = hw_template.hardware_name
            model_class = arch_template.get_model_class(task_type)
            
            init_methods += f"""    def init_{hw_type}(self, model_name, device, {hw_type}_label):
        \"\"\"Initialize {model_name.upper()} model for {hw_name} inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('{hw_type}')
            {hw_type}_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        \"\"\"
        self.init()
        
        {hw_template.get_hardware_detection_code()}
        
        # Check if hardware is available
        if not is_available():
            print(f"{hw_name} not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", {hw_type}_label.replace("{hw_type}", "cpu"))
        
        print(f"Loading {{model_name}} for {hw_name} inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Load model
            {hw_template.get_hardware_init_code(model_class, task_type)}
            
            # Create handler function
            handler = self.create_{hw_type}_{task_type}_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label={hw_type}_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            # Test the endpoint
            self.__test__(model_name, handler, {hw_type}_label, tokenizer)
            
            return model, tokenizer, handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing {hw_name} endpoint: {{e}}")
            print("Creating mock implementation instead")
            return self._create_mock_endpoint(model_name, {hw_type}_label)
        \n\n"""
        
        return init_methods
    
    def _generate_handler_methods(self,
                                 model_name: str,
                                 arch_template: BaseArchitectureTemplate,
                                 hardware_templates: List[BaseHardwareTemplate],
                                 pipeline_template: BasePipelineTemplate,
                                 task_type: str) -> str:
        """Generate handler methods."""
        handler_methods = ""
        
        # Generate handler methods for each hardware type and task type
        for hw_template in hardware_templates:
            hw_type = hw_template.hardware_type
            hw_name = hw_template.hardware_name
            
            handler_methods += f"""    def create_{hw_type}_{task_type}_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        \"\"\"Create handler function for {hw_name} {task_type} endpoint.
        
        Args:
            endpoint_model (str): The model name
            device (str): The device type ('{hw_type}')
            hardware_label (str): The hardware label
            endpoint: The loaded model
            tokenizer: The loaded tokenizer
            
        Returns:
            Handler function for this endpoint
        \"\"\"
        # Create closure that encapsulates the model and tokenizer
        def handler(text, *args, **kwargs):
            try:
                {pipeline_template.get_preprocessing_code(task_type)}
                
                # Run inference
                with self.torch.no_grad():
                    {hw_template.get_inference_code(task_type)}
                    {pipeline_template.get_postprocessing_code(task_type)}
                
                {pipeline_template.get_result_formatting_code(task_type)}
                
            except Exception as e:
                print(f"Error in {hw_name} handler: {{e}}")
                return {{"success": False, "error": str(e)}}
        
        return handler\n\n"""
        
        return handler_methods
    
    def _generate_utilities(self,
                           arch_template: BaseArchitectureTemplate,
                           pipeline_template: BasePipelineTemplate) -> str:
        """Generate utility methods."""
        return f"""    # Architecture utilities
{arch_template.get_model_config("model_name")}

    # Pipeline utilities
{pipeline_template.get_pipeline_utilities()}"""
    
    def _generate_mock_implementations(self,
                                      model_name: str,
                                      arch_template: BaseArchitectureTemplate,
                                      hardware_templates: List[BaseHardwareTemplate],
                                      pipeline_template: BasePipelineTemplate,
                                      task_type: str) -> str:
        """Generate mock implementation methods."""
        mock_impl = f"""    def _create_mock_processor(self):
        \"\"\"Create a mock tokenizer for graceful degradation when the real one fails.
        
        Returns:
            Mock tokenizer object with essential methods
        \"\"\"
        try:
            from unittest.mock import MagicMock
            
            tokenizer = MagicMock()
            
            # Configure mock tokenizer call behavior
            {arch_template.get_mock_processor_code()}
                
            tokenizer.side_effect = mock_tokenize
            tokenizer.__call__ = mock_tokenize
            
            print("(MOCK) Created mock {model_name.upper()} tokenizer")
            return tokenizer
            
        except ImportError:
            # Fallback if unittest.mock is not available
            class SimpleTokenizer:
                def __init__(self, parent):
                    self.parent = parent
                    
                def __call__(self, text, return_tensors="pt", padding=None, truncation=None, max_length=None):
                    {arch_template.get_mock_processor_code()}
            
            print("(MOCK) Created simple mock {model_name.upper()} tokenizer")
            return SimpleTokenizer(self)
    
    def _create_mock_endpoint(self, model_name, device_label):
        \"\"\"Create mock endpoint objects when real initialization fails.
        
        Args:
            model_name (str): The model name or path
            device_label (str): The device label (cpu, cuda, etc.)
            
        Returns:
            Tuple of (endpoint, tokenizer, handler, queue, batch_size)
        \"\"\"
        try:
            from unittest.mock import MagicMock
            
            # Create mock endpoint
            endpoint = MagicMock()
            
            # Configure mock endpoint behavior
            def mock_forward(**kwargs):
                batch_size = kwargs.get("input_ids", kwargs.get("inputs_embeds", None)).shape[0]
                sequence_length = kwargs.get("input_ids", kwargs.get("inputs_embeds", None)).shape[1]
                hidden_size = {arch_template.hidden_size}  # Architecture-specific hidden size
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create mock output structure
                {arch_template.get_mock_output_code()}
                
            endpoint.side_effect = mock_forward
            endpoint.__call__ = mock_forward
            
            # Create mock tokenizer
            tokenizer = self._create_mock_processor()
            
            # Create appropriate handler for the device type
            hardware_type = device_label.split(':')[0] if ':' in device_label else device_label
            
            if hardware_type.startswith('cpu'):
                handler_method = self.create_cpu_{task_type}_endpoint_handler
            elif hardware_type.startswith('cuda'):
                handler_method = self.create_cuda_{task_type}_endpoint_handler
            elif hardware_type.startswith('openvino'):
                handler_method = self.create_openvino_{task_type}_endpoint_handler
            elif hardware_type.startswith('apple'):
                handler_method = self.create_apple_{task_type}_endpoint_handler
            elif hardware_type.startswith('qualcomm'):
                handler_method = self.create_qualcomm_{task_type}_endpoint_handler
            else:
                handler_method = self.create_cpu_{task_type}_endpoint_handler
            
            # Create handler function
            mock_handler = handler_method(
                endpoint_model=model_name,
                device=hardware_type,
                hardware_label=device_label,
                endpoint=endpoint,
                tokenizer=tokenizer
            )
            
            import asyncio
            print(f"(MOCK) Created mock {model_name.upper()} endpoint for {{model_name}} on {{device_label}}")
            return endpoint, tokenizer, mock_handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error creating mock endpoint: {{e}}")
            import asyncio
            return None, None, None, asyncio.Queue(32), 0"""
        return mock_impl