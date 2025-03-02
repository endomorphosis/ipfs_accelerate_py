#!/usr/bin/env python3
"""
Unified Test Generation Tool for IPFS Accelerate Python

This advanced script streamlines test generation across multiple components:
1. Model tests - For Hugging Face and other model implementations
2. API backend tests - For API clients like OpenAI, Claude, Groq, etc.
3. Hardware backend tests - For CPU, CUDA, and OpenVINO implementations
4. Integration tests - For testing complete workflows

Features:
- Intelligent detection of required test patterns
- Automatic code generation with best practices
- Validation against existing implementations
- Comprehensive logging and reporting
- Configurable test parameters
- Standardized testing methodology
"""

import os
import sys
import json
import glob
import time
import random
import datetime
import argparse
import logging
import traceback
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_generation.log")
    ]
)
logger = logging.getLogger("unified_test_generator")

# Constants for paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = CURRENT_DIR.parent
SKILLS_DIR = CURRENT_DIR / "skills"
APIS_DIR = CURRENT_DIR / "apis"
CACHE_DIR = CURRENT_DIR / ".test_generation_cache"

# Ensure required directories exist
for directory in [SKILLS_DIR, APIS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Task category definitions (reused from existing generators)
TASK_CATEGORIES = {
    "language": {
        "tasks": [
            "text-generation", "text2text-generation", "fill-mask", "text-classification",
            "token-classification", "question-answering", "summarization", "translation"
        ],
        "example_models": {
            "text-generation": "distilgpt2",
            "text2text-generation": "t5-small",
            "fill-mask": "distilroberta-base",
            "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
            "token-classification": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "question-answering": "distilbert-base-cased-distilled-squad",
            "summarization": "sshleifer/distilbart-cnn-6-6",
            "translation": "Helsinki-NLP/opus-mt-en-de"
        },
        "test_input_examples": {
            "basic": 'self.test_text = "The quick brown fox jumps over the lazy dog"',
            "batch": 'self.test_batch = ["The quick brown fox jumps over the lazy dog", "The five boxing wizards jump quickly"]',
            "qa": 'self.test_qa = {"question": "What is the capital of France?", "context": "Paris is the capital and most populous city of France."}',
            "summarization": 'self.test_summarization = "In a groundbreaking discovery, scientists have found a new species of frog in the Amazon rainforest."'
        }
    },
    "vision": {
        "tasks": [
            "image-classification", "object-detection", "image-segmentation", "depth-estimation",
            "semantic-segmentation", "instance-segmentation"
        ],
        "example_models": {
            "image-classification": "google/vit-base-patch16-224-in21k",
            "object-detection": "facebook/detr-resnet-50",
            "image-segmentation": "facebook/mask2former-swin-base-coco-instance",
            "depth-estimation": "Intel/dpt-large"
        },
        "test_input_examples": {
            "basic": 'self.test_image_path = "test.jpg"',
            "with_pil": 'try:\n    from PIL import Image\n    self.test_image = Image.open("test.jpg") if os.path.exists("test.jpg") else None\nexcept ImportError:\n    self.test_image = None',
            "batch": 'self.test_batch_images = ["test.jpg", "test.jpg"] if os.path.exists("test.jpg") else None'
        }
    },
    "audio": {
        "tasks": [
            "automatic-speech-recognition", "audio-classification", "text-to-audio",
            "audio-to-audio", "audio-xvector"
        ],
        "example_models": {
            "automatic-speech-recognition": "openai/whisper-tiny",
            "audio-classification": "superb/hubert-base-superb-ks",
            "text-to-audio": "facebook/musicgen-small"
        },
        "test_input_examples": {
            "basic": 'self.test_audio_path = "test.mp3"',
            "with_array": 'try:\n    import librosa\n    self.test_audio, self.test_sr = librosa.load("test.mp3", sr=16000) if os.path.exists("test.mp3") else (None, 16000)\nexcept ImportError:\n    self.test_audio, self.test_sr = None, 16000',
            "batch": 'self.test_batch_audio = ["test.mp3", "test.mp3"] if os.path.exists("test.mp3") else None'
        }
    },
    "multimodal": {
        "tasks": [
            "image-to-text", "visual-question-answering", "document-question-answering",
            "video-classification"
        ],
        "example_models": {
            "image-to-text": "Salesforce/blip-image-captioning-base",
            "visual-question-answering": "Salesforce/blip-vqa-base",
            "document-question-answering": "impira/layoutlm-document-qa",
            "video-classification": "MCG-NJU/videomae-base"
        },
        "test_input_examples": {
            "basic": 'self.test_image_path = "test.jpg"',
            "vqa": 'self.test_vqa = {"image": "test.jpg", "question": "What is shown in this image?"}',
            "document_qa": 'self.test_document_qa = {"image": "test.jpg", "question": "What is the title of this document?"}'
        }
    },
    "specialized": {
        "tasks": [
            "protein-folding", "table-question-answering", "time-series-prediction",
            "graph-classification", "reinforcement-learning"
        ],
        "example_models": {
            "protein-folding": "facebook/esm2_t6_8M_UR50D",
            "table-question-answering": "google/tapas-base-finetuned-wtq",
            "time-series-prediction": "huggingface/time-series-transformer-tourism-monthly"
        },
        "test_input_examples": {
            "protein": 'self.test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"',
            "time_series": 'self.test_time_series = {\n    "past_values": [100, 120, 140, 160, 180],\n    "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],\n    "future_time_features": [[5, 0], [6, 0], [7, 0]]\n}'
        }
    }
}

# Model-specific configurations
MODEL_CONFIGS = {
    "layoutlmv2": {
        "primary_task": "document-question-answering",
        "processor_type": "LayoutLMv2Processor",
        "model_type": "LayoutLMv2ForQuestionAnswering",
        "model_example": "microsoft/layoutlmv2-base-uncased",
        "special_imports": ["from transformers import LayoutLMv2Processor, LayoutLMv2ForQuestionAnswering"],
    },
    "patchtsmixer": {
        "primary_task": "time-series-prediction",
        "processor_type": "AutoProcessor",
        "model_type": "AutoModelForTimeSeriesPrediction",
        "model_example": "huggingface/time-series-prediction-patchtsmixer-tourism-monthly",
        "special_imports": ["from transformers import AutoProcessor, PatchTSMixerForPrediction"],
    }
}

# API backend templates
API_TEMPLATES = {
    "base": {
        "imports": """import os
import sys
import json
import time
import unittest
import threading
from unittest import mock
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
""",
        "client_setup": """
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
        # Use mock server by default
        self.client = {client_class}(
            api_key="test_key",
            base_url="http://mock-server"
        )
        
        # Optional: Configure with real credentials from environment variables
        api_key = os.environ.get("{env_key}")
        base_url = os.environ.get("{env_url}", None)
        
        if api_key:
            client_args = {{
                "api_key": api_key
            }}
            if base_url:
                client_args["base_url"] = base_url
                
            self.client = {client_class}(**client_args)
            self.using_real_client = True
        else:
            self.using_real_client = False
"""
    }
}

def normalize_model_name(name: str) -> str:
    """
    Normalize model name to match file naming conventions
    
    Args:
        name: Original model name
        
    Returns:
        Normalized model name
    """
    return name.replace('-', '_').replace('.', '_').lower()

def load_model_data() -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load all model data from JSON files
    
    Returns:
        Tuple containing:
        - List of all model types
        - Dict mapping model names to pipeline tasks
        - Dict mapping pipeline tasks to model names
    """
    try:
        # Try to load from files or fall back to empty values
        all_models = []
        model_to_pipeline = {}
        pipeline_to_model = {}
        
        # Try to load model types
        model_types_path = CURRENT_DIR / "huggingface_model_types.json"
        if model_types_path.exists():
            with open(model_types_path, 'r') as f:
                all_models = json.load(f)
        else:
            logger.warning(f"Model types file not found at {model_types_path}")
            
        # Try to load model-pipeline mapping
        model_pipeline_path = CURRENT_DIR / "huggingface_model_pipeline_map.json"
        if model_pipeline_path.exists():
            with open(model_pipeline_path, 'r') as f:
                model_to_pipeline = json.load(f)
        else:
            logger.warning(f"Model-pipeline map not found at {model_pipeline_path}")
            
        # Try to load pipeline-model mapping
        pipeline_model_path = CURRENT_DIR / "huggingface_pipeline_model_map.json"
        if pipeline_model_path.exists():
            with open(pipeline_model_path, 'r') as f:
                pipeline_to_model = json.load(f)
        else:
            logger.warning(f"Pipeline-model map not found at {pipeline_model_path}")
            
        logger.info(f"Loaded {len(all_models)} models and pipeline mappings")
        return all_models, model_to_pipeline, pipeline_to_model
    
    except Exception as e:
        logger.error(f"Error loading model data: {e}")
        return [], {}, {}

def get_existing_test_files(directory: Path, prefix: str = "test_", extension: str = ".py") -> Dict[str, Path]:
    """
    Get a mapping of existing test files with their actual paths
    
    Args:
        directory: Directory to search for test files
        prefix: Prefix for test files
        extension: File extension for test files
        
    Returns:
        Dict mapping normalized test names to their file paths
    """
    if not directory.exists():
        return {}
        
    test_files = directory.glob(f"{prefix}*{extension}")
    existing_tests = {}
    
    for test_file in test_files:
        test_name = test_file.name[len(prefix):-len(extension)]
        existing_tests[test_name] = test_file
    
    logger.info(f"Found {len(existing_tests)} existing test files with prefix '{prefix}' in {directory}")
    return existing_tests

def get_task_category(tasks: List[str]) -> str:
    """
    Determine the most appropriate model category based on tasks
    
    Args:
        tasks: List of tasks
        
    Returns:
        Category name: language, vision, audio, multimodal, or specialized
    """
    # Convert tasks to a set for faster lookups
    task_set = set(tasks)
    
    # Check each category for matching tasks
    for category, data in TASK_CATEGORIES.items():
        if task_set.intersection(set(data["tasks"])):
            return category
    
    # Default to language models if no match
    return "language"

def get_primary_task(model: str, tasks: List[str]) -> str:
    """
    Determine the primary task for a model
    
    Args:
        model: Model name
        tasks: List of tasks
        
    Returns:
        Primary task name
    """
    # If the model has a specific config, use that
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]["primary_task"]
    
    # For models with tasks, use the first task
    if tasks:
        return tasks[0]
    
    # Default task
    return "feature-extraction"

def get_model_info(
    model: str, 
    all_models: List[str], 
    model_to_pipeline: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Get comprehensive model information
    
    Args:
        model: Model name
        all_models: List of all models
        model_to_pipeline: Dict mapping models to pipeline tasks
        
    Returns:
        Dict with model information
    """
    normalized_name = normalize_model_name(model)
    pipeline_tasks = model_to_pipeline.get(model, [])
    
    # Determine primary task
    primary_task = get_primary_task(model, pipeline_tasks)
    
    # Get model category
    category = get_task_category(pipeline_tasks)
    
    # Get example model
    if model in MODEL_CONFIGS:
        example_model = MODEL_CONFIGS[model]["model_example"]
    else:
        # Get from category definitions
        example_model = TASK_CATEGORIES[category]["example_models"].get(
            primary_task,
            "bert-base-uncased"  # Default fallback
        )
    
    return {
        "model": model,
        "normalized_name": normalized_name,
        "pipeline_tasks": pipeline_tasks,
        "primary_task": primary_task,
        "category": category,
        "example_model": example_model
    }

class ModelTestGenerator:
    """
    Generator for model test files
    """
    def __init__(self, output_dir: Path = SKILLS_DIR):
        self.output_dir = output_dir
        self.all_models, self.model_to_pipeline, self.pipeline_to_model = load_model_data()
        self.existing_tests = get_existing_test_files(output_dir, prefix="test_hf_")
        
    def get_missing_tests(self, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of missing test implementations
        
        Args:
            category_filter: Optional category to filter by
            
        Returns:
            List of models missing test implementations
        """
        missing = []
        
        for model in self.all_models:
            normalized_name = normalize_model_name(model)
            
            # Skip if test already exists
            if normalized_name in self.existing_tests:
                continue
            
            # Get model info
            model_info = get_model_info(model, self.all_models, self.model_to_pipeline)
            
            # Apply category filter if provided
            if category_filter and model_info["category"] != category_filter:
                continue
            
            missing.append(model_info)
        
        # Sort by category and primary task
        missing.sort(key=lambda x: (x["category"], x["primary_task"]))
        
        return missing
    
    def get_test_inputs(self, model_info: Dict[str, Any]) -> List[str]:
        """
        Get test input examples for a model
        
        Args:
            model_info: Model information
            
        Returns:
            List of test input examples
        """
        category = model_info["category"]
        primary_task = model_info["primary_task"]
        
        # Start with examples from the category
        examples = []
        
        # Add basic example
        if "basic" in TASK_CATEGORIES[category]["test_input_examples"]:
            examples.append(TASK_CATEGORIES[category]["test_input_examples"]["basic"])
        
        # Add task-specific examples
        for key, example in TASK_CATEGORIES[category]["test_input_examples"].items():
            if key != "basic" and key in primary_task.lower():
                examples.append(example)
        
        # Add batch examples for better testing
        if "batch" in ''.join(TASK_CATEGORIES[category]["test_input_examples"].keys()):
            batch_keys = [k for k in TASK_CATEGORIES[category]["test_input_examples"].keys() if "batch" in k]
            for key in batch_keys:
                examples.append(TASK_CATEGORIES[category]["test_input_examples"][key])
        
        # If no examples found, add a default
        if not examples:
            examples.append('self.test_input = "Test input for model"')
        
        return examples
    
    def get_model_config(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get model-specific configuration
        
        Args:
            model_info: Model information
            
        Returns:
            Model configuration
        """
        model = model_info["model"]
        primary_task = model_info["primary_task"]
        
        # Check if model has a specific config
        if model in MODEL_CONFIGS:
            return MODEL_CONFIGS[model]
        
        # Set default config based on category and primary task
        config = {
            "primary_task": primary_task,
            "processor_type": "AutoProcessor",
            "model_type": "AutoModel",
            "special_imports": [],
        }
        
        # Adjust model type based on task
        if primary_task == "text-generation":
            config["processor_type"] = "AutoTokenizer"
            config["model_type"] = "AutoModelForCausalLM"
            config["special_imports"] = ["from transformers import AutoTokenizer, AutoModelForCausalLM"]
        elif primary_task == "text2text-generation":
            config["processor_type"] = "AutoTokenizer"
            config["model_type"] = "AutoModelForSeq2SeqLM"
            config["special_imports"] = ["from transformers import AutoTokenizer, AutoModelForSeq2SeqLM"]
        elif primary_task == "fill-mask":
            config["processor_type"] = "AutoTokenizer"
            config["model_type"] = "AutoModelForMaskedLM"
            config["special_imports"] = ["from transformers import AutoTokenizer, AutoModelForMaskedLM"]
        elif primary_task == "text-classification":
            config["processor_type"] = "AutoTokenizer"
            config["model_type"] = "AutoModelForSequenceClassification"
            config["special_imports"] = ["from transformers import AutoTokenizer, AutoModelForSequenceClassification"]
        elif primary_task == "token-classification":
            config["processor_type"] = "AutoTokenizer"
            config["model_type"] = "AutoModelForTokenClassification"
            config["special_imports"] = ["from transformers import AutoTokenizer, AutoModelForTokenClassification"]
        elif primary_task == "question-answering":
            config["processor_type"] = "AutoTokenizer"
            config["model_type"] = "AutoModelForQuestionAnswering"
            config["special_imports"] = ["from transformers import AutoTokenizer, AutoModelForQuestionAnswering"]
        elif primary_task == "image-classification":
            config["processor_type"] = "AutoImageProcessor"
            config["model_type"] = "AutoModelForImageClassification"
            config["special_imports"] = ["from transformers import AutoImageProcessor, AutoModelForImageClassification"]
        elif primary_task == "object-detection":
            config["processor_type"] = "AutoImageProcessor"
            config["model_type"] = "AutoModelForObjectDetection"
            config["special_imports"] = ["from transformers import AutoImageProcessor, AutoModelForObjectDetection"]
        elif primary_task == "image-segmentation":
            config["processor_type"] = "AutoImageProcessor"
            config["model_type"] = "AutoModelForImageSegmentation"
            config["special_imports"] = ["from transformers import AutoImageProcessor, AutoModelForImageSegmentation"]
        elif primary_task == "automatic-speech-recognition":
            config["processor_type"] = "AutoProcessor"
            config["model_type"] = "AutoModelForSpeechSeq2Seq"
            config["special_imports"] = ["from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq"]
        
        return config
    
    def generate_test_content(self, model_info: Dict[str, Any]) -> str:
        """
        Generate comprehensive test content for a model
        
        Args:
            model_info: Model information
            
        Returns:
            Test file content
        """
        model = model_info["model"]
        normalized_name = model_info["normalized_name"]
        pipeline_tasks = model_info["pipeline_tasks"]
        primary_task = model_info["primary_task"]
        category = model_info["category"]
        example_model = model_info["example_model"]
        
        # Get test inputs
        test_inputs = self.get_test_inputs(model_info)
        test_inputs_str = "\n        ".join(test_inputs)
        
        # Get model config
        model_config = self.get_model_config(model_info)
        processor_type = model_config["processor_type"]
        model_type = model_config["model_type"]
        special_imports = "\n".join(model_config["special_imports"])
        
        # Generate documentation text
        model_docs = f"This test validates the {model} model for {primary_task}."
        if pipeline_tasks:
            model_docs += f"\nThe model supports the following tasks: {', '.join(pipeline_tasks)}"
        
        timestamp = datetime.datetime.now().isoformat()
        
        # Generate test file content
        content = f"""#!/usr/bin/env python3
# Test file for: {model}
# Generated on: {timestamp}
# Model category: {category}
# Primary task: {primary_task}
# Tasks: {', '.join(pipeline_tasks)}

import os
import sys
import json
import time
import datetime
import traceback
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Track available packages
PACKAGE_STATUS = {{}}

# Try/except pattern for optional dependencies
try:
    import torch
    PACKAGE_STATUS["torch"] = True
except ImportError:
    torch = MagicMock()
    PACKAGE_STATUS["torch"] = False
    print("Warning: torch not available, using mock implementation")

try:
    import transformers
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
    {special_imports}
    PACKAGE_STATUS["transformers"] = True
except ImportError:
    transformers = MagicMock()
    PACKAGE_STATUS["transformers"] = False
    print("Warning: transformers not available, using mock implementation")

# Import task-specific dependencies
if "{category}" == "vision" or "{category}" == "multimodal":
    try:
        from PIL import Image
        PACKAGE_STATUS["PIL"] = True
    except ImportError:
        Image = MagicMock()
        PACKAGE_STATUS["PIL"] = False
        print("Warning: PIL not available, using mock implementation")

if "{category}" == "audio":
    try:
        import librosa
        PACKAGE_STATUS["librosa"] = True
    except ImportError:
        librosa = MagicMock()
        PACKAGE_STATUS["librosa"] = False
        print("Warning: librosa not available, using mock implementation")

# Create a memory tracker for monitoring resource usage
class MemoryTracker:
    def __init__(self):
        self.baseline = 0
        self.peak = 0
        self.current = 0
        
    def start(self):
        '''Start memory tracking'''
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            self.baseline = torch.cuda.memory_allocated()
        
    def update(self):
        '''Update memory stats'''
        if torch.cuda.is_available():
            self.current = torch.cuda.memory_allocated() - self.baseline
            self.peak = max(self.peak, torch.cuda.max_memory_allocated() - self.baseline)
        
    def get_stats(self):
        '''Get current memory statistics'''
        return {{
            "current_mb": self.current / (1024 * 1024),
            "peak_mb": self.peak / (1024 * 1024),
            "baseline_mb": self.baseline / (1024 * 1024)
        }}

# Try importing the actual model implementation
try:
    from ipfs_accelerate_py.worker.skillset.hf_{normalized_name} import hf_{normalized_name}
    HAS_IMPLEMENTATION = True
except ImportError:
    # Create a mock class if not available
    class hf_{normalized_name}:
        \"\"\"
        Mock implementation of the {model} model.
        {model_docs}
        \"\"\"
        def __init__(self, resources=None, metadata=None):
            self.resources = resources or {{}}
            self.metadata = metadata or {{}}
            self._model_name = "{example_model}"
            self._primary_task = "{primary_task}"
            
        def init_cpu(self, model_name=None, model_type="{primary_task}", device="cpu", **kwargs):
            \"\"\"Initialize model for CPU inference\"\"\"
            model_name = model_name or self._model_name
            return None, None, lambda x: {{"output": "Mock CPU output for " + model_name, 
                                       "implementation_type": "MOCK"}}, None, 1
            
        def init_cuda(self, model_name=None, model_type="{primary_task}", device_label="cuda:0", **kwargs):
            \"\"\"Initialize model for CUDA inference\"\"\"
            model_name = model_name or self._model_name
            return None, None, lambda x: {{"output": "Mock CUDA output for " + model_name, 
                                       "implementation_type": "MOCK"}}, None, 4
            
        def init_openvino(self, model_name=None, model_type="{primary_task}", device="CPU", **kwargs):
            \"\"\"Initialize model for OpenVINO inference\"\"\"
            model_name = model_name or self._model_name
            return None, None, lambda x: {{"output": "Mock OpenVINO output for " + model_name, 
                                       "implementation_type": "MOCK"}}, None, 2
    
    HAS_IMPLEMENTATION = False
    print(f"Warning: hf_{normalized_name} implementation not found, using mock implementation")

class test_hf_{normalized_name}:
    \"\"\"
    Test implementation for the {model} model.
    {model_docs}
    \"\"\"
    def __init__(self, resources=None, metadata=None):
        # Initialize resources
        self.resources = resources if resources else {{
            "torch": torch,
            "numpy": np,
            "transformers": transformers
        }}
        self.metadata = metadata if metadata else {{}}
        
        # Create model implementation
        self.model = hf_{normalized_name}(resources=self.resources, metadata=self.metadata)
        
        # Define appropriate model for testing
        self.model_name = "{example_model}"
        
        # Define test inputs appropriate for this model type
        {test_inputs_str}
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {{}}
        
        # Initialize memory tracker
        self.memory_tracker = MemoryTracker()
        
        # Track performance metrics
        self.performance_metrics = {{}}
        
    def get_test_input(self, batch=False):
        '''Get appropriate test input based on testing mode'''
        # For batch testing
        if batch:
            if hasattr(self, 'test_batch'):
                return self.test_batch
            elif hasattr(self, 'test_batch_images'):
                return self.test_batch_images
            elif hasattr(self, 'test_batch_audio'):
                return self.test_batch_audio
            elif hasattr(self, 'test_batch_qa'):
                return self.test_batch_qa
        
        # For single item testing
        if "{primary_task}" == "text-generation" and hasattr(self, 'test_text'):
            return self.test_text
        elif "{category}" == "vision" and hasattr(self, 'test_image_path'):
            return self.test_image_path
        elif "{primary_task}" == "question-answering" and hasattr(self, 'test_qa'):
            return self.test_qa
        elif "{category}" == "audio" and hasattr(self, 'test_audio_path'):
            return self.test_audio_path
        
        # Default fallback
        if hasattr(self, 'test_input'):
            return self.test_input
        return "Default test input for {normalized_name}"
    
    def create_real_handler(self, model, processor, device):
        # Create a real handler function for the model
        device_str = str(device)
        
        def handler(input_data):
            try:
                start_time = time.time()
                
                # Process input based on model type
                if "{processor_type}" == "AutoTokenizer":
                    if isinstance(input_data, str):
                        inputs = processor(input_data, return_tensors="pt").to(device)
                    elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
                        inputs = processor(input_data, padding=True, truncation=True, return_tensors="pt").to(device)
                    elif isinstance(input_data, dict) and "question" in input_data and "context" in input_data:
                        inputs = processor(question=input_data["question"], context=input_data["context"], return_tensors="pt").to(device)
                    else:
                        inputs = processor(input_data, return_tensors="pt").to(device)
                elif "{processor_type}" in ["AutoImageProcessor", "AutoFeatureExtractor"]:
                    if isinstance(input_data, str) and os.path.exists(input_data):
                        from PIL import Image
                        image = Image.open(input_data)
                        inputs = processor(images=image, return_tensors="pt").to(device)
                    else:
                        inputs = processor(input_data, return_tensors="pt").to(device)
                else:
                    inputs = processor(input_data, return_tensors="pt").to(device)
                
                # Start memory tracking
                self.memory_tracker.start()
                
                # Run model inference with no gradients
                with torch.no_grad():
                    if "{primary_task}" == "text-generation":
                        output = model.generate(**inputs)
                        result = processor.decode(output[0], skip_special_tokens=True)
                    else:
                        # Handle all other model types
                        output = model(**inputs)
                        result = output
                
                # Update memory tracking
                self.memory_tracker.update()
                
                # Calculate metrics
                inference_time = time.time() - start_time
                
                # Return structured output with metadata
                return {{
                    "output": result,
                    "implementation_type": "REAL",
                    "device": device_str,
                    "inference_time": inference_time,
                    "memory_usage": self.memory_tracker.get_stats(),
                    "timestamp": datetime.datetime.now().isoformat()
                }}
            except Exception as e:
                error_str = str(e)
                traceback.print_exc()
                return {{
                    "output": None,
                    "implementation_type": "ERROR",
                    "device": device_str,
                    "error": error_str,
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.datetime.now().isoformat()
                }}
        
        return handler
    
    def _run_platform_test(self, platform, init_method, device_arg):
        """
        Run tests for a specific hardware platform
        
        Args:
            platform: Platform name (cpu, cuda, openvino)
            init_method: Method to initialize the model
            device_arg: Device argument for initialization
            
        Returns:
            Dict: Test results for this platform
        """
        platform_results = {{}}
        
        try:
            print(f"Testing {normalized_name} on {{platform.upper()}}...")
            
            # Initialize the model for this platform
            start_time = time.time()
            endpoint, processor, handler, queue, batch_size = init_method(
                self.model_name, "{primary_task}", device_arg
            )
            init_time = time.time() - start_time
            
            # Record initialization status
            valid_init = endpoint is not None and processor is not None and handler is not None
            platform_results[f"{{platform}}_init"] = "Success (REAL)" if valid_init else f"Failed {{platform.upper()}} initialization"
            platform_results[f"{{platform}}_init_time"] = init_time
            
            if not valid_init:
                platform_results[f"{{platform}}_handler"] = f"Failed {{platform.upper()}} handler init"
                return platform_results
            
            # Get test input
            test_input = self.get_test_input()
            
            # Run inference
            start_time = time.time()
            output = handler(test_input)
            elapsed_time = time.time() - start_time
            
            # Determine if output is valid
            is_valid_output = output is not None
            
            # Determine implementation type
            if isinstance(output, dict) and "implementation_type" in output:
                impl_type = output["implementation_type"]
            else:
                impl_type = "REAL" if is_valid_output else "MOCK"
            
            platform_results[f"{{platform}}_handler"] = f"Success ({{impl_type}})" if is_valid_output else f"Failed {{platform.upper()}} handler"
            platform_results[f"{{platform}}_inference_time"] = elapsed_time
            
            # Record example
            self.examples.append({{
                "input": str(test_input),
                "output": {{
                    "output_type": str(type(output)),
                    "implementation_type": impl_type
                }},
                "timestamp": datetime.datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "platform": platform.upper()
            }})
            
            # Test batch processing if available
            try:
                batch_input = self.get_test_input(batch=True)
                if batch_input is not None:
                    # Run batch inference
                    batch_start = time.time()
                    batch_output = handler(batch_input)
                    batch_time = time.time() - batch_start
                    
                    # Verify batch output
                    is_valid_batch = batch_output is not None
                    
                    # Determine implementation type
                    if isinstance(batch_output, dict) and "implementation_type" in batch_output:
                        batch_impl_type = batch_output["implementation_type"]
                    else:
                        batch_impl_type = "REAL" if is_valid_batch else "MOCK"
                    
                    platform_results[f"{{platform}}_batch"] = f"Success ({{batch_impl_type}})" if is_valid_batch else f"Failed {{platform.upper()}} batch"
                    platform_results[f"{{platform}}_batch_time"] = batch_time
                    
                    # Record batch example
                    self.examples.append({{
                        "input": str(batch_input),
                        "output": {{
                            "output_type": str(type(batch_output)),
                            "implementation_type": batch_impl_type,
                            "is_batch": True
                        }},
                        "timestamp": datetime.datetime.now().isoformat(),
                        "elapsed_time": batch_time,
                        "platform": platform.upper()
                    }})
            except Exception as batch_e:
                platform_results[f"{{platform}}_batch_error"] = str(batch_e)
        except Exception as e:
            print(f"Error in {{platform.upper()}} tests: {{e}}")
            traceback.print_exc()
            platform_results[f"{{platform}}_error"] = str(e)
            self.status_messages[platform] = f"Failed: {{str(e)}}"
        
        return platform_results
    
    def test(self):
        """
        Run comprehensive tests for the model on all platforms
        
        Returns:
            Dict: Structured test results
        """
        results = {{}}
        
        # Test basic initialization
        results["init"] = "Success" if self.model is not None else "Failed initialization"
        results["has_implementation"] = "Yes" if HAS_IMPLEMENTATION else "No (using mock)"
        
        # CPU tests
        cpu_results = self._run_platform_test("cpu", self.model.init_cpu, "cpu")
        results.update(cpu_results)
        
        # CUDA tests if available
        if torch.cuda.is_available():
            cuda_results = self._run_platform_test("cuda", self.model.init_cuda, "cuda:0")
            results.update(cuda_results)
        else:
            results["cuda_tests"] = "CUDA not available"
            self.status_messages["cuda"] = "CUDA not available"
        
        # OpenVINO tests if available
        try:
            import openvino
            openvino_results = self._run_platform_test("openvino", self.model.init_openvino, "CPU")
            results.update(openvino_results)
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
            self.status_messages["openvino"] = "OpenVINO not installed"
        
        # Create structured results
        structured_results = {{
            "status": results,
            "examples": self.examples,
            "performance_metrics": self.performance_metrics,
            "metadata": {{
                "model_name": self.model_name,
                "model": "{model}",
                "normalized_name": "{normalized_name}",
                "primary_task": "{primary_task}",
                "pipeline_tasks": {json.dumps(pipeline_tasks)},
                "category": "{category}",
                "test_timestamp": datetime.datetime.now().isoformat(),
                "package_status": PACKAGE_STATUS,
                "has_implementation": HAS_IMPLEMENTATION
            }}
        }}
        
        return structured_results
    
    def __test__(self):
        """
        Run tests and handle results
        
        Returns:
            Dict: Test results
        """
        test_results = {{}}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {{
                "status": {{"test_error": str(e)}},
                "examples": [],
                "metadata": {{
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }}
            }}
        
        # Create directories if needed
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        # Ensure directories exist
        for directory in [expected_dir, collected_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, mode=0o755, exist_ok=True)
        
        # Save test results
        results_file = os.path.join(collected_dir, 'hf_{normalized_name}_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Saved test results to {{results_file}}")
        except Exception as e:
            print(f"Error saving results: {{e}}")
        
        # Create or compare with expected results
        expected_file = os.path.join(expected_dir, 'hf_{normalized_name}_test_results.json')
        if not os.path.exists(expected_file):
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Created new expected results file")
        
        return test_results

def extract_implementation_status(results):
    """Extract implementation status from test results"""
    status_dict = results.get("status", {{}})
    
    # Extract status for each platform
    cpu_status = "UNKNOWN"
    cuda_status = "UNKNOWN"
    openvino_status = "UNKNOWN"
    
    # Check CPU status
    if "cpu_handler" in status_dict:
        if "REAL" in status_dict["cpu_handler"]:
            cpu_status = "REAL"
        elif "MOCK" in status_dict["cpu_handler"]:
            cpu_status = "MOCK"
    
    # Check CUDA status
    if "cuda_handler" in status_dict:
        if "REAL" in status_dict["cuda_handler"]:
            cuda_status = "REAL"
        elif "MOCK" in status_dict["cuda_handler"]:
            cuda_status = "MOCK"
    elif "cuda_tests" in status_dict and status_dict["cuda_tests"] == "CUDA not available":
        cuda_status = "NOT AVAILABLE"
    
    # Check OpenVINO status
    if "openvino_handler" in status_dict:
        if "REAL" in status_dict["openvino_handler"]:
            openvino_status = "REAL"
        elif "MOCK" in status_dict["openvino_handler"]:
            openvino_status = "MOCK"
    elif "openvino_tests" in status_dict and status_dict["openvino_tests"] == "OpenVINO not installed":
        openvino_status = "NOT INSTALLED"
    
    return {{
        "cpu": cpu_status,
        "cuda": cuda_status, 
        "openvino": openvino_status
    }}

if __name__ == "__main__":
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='{model} model test')
        parser.add_argument('--platform', type=str, choices=['cpu', 'cuda', 'openvino', 'all'], 
                        default='all', help='Platform to test')
        parser.add_argument('--model', type=str, help='Override model name')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        args = parser.parse_args()
        
        # Create test instance
        print(f"Starting {normalized_name} test...")
        test_instance = test_hf_{normalized_name}()
        
        # Override model if specified
        if args.model:
            test_instance.model_name = args.model
            print(f"Using model: {{args.model}}")
        
        # Run tests
        results = test_instance.__test__()
        
        # Get implementation status
        status = extract_implementation_status(results)
        
        # Show test summary
        print(f"\\n{normalized_name.upper()} TEST RESULTS SUMMARY")
        print(f"MODEL: {{results.get('metadata', {{}}).get('model_name', 'Unknown')}}")
        print(f"IMPLEMENTATION: {{results.get('metadata', {{}}).get('has_implementation', 'Unknown')}}")
        print(f"CPU STATUS: {{status['cpu']}}")
        print(f"CUDA STATUS: {{status['cuda']}}")
        print(f"OPENVINO STATUS: {{status['openvino']}}")
        
    except KeyboardInterrupt:
        print("Test stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {{e}}")
        traceback.print_exc()
        sys.exit(1)
"""
        
        return content
    
    def generate_test_file(self, model_info: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Generate a test file for a model and save it
        
        Args:
            model_info: Model information
            
        Returns:
            Tuple of (success_flag, message)
        """
        model = model_info["model"]
        normalized_name = model_info["normalized_name"]
        output_path = self.output_dir / f"test_hf_{normalized_name}.py"
        
        try:
            # Check if file already exists
            if output_path.exists():
                return False, f"Test file already exists for {model}"
            
            # Generate test content
            content = self.generate_test_content(model_info)
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(content)
            
            # Make file executable
            os.chmod(output_path, 0o755)
            
            return True, f"Generated test file for {model} at {output_path}"
        except Exception as e:
            return False, f"Error generating test for {model}: {e}"

class ApiTestGenerator:
    """Generator for API backend test files"""
    
    def __init__(self, output_dir: Path = APIS_DIR):
        self.output_dir = output_dir
        self.existing_tests = get_existing_test_files(output_dir, prefix="test_")
        
    def get_missing_tests(self) -> List[str]:
        """
        Get list of API backends missing test implementations
        
        Returns:
            List of API names missing test implementations
        """
        all_apis = [
            "claude", "gemini", "groq", "hf_tei", "hf_tgi", 
            "llvm", "ollama", "opea", "openai_api", "ovms", "s3_kit", "vllm"
        ]
        
        missing = []
        for api in all_apis:
            if api not in self.existing_tests:
                missing.append(api)
        
        logger.info(f"Found {len(missing)} APIs missing test implementations: {', '.join(missing)}")
        return missing
    
    def generate_api_test_content(self, api_name: str) -> str:
        """
        Generate test content for an API backend
        
        Args:
            api_name: Name of the API backend
            
        Returns:
            Generated test file content
        """
        # Normalize the API name and class name
        class_name = ''.join(word.capitalize() for word in api_name.split('_'))
        if api_name == "openai_api":
            class_name = "OpenAI"
        elif api_name == "s3_kit":
            class_name = "S3Kit"
        elif api_name == "hf_tei":
            class_name = "HFTextEmbeddingInference"
        elif api_name == "hf_tgi":
            class_name = "HFTextGenerationInference"
        
        client_class = f"{class_name}Client"
        
        # Determine environment variables
        env_vars = {
            "claude": {"key": "ANTHROPIC_API_KEY", "url": "ANTHROPIC_API_URL"},
            "gemini": {"key": "GOOGLE_API_KEY", "url": "GEMINI_API_URL"},
            "groq": {"key": "GROQ_API_KEY", "url": "GROQ_API_URL"},
            "hf_tei": {"key": "HF_API_TOKEN", "url": "HF_TEI_ENDPOINT"},
            "hf_tgi": {"key": "HF_API_TOKEN", "url": "HF_TGI_ENDPOINT"},
            "llvm": {"key": "LLVM_API_KEY", "url": "LLVM_API_URL"},
            "ollama": {"key": "OLLAMA_API_KEY", "url": "OLLAMA_API_URL"},
            "opea": {"key": "OPEA_API_KEY", "url": "OPEA_API_URL"},
            "openai_api": {"key": "OPENAI_API_KEY", "url": "OPENAI_API_URL"},
            "ovms": {"key": "OVMS_API_KEY", "url": "OVMS_ENDPOINT"},
            "s3_kit": {"key": "AWS_ACCESS_KEY_ID", "url": "S3_ENDPOINT"},
            "vllm": {"key": "VLLM_API_KEY", "url": "VLLM_ENDPOINT"}
        }
        
        env_key = env_vars.get(api_name, {}).get("key", f"{api_name.upper()}_API_KEY")
        env_url = env_vars.get(api_name, {}).get("url", f"{api_name.upper()}_API_URL")
        
        # Additional imports needed for each API
        api_specific_imports = {
            "claude": "from ipfs_accelerate_py.api_backends.claude import ClaudeClient",
            "gemini": "from ipfs_accelerate_py.api_backends.gemini import GeminiClient",
            "groq": "from ipfs_accelerate_py.api_backends.groq import GroqClient",
            "hf_tei": "from ipfs_accelerate_py.api_backends.hf_tei import HFTextEmbeddingInferenceClient",
            "hf_tgi": "from ipfs_accelerate_py.api_backends.hf_tgi import HFTextGenerationInferenceClient",
            "llvm": "from ipfs_accelerate_py.api_backends.llvm import LlvmClient",
            "ollama": "from ipfs_accelerate_py.api_backends.ollama import OllamaClient",
            "opea": "from ipfs_accelerate_py.api_backends.opea import OpeaClient",
            "openai_api": "from ipfs_accelerate_py.api_backends.openai_api import OpenAIClient",
            "ovms": "from ipfs_accelerate_py.api_backends.ovms import OvmsClient", 
            "s3_kit": "from ipfs_accelerate_py.api_backends.s3_kit import S3KitClient",
            "vllm": "from ipfs_accelerate_py.api_backends.vllm import VllmClient"
        }
        
        # API-specific test methods
        test_methods = {
            "claude": """
    def test_chat_completion(self):
        \"\"\"Test generating chat completions\"\"\"
        if not self.using_real_client:
            self.skipTest("Skipping test with mock client")
        
        messages = [{"role": "user", "content": "Hello, Claude!"}]
        response = self.client.chat_completion(messages=messages, model="claude-3-sonnet-20240229")
        
        self.assertIsInstance(response, dict)
        self.assertIn("choices", response)
        self.assertGreaterEqual(len(response["choices"]), 1)
        self.assertIn("message", response["choices"][0])
        
    def test_streaming_chat_completion(self):
        \"\"\"Test streaming chat completions\"\"\"
        if not self.using_real_client:
            self.skipTest("Skipping test with mock client")
        
        messages = [{"role": "user", "content": "Count from 1 to 5 briefly"}]
        responses = []
        
        for chunk in self.client.chat_completion_stream(messages=messages, model="claude-3-haiku-20240307"):
            responses.append(chunk)
            # Limit number of chunks for testing
            if len(responses) >= 10:
                break
        
        self.assertGreaterEqual(len(responses), 1)
        for chunk in responses:
            self.assertIsInstance(chunk, dict)""",
            
            "openai_api": """
    def test_chat_completion(self):
        \"\"\"Test generating chat completions\"\"\"
        if not self.using_real_client:
            self.skipTest("Skipping test with mock client")
        
        messages = [{"role": "user", "content": "Hello, Assistant!"}]
        response = self.client.chat_completion(messages=messages, model="gpt-3.5-turbo")
        
        self.assertIsInstance(response, dict)
        self.assertIn("choices", response)
        self.assertGreaterEqual(len(response["choices"]), 1)
        self.assertIn("message", response["choices"][0])
        
    def test_streaming_chat_completion(self):
        \"\"\"Test streaming chat completions\"\"\"
        if not self.using_real_client:
            self.skipTest("Skipping test with mock client")
        
        messages = [{"role": "user", "content": "Count from 1 to 5 briefly"}]
        responses = []
        
        for chunk in self.client.chat_completion_stream(messages=messages, model="gpt-3.5-turbo"):
            responses.append(chunk)
            # Limit number of chunks for testing
            if len(responses) >= 10:
                break
        
        self.assertGreaterEqual(len(responses), 1)
        for chunk in responses:
            self.assertIsInstance(chunk, dict)
            
    def test_embeddings(self):
        \"\"\"Test generating embeddings\"\"\"
        if not self.using_real_client:
            self.skipTest("Skipping test with mock client")
        
        text = "This is a test sentence for embeddings."
        response = self.client.embeddings(input=[text], model="text-embedding-ada-002")
        
        self.assertIsInstance(response, dict)
        self.assertIn("data", response)
        self.assertGreaterEqual(len(response["data"]), 1)
        self.assertIn("embedding", response["data"][0])""",
        
            "ollama": """
    def test_generate(self):
        \"\"\"Test generating completions\"\"\"
        if not self.using_real_client:
            self.skipTest("Skipping test with mock client")
        
        prompt = "What is machine learning?"
        response = self.client.generate(prompt=prompt, model="llama3")
        
        self.assertIsInstance(response, dict)
        self.assertIn("response", response)
        
    def test_streaming_generate(self):
        \"\"\"Test streaming completions\"\"\"
        if not self.using_real_client:
            self.skipTest("Skipping test with mock client")
        
        prompt = "Count from 1 to 10 briefly"
        responses = []
        
        for chunk in self.client.generate_stream(prompt=prompt, model="llama3"):
            responses.append(chunk)
            # Limit number of chunks for testing
            if len(responses) >= 10:
                break
        
        self.assertGreaterEqual(len(responses), 1)
        
    def test_list_models(self):
        \"\"\"Test listing available models\"\"\"
        if not self.using_real_client:
            self.skipTest("Skipping test with mock client")
        
        response = self.client.list_models()
        
        self.assertIsInstance(response, dict)
        self.assertIn("models", response)
        self.assertIsInstance(response["models"], list)"""
        }
        
        # Create base template for API test
        base_template = """#!/usr/bin/env python3
\"\"\"
Test suite for {class_name} API implementation.

This module tests the {class_name} API backend functionality, including:
- Connection to server
- Request handling
- Response processing
- Error handling
- Queue and backoff systems
\"\"\"

import os
import sys
import unittest
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import client implementation
try:
    {api_specific_import}
except ImportError:
    try:
        from api_backends.{api_name} import {client_class}
    except ImportError:
        # Mock implementation for testing
        class {client_class}:
            def __init__(self, **kwargs):
                self.api_key = kwargs.get("api_key", "test_key")
                self.base_url = kwargs.get("base_url", "http://localhost:8000")
                self.request_count = 0
                self.max_retries = 3
                self.retry_delay = 1
                
            def set_api_key(self, api_key):
                self.api_key = api_key


class Test{class_name}ApiBackend(unittest.TestCase):
    \"\"\"Test cases for {class_name} API backend implementation.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
        # Use mock server by default
        self.client = {client_class}(
            api_key="test_key",
            base_url="http://mock-server"
        )
        
        # Optional: Configure with real credentials from environment variables
        api_key = os.environ.get("{env_key}")
        base_url = os.environ.get("{env_url}", None)
        
        if api_key:
            client_args = {{
                "api_key": api_key
            }}
            if base_url:
                client_args["base_url"] = base_url
                
            self.client = {client_class}(**client_args)
            self.using_real_client = True
        else:
            self.using_real_client = False
    
    def test_initialization(self):
        \"\"\"Test client initialization with API key.\"\"\"
        client = {client_class}(api_key="test_api_key")
        self.assertEqual(client.api_key, "test_api_key")
        
        # Test initialization without API key
        with mock.patch.dict(os.environ, {{"{env_key}": "env_api_key"}}):
            client = {client_class}()
            self.assertEqual(client.api_key, "env_api_key")
    
    def test_api_key_handling(self):
        \"\"\"Test API key handling in requests.\"\"\"
        # Test setting a new API key
        new_key = "new_test_key"
        self.client.set_api_key(new_key)
        self.assertEqual(self.client.api_key, new_key)
    
    def test_queue_system(self):
        \"\"\"Test the request queue system.\"\"\"
        if not hasattr(self.client, 'request_queue'):
            self.skipTest("Client doesn't have queue attribute")
            
        # Test queue size configuration
        self.client.max_concurrent_requests = 2
        
        # Simulate concurrent requests that take time
        def slow_request(i):
            return self.client._with_queue(
                lambda: (time.sleep(0.5), {{"result": f"Response {{i}}"}})
            )
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(slow_request, range(4)))
        end_time = time.time()
        
        # Verify results
        self.assertEqual(len(results), 4)
        
        # Check if it took enough time for queue processing
        # (4 requests with 2 concurrency and 0.5s sleep should take ~1.0s)
        if not self.using_real_client:
            self.assertGreaterEqual(end_time - start_time, 1.0)
    
    def test_retry_mechanism(self):
        \"\"\"Test retry mechanism for failed requests.\"\"\"
        if not hasattr(self.client, '_with_backoff'):
            self.skipTest("Client doesn't have backoff method")
            
        # Mock a server error
        fail_count = [0]
        
        def flaky_function():
            fail_count[0] += 1
            if fail_count[0] <= 2:  # Fail twice then succeed
                raise Exception("Simulated server error")
            return {{"success": True}}
        
        try:
            # This should succeed after retries
            result = self.client._with_backoff(flaky_function)
            self.assertIsInstance(result, dict)
            self.assertEqual(fail_count[0], 3)  # 2 failures + 1 success
        except Exception as e:
            if not self.using_real_client:
                self.fail(f"Retry mechanism failed: {{e}}")

{test_methods}


if __name__ == "__main__":
    unittest.main()
"""
        
        # Generate the test file content
        api_specific_import = api_specific_imports.get(api_name, f"from ipfs_accelerate_py.api_backends.{api_name} import {client_class}")
        specific_test_methods = test_methods.get(api_name, "")
        
        content = base_template.format(
            api_name=api_name,
            class_name=class_name,
            client_class=client_class,
            env_key=env_key,
            env_url=env_url,
            api_specific_import=api_specific_import,
            test_methods=specific_test_methods
        )
        
        return content
    
    def generate_api_test_file(self, api_name: str) -> Tuple[bool, str]:
        """
        Generate a test file for an API backend
        
        Args:
            api_name: Name of the API backend
            
        Returns:
            Tuple of (success_flag, message)
        """
        output_path = self.output_dir / f"test_{api_name}.py"
        
        try:
            # Check if file already exists
            if output_path.exists():
                return False, f"Test file already exists for {api_name}"
            
            # Generate test content
            content = self.generate_api_test_content(api_name)
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(content)
            
            # Make file executable
            os.chmod(output_path, 0o755)
            
            return True, f"Generated test file for {api_name} at {output_path}"
        except Exception as e:
            return False, f"Error generating test for {api_name}: {e}"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified test generator for IPFS Accelerate Python")
    
    # General options
    parser.add_argument(
        "--type", type=str, 
        choices=["model", "api", "all"],
        default="model",
        help="Type of tests to generate"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    
    # Model test options
    model_group = parser.add_argument_group("Model test options")
    model_group.add_argument(
        "--models", type=str, nargs="+",
        help="List of models to generate tests for"
    )
    model_group.add_argument(
        "--category", type=str, 
        choices=["language", "vision", "audio", "multimodal", "specialized", "all"],
        default="all",
        help="Category of models to process"
    )
    model_group.add_argument(
        "--limit", type=int, default=5,
        help="Maximum number of test files to generate"
    )
    model_group.add_argument(
        "--list-missing", action="store_true",
        help="Only list missing tests, don't generate files"
    )
    
    # API test options
    api_group = parser.add_argument_group("API test options")
    api_group.add_argument(
        "--apis", type=str, nargs="+",
        help="List of APIs to generate tests for"
    )
    api_group.add_argument(
        "--force", action="store_true", 
        help="Force generation of test files even if they already exist"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", type=str,
        help="Custom output directory for generated files"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Process based on test type
    if args.type in ["model", "all"]:
        # Handle model test generation
        model_output_dir = Path(args.output_dir) if args.output_dir else SKILLS_DIR
        model_generator = ModelTestGenerator(output_dir=model_output_dir)
        
        # Apply category filter
        category_filter = args.category if args.category != "all" else None
        
        # Handle specific models if provided
        if args.models:
            logger.info(f"Processing specific models: {', '.join(args.models)}")
            results = []
            
            for model in args.models:
                model_info = get_model_info(model, model_generator.all_models, model_generator.model_to_pipeline)
                success, message = model_generator.generate_test_file(model_info)
                results.append({"model": model, "success": success, "message": message})
            
            # Print results
            for result in results:
                logger.info(f"{result['model']}: {'✅' if result['success'] else '❌'} {result['message']}")
            
            # Print summary
            successful = sum(1 for r in results if r["success"])
            logger.info(f"Summary: Successfully generated {successful}/{len(results)} model test files")
        else:
            # Get missing tests
            missing_models = model_generator.get_missing_tests(category_filter)
            logger.info(f"Found {len(missing_models)} models missing test implementations")
            
            # Print summary by category
            if missing_models:
                categories = {}
                for model_info in missing_models:
                    category = model_info["category"]
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(model_info)
                
                logger.info("\nMissing tests by category:")
                for category, models in sorted(categories.items()):
                    logger.info(f"\n{category.upper()} ({len(models)} models):")
                    for i, model_info in enumerate(models[:5]):
                        model = model_info["model"]
                        tasks = ", ".join(model_info["pipeline_tasks"])
                        logger.info(f"  {i+1}. {model}: {tasks}")
                    if len(models) > 5:
                        logger.info(f"  ... and {len(models) - 5} more {category} models")
            
            # If list-only, stop here
            if args.list_missing:
                return
            
            # Generate tests with limit
            limit = min(args.limit, len(missing_models))
            if limit > 0:
                logger.info(f"\nGenerating {limit} model test files...")
                results = []
                
                # Process models up to the limit
                for i, model_info in enumerate(missing_models[:limit]):
                    success, message = model_generator.generate_test_file(model_info)
                    results.append({"model": model_info["model"], "success": success, "message": message})
                
                # Print results
                successful = sum(1 for r in results if r["success"])
                logger.info(f"Summary: Successfully generated {successful}/{len(results)} model test files")
                
                if successful > 0:
                    logger.info("\nGenerated model test files:")
                    for result in results:
                        if result["success"]:
                            logger.info(f"  ✅ {result['model']}")
    
    if args.type in ["api", "all"]:
        # Handle API test generation
        api_output_dir = Path(args.output_dir) if args.output_dir else APIS_DIR
        api_generator = ApiTestGenerator(output_dir=api_output_dir)
        
        # Handle specific APIs if provided
        if args.apis:
            logger.info(f"Processing specific APIs: {', '.join(args.apis)}")
            results = []
            
            for api in args.apis:
                success, message = api_generator.generate_api_test_file(api)
                results.append({"api": api, "success": success, "message": message})
            
            # Print results
            for result in results:
                logger.info(f"{result['api']}: {'✅' if result['success'] else '❌'} {result['message']}")
            
            # Print summary
            successful = sum(1 for r in results if r["success"])
            logger.info(f"Summary: Successfully generated {successful}/{len(results)} API test files")
        else:
            # Get missing tests
            missing_apis = api_generator.get_missing_tests()
            logger.info(f"Found {len(missing_apis)} APIs missing test implementations: {', '.join(missing_apis)}")
            
            # Generate tests for missing APIs
            if missing_apis:
                logger.info(f"\nGenerating {len(missing_apis)} API test files...")
                results = []
                
                for api in missing_apis:
                    success, message = api_generator.generate_api_test_file(api)
                    results.append({"api": api, "success": success, "message": message})
                
                # Print results
                successful = sum(1 for r in results if r["success"])
                logger.info(f"Summary: Successfully generated {successful}/{len(results)} API test files")
                
                if successful > 0:
                    logger.info("\nGenerated API test files:")
                    for result in results:
                        if result["success"]:
                            logger.info(f"  ✅ {result['api']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Operation canceled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)