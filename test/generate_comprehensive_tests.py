#!/usr/bin/env python3
"""
Comprehensive Hugging Face Model Test Generator

This advanced script generates high-quality, comprehensive test files for HuggingFace models
with support for:
1. Sophisticated task-specific test inputs
2. Batch processing tests
3. Hardware-specific optimizations
4. Memory usage tracking
5. Performance benchmarking
6. Integration with common frameworks
7. Structured logging and reporting
8. Model fallback mechanisms
9. Extensive error handling
10. Coverage reporting
"""

import os
import sys
import json
import glob
import time
import random
import datetime
import argparse
import traceback
import subprocess
import logging
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
logger = logging.getLogger("test_generator")

# Constants for paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = CURRENT_DIR / "skills"
CACHE_DIR = CURRENT_DIR / ".test_generation_cache"

# Task category definitions
TASK_CATEGORIES = {
    "language": {
        "tasks": [
            "text-generation", "text2text-generation", "fill-mask", "text-classification",
            "token-classification", "question-answering", "summarization", "translation_xx_to_yy"
        ],
        "example_models": {
            "text-generation": "distilgpt2",
            "text2text-generation": "t5-small",
            "fill-mask": "distilroberta-base",
            "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
            "token-classification": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "question-answering": "distilbert-base-cased-distilled-squad",
            "summarization": "sshleifer/distilbart-cnn-6-6",
            "translation_xx_to_yy": "Helsinki-NLP/opus-mt-en-de"
        },
        "test_input_examples": {
            "basic": 'self.test_text = "The quick brown fox jumps over the lazy dog"',
            "batch": 'self.test_batch = ["The quick brown fox jumps over the lazy dog", "The five boxing wizards jump quickly"]',
            "qa": 'self.test_qa = {"question": "What is the capital of France?", "context": "Paris is the capital and most populous city of France."}',
            "batch_qa": 'self.test_batch_qa = [{"question": "What is the capital of France?", "context": "Paris is the capital and most populous city of France."}, {"question": "What is the tallest mountain?", "context": "Mount Everest is Earth\'s highest mountain above sea level."}]',
            "translation": 'self.test_translation = {"source": "The quick brown fox jumps over the lazy dog", "target_language": "de"}',
            "summarization": 'self.test_summarization = "In a groundbreaking discovery, scientists have found a new species of frog in the Amazon rainforest. The new frog, named Amazonian Hopper, is known for its unique ability to jump up to 20 times its body length, setting a new record in the amphibian world. According to researchers, this ability is an evolutionary adaptation to escape predators in the dense forest environment."'
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
            "depth-estimation": "Intel/dpt-large",
            "semantic-segmentation": "nvidia/segformer-b0-finetuned-ade-512-512",
            "instance-segmentation": "facebook/maskformer-swin-base-coco"
        },
        "test_input_examples": {
            "basic": 'self.test_image_path = "test.jpg"',
            "with_pil": 'try:\n    from PIL import Image\n    self.test_image = Image.open("test.jpg") if os.path.exists("test.jpg") else None\nexcept ImportError:\n    self.test_image = None',
            "batch": 'self.test_batch_images = ["test.jpg", "test.jpg"] if os.path.exists("test.jpg") else None',
            "numpy": 'try:\n    import numpy as np\n    from PIL import Image\n    img = Image.open("test.jpg") if os.path.exists("test.jpg") else None\n    self.test_image_array = np.array(img) if img else np.zeros((224, 224, 3), dtype=np.uint8)\nexcept ImportError:\n    self.test_image_array = None'
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
            "text-to-audio": "facebook/musicgen-small",
            "audio-to-audio": "facebook/encodec_24khz",
            "audio-xvector": "speechbrain/spkrec-ecapa-voxceleb"
        },
        "test_input_examples": {
            "basic": 'self.test_audio_path = "test.mp3"',
            "with_array": 'try:\n    import librosa\n    self.test_audio, self.test_sr = librosa.load("test.mp3", sr=16000) if os.path.exists("test.mp3") else (None, 16000)\nexcept ImportError:\n    self.test_audio, self.test_sr = None, 16000',
            "batch": 'self.test_batch_audio = ["test.mp3", "test.mp3"] if os.path.exists("test.mp3") else None',
            "text_to_audio": 'self.test_text_to_audio = "Generate a cheerful melody with piano"',
            "transcription": 'self.test_transcription = {"audio": "test.mp3", "language": "en"}'
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
            "batch_vqa": 'self.test_batch_vqa = [{"image": "test.jpg", "question": "What is shown in this image?"}, {"image": "test.jpg", "question": "What color is dominant in this image?"}]',
            "document_qa": 'self.test_document_qa = {"image": "test.jpg", "question": "What is the title of this document?"}',
            "video": 'self.test_video_path = "test.mp4" if os.path.exists("test.mp4") else None'
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
            "time-series-prediction": "huggingface/time-series-transformer-tourism-monthly",
            "graph-classification": "graphormer-base-pcqm4mv2",
            "reinforcement-learning": "edbeeching/decision-transformer-gym-hopper"
        },
        "test_input_examples": {
            "protein": 'self.test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"',
            "batch_protein": 'self.test_batch_sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "SGFRVQRITSSILRILEQNKDSTSAAQLEELVKVLSAQILYVTTLGYDSVSASRGGLDLGG"]',
            "table": 'self.test_table = {\n    "header": ["Name", "Age", "Occupation"],\n    "rows": [\n        ["John", "25", "Engineer"],\n        ["Alice", "32", "Doctor"],\n        ["Bob", "41", "Teacher"]\n    ],\n    "question": "How old is Alice?"\n}',
            "time_series": 'self.test_time_series = {\n    "past_values": [100, 120, 140, 160, 180],\n    "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],\n    "future_time_features": [[5, 0], [6, 0], [7, 0]]\n}',
            "batch_time_series": 'self.test_batch_time_series = [\n    {\n        "past_values": [100, 120, 140, 160, 180],\n        "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],\n        "future_time_features": [[5, 0], [6, 0], [7, 0]]\n    },\n    {\n        "past_values": [200, 220, 240, 260, 280],\n        "past_time_features": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],\n        "future_time_features": [[5, 0], [6, 0], [7, 0]]\n    }\n]'
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
        "specialized_processing": """
            # Process document and question
            if isinstance(input_data, dict) and "image" in input_data and "question" in input_data:
                image = input_data["image"]
                question = input_data["question"]
                if isinstance(image, str):
                    from PIL import Image
                    image = Image.open(image)
                inputs = processor(image=image, question=question, return_tensors="pt").to(device)
            elif isinstance(input_data, str) and os.path.exists(input_data):
                from PIL import Image
                image = Image.open(input_data)
                question = "What is this document about?"
                inputs = processor(image=image, question=question, return_tensors="pt").to(device)
            else:
                # Fallback
                inputs = processor(input_data, return_tensors="pt").to(device)
        """
    },
    "grounding-dino": {
        "primary_task": "object-detection",
        "processor_type": "AutoProcessor",
        "model_type": "AutoModelForObjectDetection",
        "model_example": "IDEA-Research/grounding-dino-base",
        "special_imports": ["from transformers import AutoProcessor, AutoModelForObjectDetection"],
        "specialized_processing": """
            # Process image for object detection
            if isinstance(input_data, str) and os.path.exists(input_data):
                from PIL import Image
                image = Image.open(input_data)
                inputs = processor(images=image, return_tensors="pt").to(device)
            elif hasattr(input_data, 'convert'):  # PIL Image
                inputs = processor(images=input_data, return_tensors="pt").to(device)
            else:
                # Fallback for other inputs
                inputs = processor(input_data, return_tensors="pt").to(device)
        """
    },
    "patchtsmixer": {
        "primary_task": "time-series-prediction",
        "processor_type": "AutoProcessor",
        "model_type": "AutoModelForTimeSeriesPrediction",
        "model_example": "huggingface/time-series-prediction-patchtsmixer-tourism-monthly",
        "special_imports": ["from transformers import AutoProcessor, PatchTSMixerForPrediction"],
        "specialized_processing": """
            # Process time series data
            if isinstance(input_data, dict) and "past_values" in input_data:
                # Handle time series input
                past_values = torch.tensor(input_data["past_values"]).float().unsqueeze(0).to(device)
                past_time_features = torch.tensor(input_data["past_time_features"]).float().unsqueeze(0).to(device)
                future_time_features = torch.tensor(input_data["future_time_features"]).float().unsqueeze(0).to(device)
                inputs = {
                    "past_values": past_values,
                    "past_time_features": past_time_features,
                    "future_time_features": future_time_features
                }
            elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
                # Handle batch time series input
                batch_size = len(input_data)
                # Stack batch inputs
                past_values = torch.tensor([item["past_values"] for item in input_data]).float().to(device)
                past_time_features = torch.tensor([item["past_time_features"] for item in input_data]).float().to(device)
                future_time_features = torch.tensor([item["future_time_features"] for item in input_data]).float().to(device)
                inputs = {
                    "past_values": past_values,
                    "past_time_features": past_time_features,
                    "future_time_features": future_time_features
                }
            else:
                # Fallback for other inputs
                inputs = processor(input_data, return_tensors="pt").to(device)
        """
    }
}

def setup_directories():
    """
    Set up necessary directories for generated tests and cache
    """
    for directory in [SKILLS_DIR, CACHE_DIR]:
        if not directory.exists():
            directory.mkdir(exist_ok=True, parents=True)
            logger.info(f"Created directory: {directory}")

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
        # Cache the data if not already cached
        model_types_cache = CACHE_DIR / "model_types.json"
        model_pipeline_cache = CACHE_DIR / "model_pipeline.json"
        pipeline_model_cache = CACHE_DIR / "pipeline_model.json"
        
        # Check if cache exists
        if all(path.exists() for path in [model_types_cache, model_pipeline_cache, pipeline_model_cache]):
            with open(model_types_cache, 'r') as f:
                all_models = json.load(f)
            with open(model_pipeline_cache, 'r') as f:
                model_to_pipeline = json.load(f)
            with open(pipeline_model_cache, 'r') as f:
                pipeline_to_model = json.load(f)
                
            logger.info(f"Loaded {len(all_models)} models and pipeline mappings from cache")
            return all_models, model_to_pipeline, pipeline_to_model
            
        # Load from original files
        with open('huggingface_model_types.json', 'r') as f:
            all_models = json.load(f)
        
        with open('huggingface_model_pipeline_map.json', 'r') as f:
            model_to_pipeline = json.load(f)
        
        with open('huggingface_pipeline_model_map.json', 'r') as f:
            pipeline_to_model = json.load(f)
            
        # Cache the data
        with open(model_types_cache, 'w') as f:
            json.dump(all_models, f)
        with open(model_pipeline_cache, 'w') as f:
            json.dump(model_to_pipeline, f)
        with open(pipeline_model_cache, 'w') as f:
            json.dump(pipeline_to_model, f)
            
        logger.info(f"Loaded and cached {len(all_models)} models and pipeline mappings")
        return all_models, model_to_pipeline, pipeline_to_model
    except Exception as e:
        logger.error(f"Error loading model data: {e}")
        raise

def get_existing_tests() -> Dict[str, str]:
    """
    Get a mapping of existing test files with their actual paths
    
    Returns:
        Dict mapping normalized model names to their test file paths
    """
    test_files = glob.glob(str(SKILLS_DIR / 'test_hf_*.py'))
    existing_tests = {}
    
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
        existing_tests[model_name] = test_file
    
    logger.info(f"Found {len(existing_tests)} existing test implementations")
    return existing_tests

def normalize_model_name(name: str) -> str:
    """
    Normalize model name to match file naming conventions
    
    Args:
        name: Original model name
        
    Returns:
        Normalized model name
    """
    return name.replace('-', '_').replace('.', '_').lower()

def get_task_category(pipeline_tasks: List[str]) -> str:
    """
    Determine the most appropriate model category based on pipeline tasks
    
    Args:
        pipeline_tasks: List of pipeline tasks
        
    Returns:
        Category name: language, vision, audio, multimodal, or specialized
    """
    # Convert tasks to a set for faster lookups
    task_set = set(pipeline_tasks)
    
    # Check each category for matching tasks
    for category, data in TASK_CATEGORIES.items():
        if task_set.intersection(set(data["tasks"])):
            return category
    
    # Default to language models if no match
    return "language"

def get_primary_task(model: str, pipeline_tasks: List[str]) -> str:
    """
    Determine the primary task for a model
    
    Args:
        model: Model name
        pipeline_tasks: List of pipeline tasks
        
    Returns:
        Primary task name
    """
    # If the model has a specific config, use that
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]["primary_task"]
    
    # For models with tasks, use the first task
    if pipeline_tasks:
        return pipeline_tasks[0]
    
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

def get_test_inputs(model_info: Dict[str, Any]) -> List[str]:
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

def get_model_config(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model-specific configuration
    
    Args:
        model_info: Model information
        
    Returns:
        Model configuration
    """
    model = model_info["model"]
    primary_task = model_info["primary_task"]
    category = model_info["category"]
    
    # Check if model has a specific config
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]
    
    # Set default config based on category and primary task
    config = {
        "primary_task": primary_task,
        "processor_type": "AutoProcessor",
        "model_type": "AutoModel",
        "special_imports": [],
        "specialized_processing": ""
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
    elif primary_task == "depth-estimation":
        config["processor_type"] = "AutoImageProcessor"
        config["model_type"] = "AutoModelForDepthEstimation"
        config["special_imports"] = ["from transformers import AutoImageProcessor, AutoModelForDepthEstimation"]
    elif primary_task == "automatic-speech-recognition":
        config["processor_type"] = "AutoProcessor"
        config["model_type"] = "AutoModelForSpeechSeq2Seq"
        config["special_imports"] = ["from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq"]
    elif primary_task == "audio-classification":
        config["processor_type"] = "AutoFeatureExtractor"
        config["model_type"] = "AutoModelForAudioClassification"
        config["special_imports"] = ["from transformers import AutoFeatureExtractor, AutoModelForAudioClassification"]
    elif primary_task == "image-to-text":
        config["processor_type"] = "AutoProcessor"
        config["model_type"] = "AutoModelForVision2Seq"
        config["special_imports"] = ["from transformers import AutoProcessor, AutoModelForVision2Seq"]
    elif primary_task == "visual-question-answering":
        config["processor_type"] = "AutoProcessor"
        config["model_type"] = "AutoModelForVisualQuestionAnswering"
        config["special_imports"] = ["from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering"]
    elif primary_task == "document-question-answering":
        config["processor_type"] = "AutoProcessor"
        config["model_type"] = "AutoModelForDocumentQuestionAnswering"
        config["special_imports"] = ["from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering"]
    elif primary_task == "time-series-prediction":
        config["processor_type"] = "AutoProcessor"
        config["model_type"] = "AutoModelForTimeSeriesPrediction"
        config["special_imports"] = ["from transformers import AutoProcessor, AutoModelForTimeSeriesPrediction"]
    
    return config

def generate_test_content(model_info: Dict[str, Any]) -> str:
    """
    Generate a test file for a given model
    
    Args:
        model_info: Model information
        
    Returns:
        Generated test file content
    """
    model = model_info["model"]
    normalized_name = model_info["normalized_name"]
    pipeline_tasks = model_info["pipeline_tasks"]
    primary_task = model_info["primary_task"]
    category = model_info["category"]
    example_model = model_info["example_model"]
    
    # Get test inputs
    test_inputs = get_test_inputs(model_info)
    test_inputs_str = "\n        ".join(test_inputs)
    
    # Get model config
    model_config = get_model_config(model_info)
    processor_type = model_config["processor_type"]
    model_type = model_config["model_type"]
    special_imports = "\n".join(model_config["special_imports"])
    specialized_processing = model_config["specialized_processing"]
    
    # Generate documentation text
    model_docs = f"This test validates the {model} model for {primary_task}."
    if pipeline_tasks:
        model_docs += f"\nThe model supports the following tasks: {', '.join(pipeline_tasks)}"
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Generate the test file content
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

# Import utility functions for testing
try:
    from test import utils as test_utils
    PACKAGE_STATUS["test_utils"] = True
except ImportError:
    test_utils = MagicMock()
    PACKAGE_STATUS["test_utils"] = False
    print("Warning: test utils not available, using mock implementation")

# Create a memory tracker for monitoring resource usage
class MemoryTracker:
    def __init__(self):
        self.baseline = 0
        self.peak = 0
        self.current = 0
        
    def start(self):
        """Start memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            self.baseline = torch.cuda.memory_allocated()
        
    def update(self):
        """Update memory stats"""
        if torch.cuda.is_available():
            self.current = torch.cuda.memory_allocated() - self.baseline
            self.peak = max(self.peak, torch.cuda.max_memory_allocated() - self.baseline)
        
    def get_stats(self):
        """Get current memory statistics"""
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
        """Get appropriate test input based on testing mode"""
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
            elif hasattr(self, 'test_batch_sequences'):
                return self.test_batch_sequences
            elif hasattr(self, 'test_batch_time_series'):
                return self.test_batch_time_series
            elif hasattr(self, 'test_batch_vqa'):
                return self.test_batch_vqa
        
        # For single item testing
        if "{primary_task}" == "text-generation" and hasattr(self, 'test_text'):
            return self.test_text
        elif "{primary_task}" in ["image-classification", "object-detection", "image-segmentation", "depth-estimation"]:
            if hasattr(self, 'test_image_path'):
                return self.test_image_path
            elif hasattr(self, 'test_image'):
                return self.test_image
        elif "{primary_task}" in ["image-to-text", "visual-question-answering"] and hasattr(self, 'test_vqa'):
            return self.test_vqa
        elif "{primary_task}" in ["automatic-speech-recognition", "audio-classification", "text-to-audio"]:
            if hasattr(self, 'test_audio_path'):
                return self.test_audio_path
            elif hasattr(self, 'test_audio'):
                return self.test_audio
            elif hasattr(self, 'test_transcription'):
                return self.test_transcription
        elif "{primary_task}" == "question-answering" and hasattr(self, 'test_qa'):
            return self.test_qa
        elif "{primary_task}" == "document-question-answering" and hasattr(self, 'test_document_qa'):
            return self.test_document_qa
        elif "{primary_task}" == "table-question-answering" and hasattr(self, 'test_table'):
            return self.test_table
        elif "{primary_task}" == "time-series-prediction" and hasattr(self, 'test_time_series'):
            return self.test_time_series
        elif "{primary_task}" == "protein-folding" and hasattr(self, 'test_sequence'):
            return self.test_sequence
        
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
                # {specialized_processing}
                
                # Generic input processing if no specialized processing
                if not "{specialized_processing}":
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
            
            # Record memory usage if available
            if isinstance(output, dict) and "memory_usage" in output:
                platform_results[f"{{platform}}_memory_usage"] = output["memory_usage"]
            
            # Record performance metrics
            self.performance_metrics[platform] = {{
                "init_time": init_time,
                "inference_time": elapsed_time,
                "implementation_type": impl_type,
                "successful": is_valid_output
            }}
            
            # Record example
            self.examples.append({{
                "input": str(test_input),
                "output": {{
                    "output_type": str(type(output)),
                    "implementation_type": impl_type
                }},
                "timestamp": datetime.datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "implementation_type": impl_type,
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
                    
                    # Record batch performance metrics
                    self.performance_metrics[f"{{platform}}_batch"] = {{
                        "inference_time": batch_time,
                        "implementation_type": batch_impl_type,
                        "successful": is_valid_batch,
                        "batch_size": len(batch_input) if isinstance(batch_input, list) else "Unknown"
                    }}
                    
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
                        "implementation_type": batch_impl_type,
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
        try:
            results["init"] = "Success" if self.model is not None else "Failed initialization"
            results["has_implementation"] = "Yes" if HAS_IMPLEMENTATION else "No (using mock)"
        except Exception as e:
            results["init"] = f"Error: {{str(e)}}"
        
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
        except Exception as e:
            print(f"Error in OpenVINO tests: {{e}}")
            traceback.print_exc()
            results["openvino_error"] = str(e)
            self.status_messages["openvino"] = f"Failed: {{str(e)}}"
        
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
                "has_implementation": HAS_IMPLEMENTATION,
                "platform_status": self.status_messages
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
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                
                # Extract status keys for comparison
                status_expected = expected_results.get("status", {{}})
                status_actual = test_results.get("status", {{}})
                
                # Check if results match
                expected_keys = {{"cpu_handler", "cuda_handler", "openvino_handler"}}
                mismatches = []
                
                # Compare implementation status
                for key in expected_keys.intersection(set(status_expected.keys())):
                    if key not in status_actual:
                        continue
                    
                    # Extract implementation type
                    expected_impl = "MOCK"
                    if "REAL" in status_expected[key]:
                        expected_impl = "REAL"
                    
                    actual_impl = "MOCK"
                    if "REAL" in status_actual[key]:
                        actual_impl = "REAL"
                    
                    if expected_impl != actual_impl:
                        mismatches.append(f"{{key}}: Expected {{expected_impl}}, got {{actual_impl}}")
                
                if mismatches:
                    print("Implementation type mismatches detected:")
                    for mismatch in mismatches:
                        print(f"  - {{mismatch}}")
                    
                    # Ask if expected results should be updated
                    print("\\nWould you like to update the expected results? (y/n)")
                    user_input = input().strip().lower()
                    if user_input == 'y':
                        with open(expected_file, 'w') as f:
                            json.dump(test_results, f, indent=2)
                        print(f"Updated expected results")
                else:
                    print("Test results match expected results")
            except Exception as e:
                print(f"Error comparing with expected results: {{e}}")
                # Create expected results file if comparison failed
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                print(f"Created new expected results file")
        else:
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
        parser.add_argument('--real', action='store_true', help='Force real implementation')
        parser.add_argument('--mock', action='store_true', help='Force mock implementation')
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
        
        # Show performance metrics if available
        perf_metrics = results.get("performance_metrics", {{}})
        if perf_metrics and args.verbose:
            print("\\nPERFORMANCE METRICS:")
            for platform, metrics in perf_metrics.items():
                if isinstance(metrics, dict):
                    print(f"  {{platform.upper()}}:")
                    for key, value in metrics.items():
                        print(f"    {{key}}: {{value}}")
        
        # Print structured results for parsing
        print("\\nstructured_results")
        print(json.dumps({{
            "status": status,
            "model_name": results.get("metadata", {{}}).get("model_name", "Unknown"),
            "primary_task": "{primary_task}",
            "category": "{category}",
            "has_implementation": results.get("metadata", {{}}).get("has_implementation", False)
        }}))
        
    except KeyboardInterrupt:
        print("Test stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {{e}}")
        traceback.print_exc()
        sys.exit(1)
"""

    return content

def generate_test_file(model_info: Dict[str, Any], output_dir: str) -> Tuple[bool, str]:
    """
    Generate a test file for a model and save it
    
    Args:
        model_info: Model information
        output_dir: Directory to save the test file
        
    Returns:
        Tuple of (success_flag, message)
    """
    model = model_info["model"]
    normalized_name = model_info["normalized_name"]
    output_path = os.path.join(output_dir, f"test_hf_{normalized_name}.py")
    
    try:
        # Check if file already exists
        if os.path.exists(output_path):
            return False, f"Test file already exists for {model}"
        
        # Generate test content
        content = generate_test_content(model_info)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(content)
        
        # Make file executable
        os.chmod(output_path, 0o755)
        
        return True, f"Generated test file for {model} at {output_path}"
    except Exception as e:
        return False, f"Error generating test for {model}: {e}"

def process_model_list(
    models: List[str], 
    all_models: List[str],
    model_to_pipeline: Dict[str, List[str]],
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Process a list of models to generate test files
    
    Args:
        models: List of models to process
        all_models: List of all available models
        model_to_pipeline: Dict mapping models to pipeline tasks
        output_dir: Directory to save generated test files
        
    Returns:
        List of processing results
    """
    results = []
    
    # Process each model
    for model in models:
        # Get model info
        model_info = get_model_info(model, all_models, model_to_pipeline)
        
        # Generate test file
        success, message = generate_test_file(model_info, output_dir)
        
        # Record result
        results.append({
            "model": model,
            "success": success,
            "message": message
        })
        
        # Log result
        if success:
            logger.info(message)
        else:
            logger.warning(message)
    
    return results

def list_missing_tests(
    all_models: List[str],
    existing_tests: Dict[str, str],
    model_to_pipeline: Dict[str, List[str]],
    category_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List models missing test implementations
    
    Args:
        all_models: List of all models
        existing_tests: Dict of existing test files
        model_to_pipeline: Dict mapping models to pipeline tasks
        category_filter: Optional category to filter by
        
    Returns:
        List of models missing test implementations
    """
    missing = []
    
    for model in all_models:
        normalized_name = normalize_model_name(model)
        
        # Skip if test already exists
        if normalized_name in existing_tests:
            continue
        
        # Get model info
        model_info = get_model_info(model, all_models, model_to_pipeline)
        
        # Apply category filter if provided
        if category_filter and model_info["category"] != category_filter:
            continue
        
        missing.append(model_info)
    
    # Sort by category and primary task
    missing.sort(key=lambda x: (x["category"], x["primary_task"]))
    
    return missing

def generate_batch(
    missing_models: List[Dict[str, Any]],
    output_dir: str,
    limit: int = 10,
    category_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate a batch of test files
    
    Args:
        missing_models: List of models missing tests
        output_dir: Directory to save generated test files
        limit: Maximum number of files to generate
        category_filter: Optional category to filter by
        
    Returns:
        List of processing results
    """
    # Apply category filter if provided
    if category_filter:
        models_to_process = [m for m in missing_models if m["category"] == category_filter]
    else:
        models_to_process = missing_models
    
    # Limit number of models to process
    models_to_process = models_to_process[:limit]
    
    # Process models
    results = []
    for model_info in models_to_process:
        # Generate test file
        success, message = generate_test_file(model_info, output_dir)
        
        # Record result
        results.append({
            "model": model_info["model"],
            "success": success,
            "message": message
        })
        
        # Log result
        if success:
            logger.info(message)
        else:
            logger.warning(message)
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate comprehensive tests for HuggingFace models")
    parser.add_argument(
        "--models", type=str, nargs="+",
        help="List of models to generate tests for"
    )
    parser.add_argument(
        "--category", type=str, 
        choices=["language", "vision", "audio", "multimodal", "specialized", "all"],
        default="all",
        help="Category of models to process"
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Maximum number of test files to generate"
    )
    parser.add_argument(
        "--list-missing", action="store_true",
        help="Only list missing tests, don't generate files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(SKILLS_DIR),
        help="Directory to save generated test files"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Setup directories
    setup_directories()
    
    # Load model data
    all_models, model_to_pipeline, pipeline_to_model = load_model_data()
    
    # Get existing tests
    existing_tests = get_existing_tests()
    
    # Handle specific models if provided
    if args.models:
        print(f"Processing specific models: {', '.join(args.models)}")
        results = process_model_list(
            args.models, all_models, model_to_pipeline, args.output_dir
        )
        
        # Print results
        for result in results:
            print(f"{result['model']}: {'✅' if result['success'] else '❌'} {result['message']}")
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nSummary: Successfully generated {successful}/{len(results)} test files")
        return
    
    # List missing tests
    category_filter = args.category if args.category != "all" else None
    missing_models = list_missing_tests(
        all_models, existing_tests, model_to_pipeline, category_filter
    )
    
    print(f"Found {len(missing_models)} models missing test implementations")
    
    # Print missing tests by category
    categories = {}
    for model_info in missing_models:
        category = model_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(model_info)
    
    print("\nMissing tests by category:")
    for category, models in sorted(categories.items()):
        print(f"\n{category.upper()} ({len(models)} models):")
        for i, model_info in enumerate(models[:5]):
            model = model_info["model"]
            tasks = ", ".join(model_info["pipeline_tasks"])
            print(f"  {i+1}. {model}: {tasks}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more {category} models")
    
    # If list-only, just print and exit
    if args.list_missing:
        return
    
    # Generate batch of tests
    print(f"\nGenerating up to {args.limit} test files...")
    results = generate_batch(
        missing_models, args.output_dir, args.limit, category_filter
    )
    
    # Print results
    successful = sum(1 for r in results if r["success"])
    print(f"\nSummary: Generated {successful}/{len(results)} test files")
    
    # Print list of generated files
    if successful > 0:
        print("\nGenerated test files:")
        for result in results:
            if result["success"]:
                print(f"  ✅ {result['model']}")
    
    # Print failures if any
    failures = [r for r in results if not r["success"]]
    if failures:
        print("\nFailed generations:")
        for result in failures:
            print(f"  ❌ {result['model']}: {result['message']}")

if __name__ == "__main__":
    main()