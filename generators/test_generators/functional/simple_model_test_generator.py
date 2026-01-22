#!/usr/bin/env python3
"""
Enhanced Hugging Face model test generator with comprehensive coverage features:
- Supports both pipeline() and from_pretrained() testing methods
- Advanced dependency detection and tracking
- Remote code support with automatic detection
- Comprehensive mock objects for missing dependencies
- Multi-model batch processing
- Performance benchmarking support
- Test case generation based on model capabilities
- Dynamic inputs based on model type
- Hardware compatibility detection (CPU/CUDA/OpenVINO)
"""

import os
import sys
import json
import time
import collections
import importlib.util
import requests
from pathlib import Path
import logging
import subprocess
import concurrent.futures
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# Import template validator integration
try:
    from validators.template_validator_integration import validate_template_for_generator
    HAS_VALIDATOR = True
except ImportError:
    try:
        # Try relative import
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from generators.validators.template_validator_integration import validate_template_for_generator
        HAS_VALIDATOR = True
    except ImportError:
        # Define minimal validation
        def validate_template_for_generator(template_content, generator_type):
            return True, []
        HAS_VALIDATOR = False
        logging.warning("Template validator not found. Templates will not be validated.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = CURRENT_DIR / "skills"
TEST_FILE_PREFIX = "test_hf_"
MODEL_DEPENDENCIES_JSON = CURRENT_DIR / "model_dependencies.json"
MODEL_DEPENDENCIES_MD = CURRENT_DIR / "MODEL_DEPENDENCIES.md"
MODEL_TESTING_PROGRESS_MD = CURRENT_DIR / "MODEL_TESTING_PROGRESS.md"
TEST_SUMMARY_JSON = CURRENT_DIR / "test_coverage_summary.json"

# Map of common external dependencies for certain model types
KNOWN_DEPENDENCIES = {
    # LLM families
    "llama": ["sentencepiece", "tokenizers>=0.13.3", "accelerate>=0.20.3"],
    "falcon": ["einops", "accelerate>=0.16.0", "safetensors>=0.3.1"],
    "mistral": ["einops", "accelerate>=0.18.0", "safetensors>=0.3.2"],
    "mixtral": ["einops", "accelerate>=0.20.0", "safetensors>=0.3.2"],
    "mamba": ["causal-conv1d>=1.0.0", "triton>=2.0.0", "einops>=0.6.1"],
    "phi": ["einops", "accelerate>=0.20.0", "safetensors>=0.3.2"],
    "qwen": ["tiktoken", "einops", "optimum>=1.12.0"],
    "gemma": ["sentencepiece", "accelerate>=0.21.0", "safetensors>=0.3.2"],
    "codellama": ["sentencepiece", "tokenizers>=0.13.3", "accelerate>=0.20.3"],
    "deepseek": ["einops", "accelerate>=0.21.0", "flash-attn>=2.3.0"],
    
    # Audio models
    "whisper": ["jiwer", "librosa", "evaluate", "numpy>=1.20.0"],
    "wavlm": ["soundfile", "librosa"],
    "wav2vec2": ["librosa", "soundfile", "jiwer", "datasets>=2.14.0"],
    "hubert": ["soundfile", "librosa", "datasets>=2.14.0"],
    "musicgen": ["audiocraft", "einops", "scipy", "librosa"],
    "bark": ["encodec", "scipy", "tokenizers"],
    "clap": ["librosa", "soundfile", "torchaudio"],
    
    # Vision and multimodal
    "blip": ["Pillow", "torchvision"],
    "blip2": ["Pillow", "torchvision", "transformers>=4.30.0"],
    "clip": ["ftfy", "regex", "tqdm", "Pillow"],
    "xclip": ["Pillow", "decord", "torchvision"],
    "sam": ["opencv-python", "Pillow", "matplotlib"],
    "vit": ["timm>=0.9.2", "Pillow"],
    "deit": ["timm>=0.9.2", "Pillow", "torchvision"],
    "swin": ["timm>=0.9.2", "Pillow", "torchvision"],
    "swinv2": ["timm>=0.9.2", "Pillow", "torchvision"],
    "convnext": ["timm>=0.9.2", "Pillow", "torchvision"],
    "vit_mae": ["timm>=0.9.2", "Pillow", "transformers>=4.27.0"],
    "layoutlm": ["pytesseract", "Pillow", "pdf2image", "opencv-python"],
    "layoutlmv2": ["pytesseract", "Pillow", "pdf2image", "opencv-python", "datasets"],
    "detr": ["opencv-python", "timm", "scipy"],
    "nougat": ["pypdf", "datasets", "nltk", "Pillow", "scikit-image"],
    
    # Core language models
    "bert": ["tokenizers>=0.11.0", "sentencepiece"],
    "roberta": ["tokenizers>=0.11.0"],
    "t5": ["sentencepiece", "tokenizers"],
    "gpt2": ["regex"],
    "gptj": ["einops", "accelerate>=0.16.0"],
    "opt": ["accelerate>=0.16.0"],
    "bloom": ["accelerate>=0.16.0", "einops"],
    
    # Multimodal
    "llava": ["Pillow", "matplotlib", "torchvision", "accelerate"],
    "videomae": ["decord", "transformers>=4.27.0", "Pillow", "torchvision"],
    "idefics": ["Pillow", "transformers>=4.31.0", "accelerate>=0.20.0"],
    "tapas": ["pandas"],
    "donut": ["sentencepiece", "datasets", "Pillow", "seqeval"],
    
    # Special models
    "peft": ["peft>=0.4.0", "accelerate>=0.20.0"],
    "lora": ["peft>=0.4.0", "accelerate>=0.20.0"]
}

def normalize_model_name(name):
    """Normalize model name to match file naming conventions."""
    # Replace slashes (org/model format) with double underscore
    normalized = name.replace('/', '__')
    # Replace other special characters
    normalized = normalized.replace('-', '_').replace('.', '_').lower()
    return normalized

def get_category_from_task(task_type):
    """Determine the model category from its task."""
    language_tasks = ["fill-mask", "text-generation", "text2text-generation", "question-answering", "summarization"]
    vision_tasks = ["image-classification", "object-detection", "image-segmentation"]
    audio_tasks = ["automatic-speech-recognition", "audio-classification", "text-to-audio"]
    multimodal_tasks = ["image-to-text", "visual-question-answering", "document-question-answering"]
    
    if task_type in language_tasks:
        return "language"
    elif task_type in vision_tasks:
        return "vision"
    elif task_type in audio_tasks:
        return "audio"
    elif task_type in multimodal_tasks:
        return "multimodal"
    else:
        return "language"  # Default

def get_appropriate_model_class(task_type, model_name=None):
    """Get the appropriate model class for a task and model combination."""
    # First check for special model families that need custom handling
    if model_name:
        lower_name = model_name.lower()
        if "llava" in lower_name:
            return "transformers.LlavaForConditionalGeneration"
        elif "blip2" in lower_name:
            return "transformers.Blip2ForConditionalGeneration"
        elif "sam" in lower_name:
            return "transformers.SamModel"
        elif "clap" in lower_name:
            return "transformers.ClapModel"
        elif "musicgen" in lower_name:
            return "transformers.MusicgenForConditionalGeneration"
        elif "idefics" in lower_name:
            return "transformers.IdeficsForVisionText2Text"
        elif "vit-mae" in lower_name or "vitmae" in lower_name:
            return "transformers.ViTMAEModel"
        elif "layoutlmv2" in lower_name:
            return "transformers.LayoutLMv2Model"
        elif "layoutlmv3" in lower_name:
            return "transformers.LayoutLMv3Model"

    # Task-based model class selection
    if task_type == "text-generation":
        return "transformers.AutoModelForCausalLM"
    elif task_type == "fill-mask":
        return "transformers.AutoModelForMaskedLM"
    elif task_type == "text2text-generation":
        return "transformers.AutoModelForSeq2SeqLM"
    elif task_type == "image-classification":
        return "transformers.AutoModelForImageClassification"
    elif task_type == "object-detection":
        return "transformers.AutoModelForObjectDetection"
    elif task_type == "image-segmentation":
        return "transformers.AutoModelForImageSegmentation"
    elif task_type == "automatic-speech-recognition":
        return "transformers.AutoModelForSpeechSeq2Seq"
    elif task_type == "audio-classification":
        return "transformers.AutoModelForAudioClassification"
    elif task_type == "text-to-audio":
        return "transformers.AutoModelForTextToWaveform"
    elif task_type == "image-to-text":
        return "transformers.AutoModelForVision2Seq"
    elif task_type == "visual-question-answering":
        return "transformers.AutoModelForVisualQuestionAnswering"
    elif task_type == "document-question-answering":
        return "transformers.AutoModelForDocumentQuestionAnswering"
    elif task_type == "question-answering":
        return "transformers.AutoModelForQuestionAnswering"
    elif task_type == "summarization":
        return "transformers.AutoModelForSeq2SeqLM"
    elif task_type == "translation":
        return "transformers.AutoModelForSeq2SeqLM"
    elif task_type == "text-classification":
        return "transformers.AutoModelForSequenceClassification"
    elif task_type == "token-classification":
        return "transformers.AutoModelForTokenClassification"
    elif task_type == "table-question-answering":
        return "transformers.AutoModelForTableQuestionAnswering"
    elif task_type == "zero-shot-classification":
        return "transformers.AutoModelForSequenceClassification"
    elif task_type == "sentence-similarity":
        return "transformers.AutoModel"
    else:
        return "transformers.AutoModel"

def get_test_inputs_for_category(category, model_name=None):
    """Get appropriate test inputs based on the category and model."""
    # Extract test data based on model type if provided
    model_specific = ""
    if model_name:
        lower_name = model_name.lower()
        if "layoutlm" in lower_name:
            model_specific = '''
        # Document image with layout for LayoutLM models
        self.test_document_path = "test.jpg" # Ideally this would be a document scan
        self.test_document_query = "What is the title of this document?"
        self.test_document_words = ["Sample", "Document", "Title", "Content"]
        self.test_document_boxes = [[0, 0, 100, 20], [0, 30, 200, 50], [0, 60, 150, 80], [0, 90, 200, 110]]
        '''
        elif "sam" in lower_name:
            model_specific = '''
        # Segmentation inputs for SAM models
        self.test_points = [[100, 100]]  # Example points to prompt the model
        self.test_bboxes = [[50, 50, 150, 150]]  # Example bounding box
        '''
        elif "llava" in lower_name or "idefics" in lower_name:
            model_specific = '''
        # Vision-language conversation inputs
        self.test_prompt = "What can you see in this image?"
        self.test_conversation = [
            {"role": "user", "content": "What can you see in this image?"},
        ]
        self.test_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What can you see in this image?"},
                {"type": "image_url", "image_url": {"url": "test.jpg"}}
            ]}
        ]
        '''
        elif "musicgen" in lower_name:
            model_specific = '''
        # Music generation inputs
        self.test_music_prompt = "A cheerful piano melody with upbeat rhythm"
        self.test_duration = 5.0  # Generate 5 seconds of audio
        '''
    
    # Base inputs by category
    if category == "language":
        base_inputs = '''
        # Text inputs
        self.test_text = "The quick brown fox jumps over the lazy dog"
        self.test_batch = ["The quick brown fox jumps over the lazy dog", "The five boxing wizards jump quickly"]
        self.test_prompt = "Complete this sentence: The quick brown fox"
        self.test_query = "What is the capital of France?"
        self.test_pairs = [("What is the capital of France?", "Paris"), ("Who wrote Hamlet?", "Shakespeare")]
        self.test_long_text = """This is a longer piece of text that spans multiple sentences.
            It can be used for summarization, translation, or other text2text tasks.
            The model should be able to process this multi-line input appropriately."""
        '''
    elif category == "vision":
        base_inputs = '''
        # Image inputs
        self.test_image_path = "test.jpg"
        self.test_image_urls = ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks.png"]
        try:
            from PIL import Image
            if os.path.exists("test.jpg"):
                self.test_image = Image.open("test.jpg")
            else:
                print("Warning: test.jpg not found, some tests may fail")
                self.test_image = None
        except ImportError:
            print("Warning: PIL not available, using mock")
            self.test_image = Mock()
            self.test_image.size = (224, 224)
        
        # Create mock image tensor if torch is available
        try:
            if HAS_TORCH and self.test_image:
                self.test_image_tensor = torch.rand(3, 224, 224)
            else:
                self.test_image_tensor = None
        except:
            self.test_image_tensor = None
        '''
    elif category == "audio":
        base_inputs = '''
        # Audio inputs
        self.test_audio_path = "test.mp3"
        self.test_audio_batch = ["test.mp3", "trans_test.mp3"] if os.path.exists("trans_test.mp3") else ["test.mp3", "test.mp3"]
        self.test_transcription = "Hello world"
        
        try:
            import librosa
            if os.path.exists("test.mp3"):
                self.test_audio, self.test_sr = librosa.load("test.mp3", sr=16000)
                # Create a short synthetic audio if needed
                if len(self.test_audio) < 16000:
                    import numpy as np
                    self.test_audio = np.sin(2 * np.pi * np.arange(16000) * 440 / 16000).astype(np.float32)
            else:
                print("Warning: test.mp3 not found, creating synthetic audio")
                import numpy as np
                self.test_audio = np.sin(2 * np.pi * np.arange(16000) * 440 / 16000).astype(np.float32)
                self.test_sr = 16000
        except ImportError:
            print("Warning: librosa not available, using mock")
            import numpy as np
            self.test_audio = np.zeros(16000, dtype=np.float32)
            self.test_sr = 16000
            
        # Create torch tensor of audio if torch is available
        try:
            if HAS_TORCH:
                self.test_audio_tensor = torch.tensor(self.test_audio).unsqueeze(0)
            else:
                self.test_audio_tensor = None
        except:
            self.test_audio_tensor = None
        '''
    elif category == "multimodal":
        base_inputs = '''
        # Multimodal inputs
        self.test_image_path = "test.jpg"
        self.test_text = "What is shown in this image?"
        self.test_vqa_prompt = "What objects are in this image?"
        self.test_caption_prompt = "Describe this image in detail."
        
        try:
            from PIL import Image
            if os.path.exists("test.jpg"):
                self.test_image = Image.open("test.jpg")
            else:
                print("Warning: test.jpg not found, some tests may fail")
                self.test_image = None
        except ImportError:
            print("Warning: PIL not available, using mock")
            self.test_image = Mock()
            self.test_image.size = (224, 224)
            
        # Create torch tensor of image if torch is available
        try:
            if HAS_TORCH and self.test_image:
                self.test_image_tensor = torch.rand(3, 224, 224)
            else:
                self.test_image_tensor = None
        except:
            self.test_image_tensor = None
            
        # Combined input formats
        self.test_image_text_pair = {"image": self.test_image_path, "text": self.test_text}
        self.test_vqa_input = {"image": self.test_image_path, "question": self.test_vqa_prompt}
        '''
    else:
        base_inputs = '''
        # Default inputs
        self.test_text = "The quick brown fox jumps over the lazy dog"
        '''
    
    # Combine base inputs with model-specific inputs
    return base_inputs + model_specific

def get_pipeline_input(category, task_type, model_name=None):
    """Get appropriate pipeline input based on category, task, and model."""
    # Special model-specific inputs
    if model_name:
        lower_name = model_name.lower()
        if "sam" in lower_name:
            return "{'image': self.test_image, 'points_per_batch': self.test_points}"
        elif "layoutlm" in lower_name and task_type == "document-question-answering":
            return "{'image': self.test_image, 'question': self.test_document_query}"
        elif "musicgen" in lower_name:
            return "self.test_music_prompt"
        elif "llava" in lower_name:
            return "{'image': self.test_image, 'text': self.test_prompt}"
    
    # Task-based input selection
    if task_type == "text-generation":
        return "self.test_prompt"
    elif task_type == "fill-mask":
        return "self.test_text.replace('lazy', '[MASK]')"
    elif task_type == "question-answering":
        return "{'question': self.test_query, 'context': self.test_long_text}"
    elif task_type == "summarization":
        return "self.test_long_text"
    elif task_type == "text2text-generation":
        return "self.test_long_text"
    elif task_type == "translation":
        return "self.test_text"
    
    # Category-based inputs (fallback)
    if category == "language":
        return "self.test_text"
    elif category == "vision":
        return "self.test_image_path if os.path.exists('test.jpg') else 'test.jpg'"
    elif category == "audio":
        return "self.test_audio_path if os.path.exists('test.mp3') else 'test.mp3'"
    elif category == "multimodal":
        if task_type == "visual-question-answering":
            return "self.test_vqa_input"
        elif task_type == "image-to-text":
            return "self.test_image_path if os.path.exists('test.jpg') else 'test.jpg'"
        else:
            return "self.test_image_path if os.path.exists('test.jpg') else 'test.jpg'"
    else:
        return "self.test_text"

def generate_test_file(model_name, normalized_name, task_type, output_dir, deps_info=None):
    """Generate a comprehensive test file for a model with all testing approaches."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine category and model class
    category = get_category_from_task(task_type)
    model_class = get_appropriate_model_class(task_type, model_name)
    test_inputs = get_test_inputs_for_category(category, model_name)
    pipeline_input = get_pipeline_input(category, task_type, model_name)
    
    # Process dependency information
    if deps_info is None:
        deps_info = get_model_dependencies(model_name)
    
    use_remote_code = deps_info.get("use_remote_code", False)
    dependencies = deps_info.get("dependencies", [])
    
    # Create imports and mocks for dependencies
    dependency_imports = ""
    dependency_mocks = ""
    dependency_check_vars = ""
    
    for dep in dependencies:
        base_dep = dep.split(">=")[0].split(">")[0].strip()
        var_name = base_dep.replace("-", "_").upper()
        
        dependency_check_vars += f"HAS_{var_name} = False\n"
        
        # Import statement with simplified version check
        dependency_imports += f"""
# Try to import {base_dep}
try:
    import {base_dep}
    HAS_{var_name} = True
except ImportError:
    {base_dep} = MagicMock()
    print(f"Warning: {base_dep} not available, using mock")
"""
        
        # Create specific mocks for known dependencies
        if base_dep == "sentencepiece":
            dependency_mocks += """
# Mock for sentencepiece
class MockSentencePieceProcessor:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 32000
        
    def encode(self, text, out_type=str):
        return [1, 2, 3, 4, 5]
        
    def decode(self, ids):
        return "Decoded text from mock"
        
    def get_piece_size(self):
        return 32000
        
    @staticmethod
    def load(model_file):
        return MockSentencePieceProcessor()

if not HAS_SENTENCEPIECE:
    sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor
"""
        elif base_dep == "tokenizers":
            dependency_mocks += """
# Mock for tokenizers
class MockTokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 32000
        
    def encode(self, text, **kwargs):
        return {"ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
        
    def decode(self, ids, **kwargs):
        return "Decoded text from mock"
        
    @staticmethod
    def from_file(vocab_filename):
        return MockTokenizer()

if not HAS_TOKENIZERS:
    tokenizers.Tokenizer = MockTokenizer
"""
        elif base_dep == "safetensors":
            dependency_mocks += """
# Mock for safetensors
class MockSafeTensors:
    @staticmethod
    def safe_open(path, framework="pt", device="cpu"):
        return {"mock_tensor": torch.ones(10, 10) if HAS_TORCH else None}
        
    @staticmethod
    def load_file(path, device="cpu"):
        return {"mock_tensor": torch.ones(10, 10) if HAS_TORCH else None}

if not HAS_SAFETENSORS:
    # Create module structure if needed
    if not hasattr(sys.modules, "safetensors"):
        sys.modules["safetensors"] = MagicMock()
    if not hasattr(sys.modules, "safetensors.torch"):
        sys.modules["safetensors.torch"] = MagicMock()
        
    # Assign mocks
    safetensors.torch.load_file = MockSafeTensors.load_file
    safetensors.safe_open = MockSafeTensors.safe_open
"""
        elif base_dep == "peft":
            dependency_mocks += """
# Mock for PEFT
class MockPeftModel:
    @staticmethod
    def from_pretrained(model, peft_model_id, **kwargs):
        return model
        
class MockPeftConfig:
    @staticmethod
    def from_pretrained(peft_model_id):
        return {"peft_type": "mock"}

if not HAS_PEFT:
    # Create module structure
    if not hasattr(sys.modules, "peft"):
        sys.modules["peft"] = MagicMock()
        
    # Assign mocks
    peft.PeftModel = MockPeftModel
    peft.PeftConfig = MockPeftConfig
"""
        elif base_dep == "accelerate":
            dependency_mocks += """
# Mock for accelerate
class MockAccelerator:
    def __init__(self, *args, **kwargs):
        pass
        
    def prepare(self, *args, **kwargs):
        return args
        
    def unwrap_model(self, model):
        return model
        
    def device(self):
        return "cpu"

if not HAS_ACCELERATE:
    # Create module structure
    if not hasattr(sys.modules, "accelerate"):
        sys.modules["accelerate"] = MagicMock()
        
    # Assign mocks
    accelerate.Accelerator = MockAccelerator
"""
    
    # Add hardware detection code
    hardware_detection = """
# Hardware detection
def check_hardware():
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available()
        if capabilities["cuda"]:
            capabilities["cuda_devices"] = torch.cuda.device_count()
            capabilities["cuda_version"] = torch.version.cuda
    
    # Check MPS (Apple Silicon)
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        capabilities["mps"] = torch.mps.is_available()
    
    # Check OpenVINO
    try:
        import openvino
        capabilities["openvino"] = True
    except ImportError:
        pass
    
    return capabilities

# Get hardware capabilities
HW_CAPABILITIES = check_hardware()
"""
    
    # Build the template with dependencies
    remote_code_import = """
# Support for models that require trust_remote_code
# This is important for models with custom code like LLaVA, SAM, BLIP, etc.
"""

    content = f'''#!/usr/bin/env python3
"""
Comprehensive test file for {model_name}
- Tests both pipeline() and from_pretrained() methods
- Includes CPU, CUDA, and OpenVINO hardware support
- Handles missing dependencies with sophisticated mocks
- Supports benchmarking with multiple input sizes
- Tracks hardware-specific performance metrics
- Reports detailed dependency information
"""

import os
import sys
import json
import time
import datetime
import traceback
import logging
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    print("Warning: torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, using mock")

# Additional imports based on model type
if "{category}" == "vision" or "{category}" == "multimodal":
    try:
        from PIL import Image
        HAS_PIL = True
    except ImportError:
        Image = MagicMock()
        HAS_PIL = False
        print("Warning: PIL not available, using mock")

if "{category}" == "audio":
    try:
        import librosa
        HAS_LIBROSA = True
    except ImportError:
        librosa = MagicMock()
        HAS_LIBROSA = False
        print("Warning: librosa not available, using mock")

{dependency_imports if dependencies else ""}

{remote_code_import if use_remote_code else ""}

{dependency_mocks if dependencies else ""}

{hardware_detection}

# Check for other required dependencies
{dependency_check_vars if dependencies else ""}

class test_hf_{normalized_name}:
    def __init__(self):
        # Use appropriate model for testing
        self.model_name = "{model_name}"
        
        # Test inputs appropriate for this model type
        {test_inputs}
        
        # Results storage
        self.examples = []
        self.performance_stats = {{}}
        
        # Hardware selection for testing (prioritize CUDA if available)
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
            
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
    def get_input_for_pipeline(self):
        """Get appropriate input for pipeline testing based on model type."""
        return {pipeline_input}
        
    def test_pipeline(self, device="auto"):
        """Test using the transformers pipeline() method."""
        results = {{}}
        
        if device == "auto":
            device = self.preferred_device
        
        results["device"] = device
        
        if not HAS_TRANSFORMERS:
            results["pipeline_test"] = "Transformers not available"
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            return results
            
        # Check required dependencies for this model
        missing_deps = []
        {"""
        # Check each dependency
        """ + "".join([f"""
        if not HAS_{dep.split('>=')[0].split('>')[0].strip().replace('-', '_').upper()}:
            missing_deps.append("{dep}")
        """ for dep in dependencies]) if dependencies else ""}
        
        if missing_deps:
            results["pipeline_missing_deps"] = missing_deps
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_test"] = f"Missing dependencies: {{', '.join(missing_deps)}}"
            return results
            
        try:
            logger.info(f"Testing {normalized_name} with pipeline() on {{device}}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {{
                "task": "{task_type}",
                "model": self.model_name,
                "trust_remote_code": {str(use_remote_code).lower()},
                "device": device
            }}
            
            # Time the model loading separately
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            results["pipeline_load_time"] = load_time
            
            # Get appropriate input
            pipeline_input = self.get_input_for_pipeline()
            
            # Run warmup inference if on CUDA
            if device == "cuda":
                try:
                    _ = pipeline(pipeline_input)
                except Exception:
                    pass
            
            # Run multiple inferences for better timing
            num_runs = 3
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                output = pipeline(pipeline_input)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_avg_time"] = avg_time
            results["pipeline_min_time"] = min_time
            results["pipeline_max_time"] = max_time
            results["pipeline_times"] = times
            results["pipeline_uses_remote_code"] = {str(use_remote_code).lower()}
            
            # Add error type classification for detailed tracking
            results["pipeline_error_type"] = "none"
            
            # Store in performance stats
            self.performance_stats[f"pipeline_{{device}}"] = {{
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "load_time": load_time,
                "num_runs": num_runs
            }}
            
            # Add to examples
            self.examples.append({{
                "method": f"pipeline() on {{device}}",
                "input": str(pipeline_input),
                "output_type": str(type(output)),
                "output": str(output)[:500] + ("..." if str(output) and len(str(output)) > 500 else "")
            }})
            
        except Exception as e:
            # Store basic error info
            results["pipeline_error"] = str(e)
            results["pipeline_traceback"] = traceback.format_exc()
            logger.error(f"Error testing pipeline on {{device}}: {{e}}")
            
            # Classify error type for better diagnostics
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["pipeline_error_type"] = "cuda_error"
            elif "memory" in error_str or "cuda out of memory" in traceback_str:
                results["pipeline_error_type"] = "out_of_memory"
            elif "trust_remote_code" in error_str:
                results["pipeline_error_type"] = "remote_code_required"
            elif "permission" in error_str or "access" in error_str:
                results["pipeline_error_type"] = "permission_error"
            elif "module" in error_str and "has no attribute" in error_str:
                results["pipeline_error_type"] = "missing_attribute"
            elif "no module named" in error_str.lower():
                results["pipeline_error_type"] = "missing_dependency"
                # Try to extract the missing module name
                import re
                match = re.search(r"no module named '([^']+)'", error_str.lower())
                if match:
                    results["pipeline_missing_module"] = match.group(1)
            else:
                results["pipeline_error_type"] = "other"
            
        return results
        
    def test_from_pretrained(self, device="auto"):
        """Test using from_pretrained() method."""
        results = {{}}
        
        if device == "auto":
            device = self.preferred_device
        
        results["device"] = device
        
        if not HAS_TRANSFORMERS:
            results["from_pretrained_test"] = "Transformers not available"
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_core"] = ["transformers"]
            return results
            
        # Check required dependencies for this model
        missing_deps = []
        {"""
        # Check each dependency
        """ + "".join([f"""
        if not HAS_{dep.split('>=')[0].split('>')[0].strip().replace('-', '_').upper()}:
            missing_deps.append("{dep}")
        """ for dep in dependencies]) if dependencies else ""}
        
        if missing_deps:
            results["from_pretrained_missing_deps"] = missing_deps
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_test"] = f"Missing dependencies: {{', '.join(missing_deps)}}"
            return results
            
        try:
            logger.info(f"Testing {normalized_name} with from_pretrained() on {{device}}...")
            
            # Record remote code requirements
            results["requires_remote_code"] = {str(use_remote_code).lower()}
            if {str(use_remote_code).lower()}:
                results["remote_code_reason"] = "Model requires custom code"
            
            # Common parameters for loading model components
            pretrained_kwargs = {{
                "trust_remote_code": {str(use_remote_code).lower()},
                "local_files_only": False
            }}
            
            # Time tokenizer loading
            tokenizer_load_start = time.time()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                **pretrained_kwargs
            )
            tokenizer_load_time = time.time() - tokenizer_load_start
            
            # Time model loading
            model_load_start = time.time()
            model = {model_class}.from_pretrained(
                self.model_name,
                **pretrained_kwargs
            )
            model_load_time = time.time() - model_load_start
            
            # Move model to device
            if device != "cpu":
                model = model.to(device)
                
            # Get input based on model category
            if "{category}" == "language":
                # Tokenize input
                inputs = tokenizer(self.test_text, return_tensors="pt")
                # Move inputs to device
                if device != "cpu":
                    inputs = {{key: val.to(device) for key, val in inputs.items()}}
                
            elif "{category}" == "vision":
                # Use image inputs
                if hasattr(self, "test_image_tensor") and self.test_image_tensor is not None:
                    inputs = {{"pixel_values": self.test_image_tensor.unsqueeze(0)}}
                    if device != "cpu":
                        inputs = {{key: val.to(device) for key, val in inputs.items()}}
                else:
                    results["from_pretrained_test"] = "Image tensor not available"
                    return results
                    
            elif "{category}" == "audio":
                # Use audio inputs
                if hasattr(self, "test_audio_tensor") and self.test_audio_tensor is not None:
                    inputs = {{"input_values": self.test_audio_tensor}}
                    if device != "cpu":
                        inputs = {{key: val.to(device) for key, val in inputs.items()}}
                else:
                    results["from_pretrained_test"] = "Audio tensor not available"
                    return results
                    
            elif "{category}" == "multimodal":
                # Use combined inputs based on model
                results["from_pretrained_test"] = "Complex multimodal input not implemented for direct model testing"
                return results
            else:
                # Default to text input
                inputs = tokenizer(self.test_text, return_tensors="pt")
                if device != "cpu":
                    inputs = {{key: val.to(device) for key, val in inputs.items()}}
            
            # Run warmup inference if using CUDA
            if device == "cuda":
                try:
                    with torch.no_grad():
                        _ = model(**inputs)
                except Exception:
                    pass
            
            # Run multiple inference passes for better timing
            num_runs = 3
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(**inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Get model size if possible
            model_size_mb = None
            try:
                model_size_params = sum(p.numel() for p in model.parameters())
                model_size_mb = model_size_params * 4 / (1024 * 1024)  # Rough estimate in MB
            except Exception:
                pass
            
            # Store results
            results["from_pretrained_success"] = True
            results["from_pretrained_avg_time"] = avg_time
            results["from_pretrained_min_time"] = min_time
            results["from_pretrained_max_time"] = max_time
            results["from_pretrained_times"] = times
            results["tokenizer_load_time"] = tokenizer_load_time
            results["model_load_time"] = model_load_time
            results["model_size_mb"] = model_size_mb
            results["from_pretrained_uses_remote_code"] = {str(use_remote_code).lower()}
            
            # Store in performance stats
            self.performance_stats[f"from_pretrained_{{device}}"] = {{
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "tokenizer_load_time": tokenizer_load_time,
                "model_load_time": model_load_time,
                "model_size_mb": model_size_mb,
                "num_runs": num_runs
            }}
            
            # Add to examples
            self.examples.append({{
                "method": f"from_pretrained() on {{device}}",
                "input_keys": str(list(inputs.keys())),
                "output_type": str(type(outputs)),
                "output_keys": str(outputs._fields if hasattr(outputs, "_fields") else list(outputs.keys()) if hasattr(outputs, "keys") else "N/A"),
                "has_logits": hasattr(outputs, "logits")
            }})
            
        except Exception as e:
            # Store basic error info
            results["from_pretrained_error"] = str(e)
            results["from_pretrained_traceback"] = traceback.format_exc()
            logger.error(f"Error testing from_pretrained on {{device}}: {{e}}")
            
            # Classify error type for better diagnostics
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["from_pretrained_error_type"] = "cuda_error"
            elif "memory" in error_str or "cuda out of memory" in traceback_str:
                results["from_pretrained_error_type"] = "out_of_memory"
            elif "trust_remote_code" in error_str:
                results["from_pretrained_error_type"] = "remote_code_required"
            elif "permission" in error_str or "access" in error_str:
                results["from_pretrained_error_type"] = "permission_error"
            elif "module" in error_str and "has no attribute" in error_str:
                results["from_pretrained_error_type"] = "missing_attribute"
            elif "no module named" in error_str.lower():
                results["from_pretrained_error_type"] = "missing_dependency"
                # Try to extract the missing module name
                import re
                match = re.search(r"no module named '([^']+)'", error_str.lower())
                if match:
                    results["from_pretrained_missing_module"] = match.group(1)
            elif "could not find model" in error_str or "404" in error_str:
                results["from_pretrained_error_type"] = "model_not_found"
            else:
                results["from_pretrained_error_type"] = "other"
            
        return results
        
    def test_with_openvino(self):
        """Test model with OpenVINO if available."""
        results = {{}}
        
        if not HW_CAPABILITIES["openvino"]:
            results["openvino_test"] = "OpenVINO not available"
            return results
            
        try:
            from optimum.intel import OVModelForSequenceClassification, OVModelForCausalLM
            
            # Load the model with OpenVINO
            logger.info(f"Testing {normalized_name} with OpenVINO...")
            
            # Determine which OV model class to use based on task
            if "{task_type}" == "text-generation":
                ov_model_class = OVModelForCausalLM
            else:
                ov_model_class = OVModelForSequenceClassification
            
            # Load tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with OpenVINO
            load_start_time = time.time()
            model = ov_model_class.from_pretrained(
                self.model_name,
                export=True,
                trust_remote_code={str(use_remote_code).lower()}
            )
            load_time = time.time() - load_start_time
            
            # Tokenize input
            inputs = tokenizer(self.test_text, return_tensors="pt")
            
            # Run inference
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
            
            # Store results
            results["openvino_success"] = True
            results["openvino_load_time"] = load_time
            results["openvino_inference_time"] = inference_time
            
            # Store in performance stats
            self.performance_stats["openvino"] = {{
                "load_time": load_time,
                "inference_time": inference_time
            }}
            
            # Add to examples
            self.examples.append({{
                "method": "OpenVINO inference",
                "input": self.test_text,
                "output_type": str(type(outputs)),
                "has_logits": hasattr(outputs, "logits")
            }})
            
        except Exception as e:
            results["openvino_error"] = str(e)
            results["openvino_traceback"] = traceback.format_exc()
            logger.error(f"Error testing with OpenVINO: {{e}}")
            
        return results
        
    def run_all_hardware_tests(self):
        """Run tests on all available hardware."""
        all_results = {{}}
        
        # Always run CPU tests
        cpu_pipeline_results = self.test_pipeline(device="cpu")
        all_results["cpu_pipeline"] = cpu_pipeline_results
        
        cpu_pretrained_results = self.test_from_pretrained(device="cpu")
        all_results["cpu_pretrained"] = cpu_pretrained_results
        
        # Run CUDA tests if available
        if HW_CAPABILITIES["cuda"]:
            cuda_pipeline_results = self.test_pipeline(device="cuda")
            all_results["cuda_pipeline"] = cuda_pipeline_results
            
            cuda_pretrained_results = self.test_from_pretrained(device="cuda")
            all_results["cuda_pretrained"] = cuda_pretrained_results
        
        # Run OpenVINO tests if available
        if HW_CAPABILITIES["openvino"]:
            openvino_results = self.test_with_openvino()
            all_results["openvino"] = openvino_results
        
        return all_results
        
    def run_tests(self):
        """Run all tests and return results."""
        # Collect hardware capabilities
        hw_info = {{
            "capabilities": HW_CAPABILITIES,
            "preferred_device": self.preferred_device
        }}
        
        # Run tests on preferred device
        pipeline_results = self.test_pipeline()
        pretrained_results = self.test_from_pretrained()
        
        # Build dependency information
        dependency_status = {{}}
        {"""
        # Check each dependency
        """ + "".join([f"""
        dependency_status["{dep}"] = HAS_{dep.split('>=')[0].split('>')[0].strip().replace('-', '_').upper()}
        """ for dep in dependencies]) if dependencies else ""}
        
        # Run all hardware tests if --all-hardware flag is provided
        all_hardware_results = None
        if "--all-hardware" in sys.argv:
            all_hardware_results = self.run_all_hardware_tests()
        
        return {{
            "results": {{
                "pipeline": pipeline_results,
                "from_pretrained": pretrained_results,
                "all_hardware": all_hardware_results
            }},
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": hw_info,
            "metadata": {{
                "model": self.model_name,
                "category": "{category}",
                "task": "{task_type}",
                "timestamp": datetime.datetime.now().isoformat(),
                "generation_timestamp": "{timestamp}",
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "dependencies": dependency_status,
                "uses_remote_code": {"False" if not use_remote_code else "True"}
            }}
        }}
        
if __name__ == "__main__":
    logger.info(f"Running tests for {normalized_name}...")
    tester = test_hf_{normalized_name}()
    test_results = tester.run_tests()
    
    # Save results to file if --save flag is provided
    if "--save" in sys.argv:
        output_dir = "collected_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"hf_{normalized_name}_test_results.json")
        with open(output_file, "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Saved results to {{output_file}}")
    
    # Print summary results
    print("\\nTEST RESULTS SUMMARY:")
    if test_results["results"]["pipeline"].get("pipeline_success", False):
        pipeline_time = test_results["results"]["pipeline"].get("pipeline_avg_time", 0)
        print(f"✅ Pipeline test successful ({{pipeline_time:.4f}}s)")
    else:
        error = test_results["results"]["pipeline"].get("pipeline_error", "Unknown error")
        print(f"❌ Pipeline test failed: {{error}}")
        
    if test_results["results"]["from_pretrained"].get("from_pretrained_success", False):
        model_time = test_results["results"]["from_pretrained"].get("from_pretrained_avg_time", 0)
        print(f"✅ from_pretrained test successful ({{model_time:.4f}}s)")
    else:
        error = test_results["results"]["from_pretrained"].get("from_pretrained_error", "Unknown error")
        print(f"❌ from_pretrained test failed: {{error}}")
        
    # Show top 3 examples
    if test_results["examples"]:
        print("\\nEXAMPLES:")
        for i, example in enumerate(test_results["examples"][:2]):
            print(f"Example {{i+1}}: {{example['method']}}")
            if "input" in example:
                print(f"  Input: {{example['input']}}")
            if "output_type" in example:
                print(f"  Output type: {{example['output_type']}}")
                
    print("\\nFor detailed results, use --save flag and check the JSON output file.")
'''
    
    # Determine whether to validate based on args
    should_validate = HAS_VALIDATOR and (args.validate or not args.skip_validation)
    
    # Validate template before saving
    if should_validate:
        logger.info(f"Validating template for {model_name}...")
        is_valid, validation_errors = validate_template_for_generator(
            content, 
            "simple_test_generator",
            validate_hardware=True,
            check_resource_pool=False
        )
        
        if not is_valid:
            logger.warning(f"Template validation failed for {model_name}:")
            for error in validation_errors:
                logger.warning(f"  - {error}")
            
            # Decide whether to continue based on error severity
            # For now, we'll continue with a warning
            logger.warning("Continuing with generation despite validation errors.")
        else:
            logger.info(f"Template validation passed for {model_name}")
    elif args.validate and not HAS_VALIDATOR:
        logger.warning("Template validation requested but validator not available. Skipping validation.")

    # Save file
    output_path = os.path.join(output_dir, f"{TEST_FILE_PREFIX}{normalized_name}.py")
    
    with open(output_path, 'w') as f:
        f.write(content)
        
    # Make executable
    os.chmod(output_path, 0o755)
    
    return output_path

def check_module_installed(module_name):
    """Check if a Python module is installed."""
    try:
        spec = importlib.util.find_spec(module_name.split(">=")[0].split(">")[0].strip())
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False

def get_model_dependencies(model_name):
    """
    Determine external dependencies and remote code requirements for a model.
    
    This function checks:
    1. Known dependencies based on model type/family
    2. Hugging Face model card requirements and tags
    3. Custom code repositories and non-standard dependencies
    4. Required trust_remote_code flag settings
    
    Returns a comprehensive dictionary with dependency information.
    """
    dependencies = []
    use_remote_code = False
    remote_code_reason = None
    special_installation_notes = []
    
    # First check for known dependencies based on model name
    for model_type, deps in KNOWN_DEPENDENCIES.items():
        if model_type.lower() in model_name.lower():
            dependencies.extend(deps)
    
    # Try to get detailed info from Hugging Face API
    try:
        model_id = model_name.replace("__", "/")
        api_url = f"https://huggingface.co/api/models/{model_id}"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            model_info = response.json()
            
            # Check model tags for remote code indicators
            tags = model_info.get("tags", [])
            
            # Models that explicitly require remote code
            if "trust_remote_code" in tags or "requires-trust-remote-code" in tags:
                use_remote_code = True
                remote_code_reason = "Model is tagged with 'trust_remote_code'"
                
            # Check for custom code in the repository
            if "custom-code" in tags or "custom-modeling-class" in tags:
                use_remote_code = True
                remote_code_reason = "Model uses custom modeling code"
            
            # Check pipeline and architecture for remote code patterns
            pipeline_tag = model_info.get("pipeline_tag")
            model_arch = model_info.get("model-architecture", "")
            
            # Check for specialized LLM architectures
            if pipeline_tag == "text-generation":
                # Known LLM architectures needing remote code
                if any(arch in model_name.lower() for arch in ["llama", "falcon", "mistral", "mixtral", "mamba"]):
                    use_remote_code = True
                    remote_code_reason = f"LLM architecture '{model_name}' often requires remote code"
                
                # Check for quantized models which often need remote code
                if any(qt in model_name.lower() or qt in tags for qt in ["gguf", "ggml", "awq", "gptq", "quantized"]):
                    use_remote_code = True
                    remote_code_reason = "Quantized model often requires special handlers"
                    special_installation_notes.append("Model may require special quantization libraries")
            
            # Check for vision-language models needing remote code
            if any(x in pipeline_tag for x in ["image-to-text", "visual-question-answering"]):
                if any(arch in model_name.lower() for arch in ["llava", "blip", "flamingo", "idefics"]):
                    use_remote_code = True
                    remote_code_reason = f"Vision-language model '{model_name}' requires remote code"
            
            # Check for segment anything models
            if "sam" in model_name.lower() or "segment-anything" in tags:
                use_remote_code = True
                remote_code_reason = "Segment Anything Model requires remote code"
            
            # Check for advanced multimodal models
            if "video" in pipeline_tag or "speech" in pipeline_tag:
                use_remote_code = True
                remote_code_reason = f"Complex {pipeline_tag} model may need special handlers"
            
            # Check for transformers library version requirements in tags
            for tag in tags:
                if tag.startswith("transformers") and ">=" in tag:
                    version_req = tag.split(">=")[1].strip()
                    special_installation_notes.append(f"Requires transformers>={version_req}")
                    dependencies.append(f"transformers>={version_req}")
            
            # Check library name in model card if available
            if "library_name" in model_info and model_info["library_name"] != "transformers":
                library = model_info["library_name"]
                if library == "sentence_transformers":
                    dependencies.append("sentence-transformers")
                    special_installation_notes.append("Requires sentence-transformers package")
                else:
                    dependencies.append(library)
                    special_installation_notes.append(f"Uses non-standard library: {library}")
    except Exception as e:
        logger.warning(f"Error fetching model info for {model_name}: {e}")
    
    # Check for specific model families known to require remote code
    model_lower = model_name.lower()
    
    # Check for specialized vision models
    if "sam" in model_lower or "segment-anything" in model_lower:
        use_remote_code = True
        remote_code_reason = "Segment Anything Model requires custom code"
        
    # Check for multimodal models
    elif "llava" in model_lower:
        use_remote_code = True
        remote_code_reason = "LLaVA multimodal models require custom processors"
        
    elif "blip" in model_lower:
        use_remote_code = True
        remote_code_reason = "BLIP models often require custom processors"
    
    # Check for specialized audio models
    elif "musicgen" in model_lower or "audiogen" in model_lower:
        use_remote_code = True
        remote_code_reason = "Audio generation models require custom code"
        
    # Check for specialized NeRF/3D models
    elif "nerf" in model_lower or "3d" in model_lower:
        use_remote_code = True
        remote_code_reason = "3D models typically require custom code"
    
    # Remove duplicates while preserving order
    seen = set()
    unique_deps = []
    for dep in dependencies:
        if dep not in seen:
            seen.add(dep)
            unique_deps.append(dep)
    
    # Check which dependencies are installed
    installed_deps = []
    missing_deps = []
    for dep in unique_deps:
        base_dep = dep.split(">=")[0].split(">")[0].strip()
        if check_module_installed(base_dep):
            installed_deps.append(dep)
        else:
            missing_deps.append(dep)
    
    return {
        "dependencies": unique_deps,
        "installed": installed_deps,
        "missing": missing_deps,
        "use_remote_code": use_remote_code,
        "remote_code_reason": remote_code_reason,
        "special_installation_notes": special_installation_notes
    }

def update_dependencies_files(dependencies_dict):
    """Update the JSON and MD files with model dependencies and remote code requirements."""
    # Load existing dependencies if available
    if os.path.exists(MODEL_DEPENDENCIES_JSON):
        try:
            with open(MODEL_DEPENDENCIES_JSON, 'r') as f:
                existing_deps = json.load(f)
        except json.JSONDecodeError:
            existing_deps = {}
    else:
        existing_deps = {}
    
    # Update with new dependencies
    existing_deps.update(dependencies_dict)
    
    # Save updated JSON
    with open(MODEL_DEPENDENCIES_JSON, 'w') as f:
        json.dump(existing_deps, f, indent=2, sort_keys=True)
    
    # Generate enhanced markdown file
    with open(MODEL_DEPENDENCIES_MD, 'w') as f:
        # Header and introduction
        f.write("# Hugging Face Model Dependencies\n\n")
        f.write("This file tracks external dependencies and remote code requirements for different model types.\n")
        f.write("Models are categorized by their dependency needs and any special installation requirements.\n\n")
        
        # Remote code section with reasons
        f.write("## Models with Remote Code Requirements\n\n")
        f.write("The following models require `trust_remote_code=True` when loading with Transformers:\n\n")
        
        remote_code_models = sorted([(model, info) for model, info in existing_deps.items() 
                                  if info.get("use_remote_code", False)])
        
        # Create a more detailed table with reasons
        f.write("| Model | Reason for Remote Code | Special Notes |\n")
        f.write("|-------|------------------------|---------------|\n")
        
        for model, info in remote_code_models:
            reason = info.get("remote_code_reason", "Unknown reason")
            notes = ", ".join(info.get("special_installation_notes", [])) or "None"
            f.write(f"| `{model}` | {reason} | {notes} |\n")
        
        # Dependency Matrix with improved formatting
        f.write("\n## Dependency Matrix\n\n")
        f.write("This table shows which dependencies are required by each model:\n\n")
        
        # Get all unique dependencies
        all_deps = set()
        for info in existing_deps.values():
            all_deps.update(info.get("dependencies", []))
        
        all_deps = sorted(all_deps)
        
        # Create markdown table with improved formatting
        f.write("| Model | " + " | ".join([f"`{dep.split('>=')[0]}`" for dep in all_deps]) + " |\n")
        f.write("|-------|" + "|".join(["-" * (len(dep.split('>=')[0]) + 2) for dep in all_deps]) + "|\n")
        
        for model, info in sorted(existing_deps.items()):
            model_deps = info.get("dependencies", [])
            row = [f"`{model}`"]
            
            for dep in all_deps:
                base_dep = dep.split('>=')[0]
                matching_deps = [d for d in model_deps if d.startswith(base_dep)]
                
                if matching_deps:
                    # Include version if specified
                    if '>=' in matching_deps[0]:
                        version = matching_deps[0].split('>=')[1]
                        row.append(f"✅ {version}")
                    else:
                        row.append("✅")
                else:
                    row.append("")
            
            f.write("| " + " | ".join(row) + " |\n")
        
        # Add installation guide section
        f.write("\n## Installation Guide\n\n")
        f.write("### Common Dependency Groups\n\n")
        
        # Group models by dependency patterns
        dependency_groups = {}
        for model, info in existing_deps.items():
            deps_key = tuple(sorted(info.get("dependencies", [])))
            if deps_key:  # Only include models with dependencies
                if deps_key not in dependency_groups:
                    dependency_groups[deps_key] = []
                dependency_groups[deps_key].append(model)
        
        # Display installation commands for each group
        group_num = 1
        for deps, models in sorted(dependency_groups.items(), key=lambda x: len(x[1]), reverse=True):
            if len(deps) > 0:
                f.write(f"#### Group {group_num}: {len(models)} models\n\n")
                f.write("Models in this group:\n")
                for model in sorted(models):
                    remote_flag = " (requires remote code)" if existing_deps[model].get("use_remote_code", False) else ""
                    f.write(f"- `{model}`{remote_flag}\n")
                
                f.write("\nRequired dependencies:\n")
                for dep in deps:
                    f.write(f"- `{dep}`\n")
                
                # Generate pip install command
                pip_cmd = "pip install " + " ".join([f'"{dep}"' for dep in deps])
                f.write(f"\n```bash\n{pip_cmd}\n```\n\n")
                
                group_num += 1
    
    logger.info(f"Updated enhanced dependency files: {MODEL_DEPENDENCIES_JSON} and {MODEL_DEPENDENCIES_MD}")

def generate_comprehensive_model_list():
    """Generate a comprehensive list of models to test."""
    return [
        # Essential Language Models
        {"name": "bert-base-uncased", "task": "fill-mask", "priority": "high"},
        {"name": "gpt2", "task": "text-generation", "priority": "high"},
        {"name": "t5-small", "task": "text2text-generation", "priority": "high"},
        {"name": "distilroberta-base", "task": "fill-mask", "priority": "high"},
        {"name": "meta-llama/Llama-2-7b-hf", "task": "text-generation", "priority": "high"},
        {"name": "bigscience/bloom-560m", "task": "text-generation", "priority": "medium"},
        {"name": "facebook/opt-350m", "task": "text-generation", "priority": "medium"},
        {"name": "EleutherAI/gpt-j-6B", "task": "text-generation", "priority": "medium"},
        
        # Modern LLMs
        {"name": "mistralai/Mistral-7B-v0.1", "task": "text-generation", "priority": "high"},
        {"name": "mistralai/Mixtral-8x7B-v0.1", "task": "text-generation", "priority": "medium"},
        {"name": "codellama/CodeLlama-7b-hf", "task": "text-generation", "priority": "medium"},
        {"name": "google/gemma-2b", "task": "text-generation", "priority": "high"},
        {"name": "microsoft/phi-2", "task": "text-generation", "priority": "medium"},
        {"name": "Qwen/Qwen1.5-1.8B", "task": "text-generation", "priority": "medium"},
        {"name": "deepseek-ai/deepseek-llm-7b-base", "task": "text-generation", "priority": "medium"},
        {"name": "mosaicml/mpt-7b", "task": "text-generation", "priority": "low"},
        {"name": "state-spaces/mamba-2.8b", "task": "text-generation", "priority": "medium"},
        {"name": "stabilityai/stablelm-3b-4e1t", "task": "text-generation", "priority": "low"},
        
        # Vision Models
        {"name": "google/vit-base-patch16-224", "task": "image-classification", "priority": "high"},
        {"name": "facebook/detr-resnet-50", "task": "object-detection", "priority": "high"},
        {"name": "facebook/sam-vit-base", "task": "image-segmentation", "priority": "high"},
        {"name": "facebook/deit-base-patch16-224", "task": "image-classification", "priority": "medium"},
        {"name": "microsoft/swin-base-patch4-window7-224", "task": "image-classification", "priority": "medium"},
        {"name": "microsoft/swinv2-base-patch4-window8-256", "task": "image-classification", "priority": "medium"},
        {"name": "facebook/convnext-base-224", "task": "image-classification", "priority": "medium"},
        {"name": "facebook/regnet-y-080", "task": "image-classification", "priority": "low"},
        {"name": "facebook/dinov2-base", "task": "feature-extraction", "priority": "medium"},
        {"name": "facebook/vit-mae-base", "task": "feature-extraction", "priority": "medium"},
        
        # Audio Models
        {"name": "openai/whisper-tiny", "task": "automatic-speech-recognition", "priority": "high"},
        {"name": "facebook/wav2vec2-base-960h", "task": "audio-classification", "priority": "high"},
        {"name": "facebook/hubert-base-ls960", "task": "automatic-speech-recognition", "priority": "medium"},
        {"name": "facebook/data2vec-audio-base", "task": "automatic-speech-recognition", "priority": "medium"},
        {"name": "facebook/musicgen-small", "task": "text-to-audio", "priority": "medium"},
        {"name": "laion/clap-htsat-fused", "task": "audio-classification", "priority": "medium"},
        
        # Multimodal Models
        {"name": "Salesforce/blip-image-captioning-base", "task": "image-to-text", "priority": "high"},
        {"name": "Salesforce/blip2-opt-2.7b", "task": "image-to-text", "priority": "medium"},
        {"name": "microsoft/layoutlm-base-uncased", "task": "document-question-answering", "priority": "high"},
        {"name": "microsoft/layoutlmv2-base-uncased", "task": "document-question-answering", "priority": "medium"},
        {"name": "microsoft/layoutlmv3-base", "task": "document-question-answering", "priority": "medium"},
        {"name": "llava-hf/llava-1.5-7b-hf", "task": "image-to-text", "priority": "high"},
        {"name": "llava-hf/llava-1.6-mistral-7b-hf", "task": "image-to-text", "priority": "medium"},
        {"name": "microsoft/florence-2-base", "task": "image-to-text", "priority": "medium"},
        {"name": "facebook/nougat-base", "task": "document-question-answering", "priority": "medium"},
        {"name": "facebook/imagebind-huge", "task": "image-to-text", "priority": "low"},
        {"name": "HuggingFaceM4/idefics2-8b", "task": "image-to-text", "priority": "medium"},
        {"name": "facebook/xclip-base-patch32", "task": "video-classification", "priority": "medium"},
        {"name": "facebook/videomae-base", "task": "video-classification", "priority": "medium"}
    ]

def generate_models_batch(models_to_generate=None, concurrency=1, priority_filter=None):
    """Generate test files for a batch of models with optional filtering and parallelization"""
    
    if models_to_generate is None:
        # Use the default comprehensive list
        models_to_generate = generate_comprehensive_model_list()
    
    # Filter by priority if specified
    if priority_filter:
        models_to_generate = [m for m in models_to_generate if m.get("priority", "medium") in priority_filter]
    
    # Track dependencies for each model
    dependency_dict = {}
    
    generated_files = []
    failed_models = []
    
    # Process models sequentially or in parallel based on concurrency
    if concurrency > 1 and len(models_to_generate) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # First analyze dependencies for all models
            logger.info(f"Analyzing dependencies for {len(models_to_generate)} models with {concurrency} workers...")
            
            def analyze_model(model):
                try:
                    normalized_name = normalize_model_name(model["name"])
                    deps_info = get_model_dependencies(model["name"])
                    return model, normalized_name, deps_info
                except Exception as e:
                    logger.error(f"Error analyzing {model['name']}: {e}")
                    return model, None, None
            
            # Submit all dependency analysis tasks
            dep_futures = {executor.submit(analyze_model, model): model for model in models_to_generate}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(dep_futures):
                model, normalized_name, deps_info = future.result()
                if normalized_name and deps_info:
                    dependency_dict[model["name"]] = deps_info
                    logger.info(f"Completed dependency analysis for {model['name']}")
            
            # Now generate the test files
            logger.info(f"Generating test files for {len(dependency_dict)} models...")
            
            def generate_model_test(model):
                try:
                    normalized_name = normalize_model_name(model["name"])
                    deps_info = dependency_dict[model["name"]]
                    
                    output_file = generate_test_file(
                        model["name"], 
                        normalized_name, 
                        model["task"], 
                        SKILLS_DIR, 
                        deps_info
                    )
                    
                    return model["name"], output_file, deps_info, None
                except Exception as e:
                    logger.error(f"Error generating test for {model['name']}: {e}")
                    return model["name"], None, None, str(e)
            
            # Submit all test generation tasks
            file_futures = {executor.submit(generate_model_test, model): model 
                           for model in models_to_generate if model["name"] in dependency_dict}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(file_futures):
                model_name, output_file, deps_info, error = future.result()
                if output_file and deps_info:
                    generated_files.append((model_name, output_file, deps_info))
                    logger.info(f"Generated test file: {output_file}")
                else:
                    failed_models.append((model_name, error))
                    logger.error(f"Failed to generate test for {model_name}: {error}")
    else:
        # Sequential processing
        for model in models_to_generate:
            try:
                normalized_name = normalize_model_name(model["name"])
                logger.info(f"Analyzing dependencies for {model['name']}")
                
                # Get model dependencies
                deps_info = get_model_dependencies(model["name"])
                dependency_dict[model["name"]] = deps_info
                
                # Generate test file with dependency info
                output_file = generate_test_file(
                    model["name"], 
                    normalized_name, 
                    model["task"], 
                    SKILLS_DIR, 
                    deps_info
                )
                
                generated_files.append((model["name"], output_file, deps_info))
                logger.info(f"Generated test file: {output_file}")
                logger.info(f"  Dependencies: {', '.join(deps_info['dependencies']) if deps_info['dependencies'] else 'None'}")
                logger.info(f"  Use remote code: {deps_info['use_remote_code']}")
            except Exception as e:
                failed_models.append((model["name"], str(e)))
                logger.error(f"Error generating test for {model['name']}: {e}")
    
    # Update dependency tracking files
    update_dependencies_files(dependency_dict)
    
    # Generate summary
    num_successful = len(generated_files)
    num_failed = len(failed_models)
    total = num_successful + num_failed
    
    print("\nSummary:")
    print(f"Generated {num_successful}/{total} test files ({num_successful/total*100:.1f}% success rate)")
    
    # Group by category/task
    models_by_task = {}
    for model, file, deps in generated_files:
        task = next((m["task"] for m in models_to_generate if m["name"] == model), "unknown")
        remote_code = " (remote code)" if deps["use_remote_code"] else ""
        models_by_task.setdefault(task, []).append(f"{model}{remote_code}")
    
    # Print by task
    for task, models in sorted(models_by_task.items()):
        print(f"\n{task.upper()} ({len(models)}):")
        for model in models:
            print(f"  - {model}")
    
    # Print failures if any
    if failed_models:
        print("\nFailed models:")
        for model, error in failed_models:
            print(f"  - {model}: {error}")
            
    # Update test progress tracking
    update_test_progress(generated_files, models_to_generate, failed_models)
    
    return generated_files, failed_models

def update_test_progress(generated_files, all_models, failed_models):
    """Update the test progress tracking files with current state."""
    try:
        import datetime
        
        # Try to load existing progress data if available
        existing_data = {}
        if os.path.exists(TEST_SUMMARY_JSON):
            try:
                with open(TEST_SUMMARY_JSON, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not load existing test summary from {TEST_SUMMARY_JSON}")
        
        # Load existing progress data
        progress_data = {
            "total_models": len(all_models),
            "successful_models": len(generated_files),
            "failed_models": len(failed_models),
            "timestamp": datetime.datetime.now().isoformat(),
            "coverage_percentage": len(generated_files) / len(all_models) * 100 if all_models else 0,
            "by_task": {},
            "by_priority": {},
            "total_available_models": len(generate_comprehensive_model_list()),
            "overall_percentage": 0  # Will be calculated later
        }
        
        # Group by task
        for model in all_models:
            task = model["task"]
            priority = model.get("priority", "medium")
            
            # Initialize if not exists
            if task not in progress_data["by_task"]:
                progress_data["by_task"][task] = {"total": 0, "success": 0, "failed": 0}
            if priority not in progress_data["by_priority"]:
                progress_data["by_priority"][priority] = {"total": 0, "success": 0, "failed": 0}
            
            # Increment total
            progress_data["by_task"][task]["total"] += 1
            progress_data["by_priority"][priority]["total"] += 1
            
            # Check if generated successfully
            if any(gen_model == model["name"] for gen_model, _, _ in generated_files):
                progress_data["by_task"][task]["success"] += 1
                progress_data["by_priority"][priority]["success"] += 1
            elif any(failed_model == model["name"] for failed_model, _ in failed_models):
                progress_data["by_task"][task]["failed"] += 1
                progress_data["by_priority"][priority]["failed"] += 1
        
        # Calculate overall test coverage percentage across all available models
        total_available = progress_data["total_available_models"]
        if total_available > 0:
            progress_data["overall_percentage"] = (len(generated_files) / total_available) * 100
            
        # Count models requiring remote code
        remote_code_models = [m for m, _, info in generated_files if info.get("use_remote_code", False)]
        progress_data["remote_code_models"] = len(remote_code_models)
        progress_data["remote_code_percentage"] = (len(remote_code_models) / len(generated_files)) * 100 if generated_files else 0
        
        # Add dependency statistics
        all_dependencies = set()
        for _, _, info in generated_files:
            all_dependencies.update(info.get("dependencies", []))
        
        progress_data["unique_dependencies"] = len(all_dependencies)
        progress_data["dependencies_list"] = sorted(list(all_dependencies))
        
        # Add installation stats - which dependencies are most common
        dependency_counts = {}
        for _, _, info in generated_files:
            for dep in info.get("dependencies", []):
                base_dep = dep.split(">=")[0].split(">")[0].strip()
                dependency_counts[base_dep] = dependency_counts.get(base_dep, 0) + 1
                
        progress_data["dependency_counts"] = dependency_counts
        
        # Save to JSON
        with open(TEST_SUMMARY_JSON, "w") as f:
            json.dump(progress_data, f, indent=2)
        
        # Generate markdown report
        with open(MODEL_TESTING_PROGRESS_MD, "w") as f:
            f.write("# Hugging Face Model Testing Progress\n\n")
            f.write(f"*Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Add summary cards for key metrics
            f.write("## Summary\n\n")
            f.write("<div style='display: flex; flex-wrap: wrap; gap: 10px;'>\n\n")
            
            # Test coverage card
            f.write("<div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; min-width: 200px; flex: 1;'>\n")
            f.write(f"<h3>Test Coverage</h3>\n")
            f.write(f"<p><b>{progress_data['successful_models']}</b> of {progress_data['total_available_models']} models</p>\n")
            f.write(f"<p><b>{progress_data['overall_percentage']:.1f}%</b> coverage</p>\n")
            f.write("</div>\n\n")
            
            # Remote code card
            f.write("<div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; min-width: 200px; flex: 1;'>\n")
            f.write(f"<h3>Remote Code Models</h3>\n")
            f.write(f"<p><b>{progress_data.get('remote_code_models', 0)}</b> models requiring remote code</p>\n")
            f.write(f"<p><b>{progress_data.get('remote_code_percentage', 0):.1f}%</b> of tested models</p>\n")
            f.write("</div>\n\n")
            
            # Dependencies card
            f.write("<div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; min-width: 200px; flex: 1;'>\n")
            f.write(f"<h3>Dependencies</h3>\n")
            f.write(f"<p><b>{progress_data.get('unique_dependencies', 0)}</b> unique dependencies</p>\n")
            
            # Add top 3 dependencies if available
            top_deps = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_deps:
                f.write("<p><b>Top dependencies:</b></p>\n<ul>\n")
                for dep, count in top_deps:
                    f.write(f"<li>{dep}: {count} models</li>\n")
                f.write("</ul>\n")
            
            f.write("</div>\n\n")
            
            f.write("</div>\n\n")
            
            # Overall progress
            f.write("## Overall Progress\n\n")
            f.write(f"- **Total Models**: {progress_data['total_models']}\n")
            f.write(f"- **Successfully Tested**: {progress_data['successful_models']} ({progress_data['coverage_percentage']:.1f}%)\n")
            f.write(f"- **Failed Models**: {progress_data['failed_models']}\n\n")
            
            # By priority
            f.write("## Coverage by Priority\n\n")
            f.write("| Priority | Total | Tested | Coverage |\n")
            f.write("|----------|-------|--------|----------|\n")
            
            for priority in ["high", "medium", "low"]:
                if priority in progress_data["by_priority"]:
                    data = progress_data["by_priority"][priority]
                    coverage = data["success"] / data["total"] * 100 if data["total"] > 0 else 0
                    f.write(f"| {priority.capitalize()} | {data['total']} | {data['success']} | {coverage:.1f}% |\n")
            
            # By task
            f.write("\n## Coverage by Task\n\n")
            f.write("| Task | Total | Tested | Coverage |\n")
            f.write("|------|-------|--------|----------|\n")
            
            for task, data in sorted(progress_data["by_task"].items()):
                coverage = data["success"] / data["total"] * 100 if data["total"] > 0 else 0
                f.write(f"| {task} | {data['total']} | {data['success']} | {coverage:.1f}% |\n")
            
            # Recently tested models
            f.write("\n## Recently Tested Models\n\n")
            for model, file, deps in generated_files[-10:]:  # Show last 10
                remote_code = " (requires remote code)" if deps["use_remote_code"] else ""
                dep_list = ", ".join(deps["dependencies"]) if deps["dependencies"] else "None"
                f.write(f"- **{model}**{remote_code}\n")
                f.write(f"  - Dependencies: {dep_list}\n")
            
            # Failed models
            if failed_models:
                f.write("\n## Failed Models\n\n")
                for model, error in failed_models:
                    f.write(f"- **{model}**: {error}\n")
        
        logger.info(f"Updated test progress tracking in {MODEL_TESTING_PROGRESS_MD}")
    except Exception as e:
        logger.error(f"Error updating test progress: {e}")

def main():
    """Parse command line arguments and run the appropriate action."""
    # Create argument parser
    import argparse
    parser = argparse.ArgumentParser(description="Generate Hugging Face model test files")
    
    # Add arguments for single model generation
    parser.add_argument("--model", type=str, help="Generate test for a specific model")
    parser.add_argument("--task", type=str, help="Task type for the model (required if --model is specified)")
    
    # Add arguments for batch processing
    parser.add_argument("--batch", action="store_true", help="Generate batch of model tests")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent model generations for batch mode")
    parser.add_argument("--priority", type=str, choices=["high", "medium", "low", "all"], default="all", 
                      help="Filter models by priority (high, medium, low, or all)")
    
    # Add arguments for filtering
    parser.add_argument("--category", type=str, choices=["language", "vision", "audio", "multimodal"],
                      help="Filter models by category")
    parser.add_argument("--remote-code-only", action="store_true", 
                      help="Only generate tests for models requiring remote code")
    
    # Add arguments for listing and information
    parser.add_argument("--list-models", action="store_true", help="List all available models without generating tests")
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks")
    parser.add_argument("--list-dependencies", action="store_true", 
                      help="List all dependencies for selected models")
    
    # Add arguments for output control
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for test files")
    parser.add_argument("--save-progress", action="store_true", help="Save test progress to markdown file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing test files")
    
    # Add arguments for updating existing tests
    parser.add_argument("--update-existing", action="store_true", 
                      help="Update existing test files with new features")
    parser.add_argument("--add-mocks", action="store_true", 
                      help="Add comprehensive mocks to existing test files")
    parser.add_argument("--update-dependencies", action="store_true",
                      help="Update dependency tracking in existing test files")
                      
    # Add arguments for template validation
    parser.add_argument("--validate", action="store_true",
                      help="Validate templates before generation (default if validator available)")
    parser.add_argument("--skip-validation", action="store_true",
                      help="Skip template validation even if validator is available")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else SKILLS_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # List tasks if requested
    if args.list_tasks:
        all_tasks = set()
        for model in generate_comprehensive_model_list():
            all_tasks.add(model["task"])
        
        print("\nAvailable tasks:")
        for task in sorted(all_tasks):
            print(f"  - {task}")
        return
    
    # List all models if requested
    if args.list_models:
        all_models = generate_comprehensive_model_list()
        
        # Apply category filter if specified
        if args.category:
            filtered_models = []
            for model in all_models:
                model_name = model["name"]
                task = model["task"]
                category = get_category_from_task(task)
                if category == args.category:
                    filtered_models.append(model)
            all_models = filtered_models
            
        # Apply priority filter if not "all"
        if args.priority != "all":
            all_models = [m for m in all_models if m.get("priority", "medium") == args.priority]
            
        # Group by task
        models_by_task = {}
        for model in all_models:
            task = model["task"]
            priority = model.get("priority", "medium")
            models_by_task.setdefault(task, []).append((model["name"], priority))
            
        print(f"\nAvailable models ({len(all_models)} total):")
        for task, models in sorted(models_by_task.items()):
            print(f"\n{task.upper()} ({len(models)}):")
            for model_name, priority in sorted(models):
                print(f"  - {model_name} [Priority: {priority}]")
        return
    
    # Generate batch of models
    if args.batch:
        # Get all models
        all_models = generate_comprehensive_model_list()
        
        # Apply priority filter if not "all"
        priority_filter = None if args.priority == "all" else [args.priority]
        
        # Apply category filter if specified
        if args.category:
            filtered_models = []
            for model in all_models:
                task = model["task"]
                category = get_category_from_task(task)
                if category == args.category:
                    filtered_models.append(model)
            all_models = filtered_models
        
        # Generate models
        generate_models_batch(
            models_to_generate=all_models,
            concurrency=args.concurrency,
            priority_filter=priority_filter
        )
        return
    
    # Generate single model
    if args.model:
        if not args.task:
            print("Error: --task is required when --model is specified")
            return
        
        model_name = args.model
        normalized_name = normalize_model_name(model_name)
        task_type = args.task
        
        print(f"Generating test file for {model_name} ({task_type})...")
        output_file = generate_test_file(model_name, normalized_name, task_type, output_dir)
        print(f"Generated test file: {output_file}")
        return
    
    # Default: generate a single test for a common model
    model_name = "bert-base-uncased"
    normalized_name = normalize_model_name(model_name)
    task_type = "fill-mask"
    
    print(f"No specific model requested, using default model {model_name}")
    output_file = generate_test_file(model_name, normalized_name, task_type, output_dir)
    print(f"Generated test file: {output_file}")
    print("\nTip: Run with --help to see all available options")
    print("     Try --batch to generate tests for multiple models")
    print("     Use --list-models to see all available models")

if __name__ == "__main__":
    main()