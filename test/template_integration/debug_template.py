#!/usr/bin/env python3
"""Debugging script for template issues."""

import os
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEMP_FILE = "temp_test.py"

# Templates dictionary with model type as key
TEMPLATES = {
    "vision": """#!/usr/bin/env python3
\"\"\"
Test file for ViT models using the refactored test suite structure.
\"\"\"

import os
import sys
import json
import time
import logging
import torch
import numpy as np
from pathlib import Path
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestVitModel(ModelTest):
    \"\"\"Test class for vision transformer models.\"\"\"
    
    def setUp(self):
        \"\"\"Set up the test environment.\"\"\"
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "MODEL_ID"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define model parameters
        self.task = "image-classification"
        
        # Define test inputs
        self.test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    def tearDown(self):
        \"\"\"Clean up resources after the test.\"\"\"
        super().tearDown()
    
    def load_model(self):
        \"\"\"Load the model for testing.\"\"\"
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Load processor and model
            processor = AutoImageProcessor.from_pretrained(self.model_id)
            model = AutoModelForImageClassification.from_pretrained(self.model_id)
            model = model.to(self.device)
            
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def test_model_loading(self):
        \"\"\"Test that the model loads correctly.\"\"\"
        model_components = self.load_model()
        
        # Verify model and processor
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["processor"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        \"\"\"Test basic inference with the model.\"\"\"
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        processor = model_components["processor"]
        
        # Create dummy image for testing
        from PIL import Image
        dummy_image = Image.new('RGB', (224, 224), color='white')
        
        # Process image
        inputs = processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertTrue(hasattr(outputs, "logits"))
        logger.info(f"Inference successful: {outputs.logits.shape}")
    
    def test_hardware_compatibility(self):
        \"\"\"Test model compatibility across hardware platforms.\"\"\"
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
        
        results = {}
        original_device = self.device
        
        for device in available_devices:
            try:
                self.device = device
                model_components = self.load_model()
                model = model_components["model"]
                
                # Basic verification
                self.assertIsNotNone(model)
                results[device] = True
                logger.info(f"Model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed on {device}: {e}")
                results[device] = False
            finally:
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()))
""",
    
    "bert": """#!/usr/bin/env python3
\"\"\"
Test file for BERT models using the refactored test suite structure.
\"\"\"

import os
import sys
import json
import time
import logging
import torch
import numpy as np
from pathlib import Path
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestBertModel(ModelTest):
    \"\"\"Test class for BERT encoder-only models.\"\"\"
    
    def setUp(self):
        \"\"\"Set up the test environment.\"\"\"
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "MODEL_ID"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define model parameters
        self.task = "fill-mask"
        
        # Define test inputs
        self.test_text = "The quick brown fox jumps over the [MASK] dog."
    
    def tearDown(self):
        \"\"\"Clean up resources after the test.\"\"\"
        super().tearDown()
    
    def load_model(self):
        \"\"\"Load the model for testing.\"\"\"
        try:
            from transformers import AutoTokenizer, BertForMaskedLM
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = BertForMaskedLM.from_pretrained(self.model_id)
            model = model.to(self.device)
            
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def test_model_loading(self):
        \"\"\"Test that the model loads correctly.\"\"\"
        model_components = self.load_model()
        
        # Verify model and tokenizer
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["tokenizer"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        \"\"\"Test basic inference with the model.\"\"\"
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        
        # Prepare input
        inputs = tokenizer(self.test_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertTrue(hasattr(outputs, "logits"))
        
        # Get the mask token prediction
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        top_tokens_words = [tokenizer.decode([token]).strip() for token in top_tokens]
        
        logger.info(f"Top predictions: {', '.join(top_tokens_words)}")
        logger.info(f"Inference successful: {outputs.logits.shape}")
    
    def test_hardware_compatibility(self):
        \"\"\"Test model compatibility across hardware platforms.\"\"\"
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
        
        results = {}
        original_device = self.device
        
        for device in available_devices:
            try:
                self.device = device
                model_components = self.load_model()
                model = model_components["model"]
                
                # Basic verification
                self.assertIsNotNone(model)
                results[device] = True
                logger.info(f"Model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed on {device}: {e}")
                results[device] = False
            finally:
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()))
""",
    
    "gpt": """#!/usr/bin/env python3
\"\"\"
Test file for GPT models using the refactored test suite structure.
\"\"\"

import os
import sys
import json
import time
import logging
import torch
import numpy as np
from pathlib import Path
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestGptModel(ModelTest):
    \"\"\"Test class for GPT decoder-only models.\"\"\"
    
    def setUp(self):
        \"\"\"Set up the test environment.\"\"\"
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "MODEL_ID"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define model parameters
        self.task = "text-generation"
        self.max_new_tokens = 20
        
        # Define test inputs
        self.test_text = "Once upon a time in a galaxy far, far away,"
    
    def tearDown(self):
        \"\"\"Clean up resources after the test.\"\"\"
        super().tearDown()
    
    def load_model(self):
        \"\"\"Load the model for testing.\"\"\"
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
            model = model.to(self.device)
            
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def test_model_loading(self):
        \"\"\"Test that the model loads correctly.\"\"\"
        model_components = self.load_model()
        
        # Verify model and tokenizer
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["tokenizer"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        \"\"\"Test basic inference with the model.\"\"\"
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        
        # Prepare input
        inputs = tokenizer(self.test_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check that the output contains the input and has been extended
        self.assertTrue(self.test_text in generated_text)
        self.assertGreater(len(generated_text), len(self.test_text))
        
        logger.info(f"Generated text: {generated_text}")
        logger.info("Inference successful")
    
    def test_hardware_compatibility(self):
        \"\"\"Test model compatibility across hardware platforms.\"\"\"
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
        
        results = {}
        original_device = self.device
        
        for device in available_devices:
            try:
                self.device = device
                model_components = self.load_model()
                model = model_components["model"]
                
                # Basic verification
                self.assertIsNotNone(model)
                results[device] = True
                logger.info(f"Model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed on {device}: {e}")
                results[device] = False
            finally:
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()))
""",

    "speech": """#!/usr/bin/env python3
\"\"\"
Test file for speech models using the refactored test suite structure.
\"\"\"

import os
import sys
import json
import time
import logging
import torch
import numpy as np
from pathlib import Path
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSpeechModel(ModelTest):
    \"\"\"Test class for speech/audio models like Whisper, Wav2Vec2, etc.\"\"\"
    
    def setUp(self):
        \"\"\"Set up the test environment.\"\"\"
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "MODEL_ID"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define model parameters
        self.task = "automatic-speech-recognition"
        self.audio_sampling_rate = 16000
        
        # Define test audio path
        self.test_audio_path = "test_audio.wav"
    
    def tearDown(self):
        \"\"\"Clean up resources after the test.\"\"\"
        super().tearDown()
    
    def create_test_audio(self):
        \"\"\"Create a test audio file if it doesn't exist.\"\"\"
        if not os.path.exists(self.test_audio_path):
            try:
                # Generate a simple sine wave
                import scipy.io.wavfile as wav
                sample_rate = self.audio_sampling_rate
                duration = 3  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
                wav.write(self.test_audio_path, sample_rate, audio.astype(np.float32))
                return True
            except Exception as e:
                logger.error(f"Error creating test audio: {e}")
                return False
        return True
    
    def load_audio(self):
        \"\"\"Load audio data from file.\"\"\"
        # Ensure test audio exists
        self.create_test_audio()
        
        try:
            # Try to use soundfile
            import soundfile as sf
            audio, sample_rate = sf.read(self.test_audio_path)
        except ImportError:
            # Fallback to scipy
            import scipy.io.wavfile as wav
            sample_rate, audio = wav.read(self.test_audio_path)
            # Convert to float if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        
        return audio, sample_rate
    
    def load_model(self):
        \"\"\"Load the model for testing.\"\"\"
        try:
            if "whisper" in self.model_id.lower():
                # For Whisper models
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                processor = WhisperProcessor.from_pretrained(self.model_id)
                model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
            elif "wav2vec2" in self.model_id.lower():
                # For Wav2Vec2 models
                from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
                
                processor = Wav2Vec2Processor.from_pretrained(self.model_id)
                model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            else:
                # For other speech models
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                
                processor = AutoProcessor.from_pretrained(self.model_id)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
            
            # Move to device
            model = model.to(self.device)
            
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def test_model_loading(self):
        \"\"\"Test that the model loads correctly.\"\"\"
        model_components = self.load_model()
        
        # Verify model and processor
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["processor"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        \"\"\"Test basic inference with the model.\"\"\"
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        processor = model_components["processor"]
        
        # Load audio
        audio, sample_rate = self.load_audio()
        
        # Process audio
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Check output shape
        if hasattr(outputs, "logits"):
            logger.info(f"Output shape: {outputs.logits.shape}")
        
        logger.info("Basic inference successful")
    
    def test_transcription(self):
        \"\"\"Test transcription with the model.\"\"\"
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        processor = model_components["processor"]
        
        # Load audio
        audio, sample_rate = self.load_audio()
        
        # Process audio
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Model-specific transcription
        if "whisper" in self.model_id.lower():
            # Whisper model
            with torch.no_grad():
                generated_ids = model.generate(inputs["input_features"], max_length=100)
                
            # Decode the output
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # CTC-based models like Wav2Vec2
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Get the predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode the output
            transcription = processor.batch_decode(predicted_ids)[0]
        
        logger.info(f"Transcription: {transcription}")
        logger.info("Transcription successful")
    
    def test_hardware_compatibility(self):
        \"\"\"Test model compatibility across hardware platforms.\"\"\"
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
        
        results = {}
        original_device = self.device
        
        for device in available_devices:
            try:
                self.device = device
                model_components = self.load_model()
                model = model_components["model"]
                
                # Basic verification
                self.assertIsNotNone(model)
                results[device] = True
                logger.info(f"Model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed on {device}: {e}")
                results[device] = False
            finally:
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()))
"""
}

# Model configurations
MODEL_CONFIGS = {
    "vision": {
        "model_id": "google/vit-base-patch16-224",
        "output_dir": "vision",
        "output_file": "test_vit_base_patch16_224.py"
    },
    "bert": {
        "model_id": "bert-base-uncased",
        "output_dir": "text",
        "output_file": "test_bert_base_uncased.py"
    },
    "gpt": {
        "model_id": "gpt2",
        "output_dir": "text",
        "output_file": "test_gpt2.py"
    },
    "speech": {
        "model_id": "openai/whisper-tiny",
        "output_dir": "audio",
        "output_file": "test_whisper_tiny.py"
    }
}

def create_test_file(model_type="vision", model_id=None):
    """Create a test file using the template."""
    # Validate model type
    if model_type not in TEMPLATES:
        logger.error(f"Invalid model type: {model_type}. Available types: {', '.join(TEMPLATES.keys())}")
        return False
        
    # Get config
    config = MODEL_CONFIGS[model_type]
    
    # Override model ID if provided
    if model_id:
        config["model_id"] = model_id
        # Update output file name
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        config["output_file"] = f"test_{model_name.replace('-', '_')}.py"
    
    # Get template content
    content = TEMPLATES[model_type]
    
    # Replace MODEL_ID with actual model ID
    content = content.replace("MODEL_ID", config["model_id"])
    
    # Write to temp file
    with open(TEMP_FILE, "w") as f:
        f.write(content)
    
    logger.info(f"Created temporary test file: {TEMP_FILE}")
    
    # Validate syntax
    try:
        with open(TEMP_FILE, "r") as f:
            source = f.read()
        
        # Compile to check syntax
        compile(source, TEMP_FILE, "exec")
        logger.info("Syntax check passed")
        
        # Create final path
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "refactored_test_suite", "models", config["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, config["output_file"])
        
        # Create a backup if file exists
        if os.path.exists(output_path):
            import datetime
            backup_path = f"{output_path}.bak.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                import shutil
                shutil.copy2(output_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # Copy to final location
        with open(output_path, "w") as f:
            f.write(content)
        
        logger.info(f"Created validated test file: {output_path}")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error: {e}")
        logger.error(f"Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Debug template generation for refactored tests")
    parser.add_argument("--model-type", type=str, choices=TEMPLATES.keys(), default="vision", 
                        help="Type of model to generate test for (vision, bert, gpt, speech)")
    parser.add_argument("--model-id", type=str, help="Model ID to use (overrides default)")
    
    args = parser.parse_args()
    
    # Generate test file
    create_test_file(args.model_type, args.model_id)

if __name__ == "__main__":
    main()