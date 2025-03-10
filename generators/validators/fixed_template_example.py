#!/usr/bin/env python3
"""
Fixed Template Example

This file demonstrates a properly formatted template that passes validation.
"""

STANDARD_TEMPLATE = '''#!/usr/bin/env python3
"""
Test file for {{model_name}} model.

This file is auto-generated using the template-based test generator.
Generated: {{generation_date}}
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{{model_class_name}}:
    """Test class for {{model_name}} model."""
    
    def __init__(self):
        """Initialize the test with model details and hardware detection."""
        self.model_name = "{{model_name}}"
        self.model_type = "{{model_type}}"
        self.setup_hardware()
    
    def setup_hardware(self):
        """Set up hardware detection for the template."""
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        # MPS support (Apple Silicon)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        # OpenVINO support
        self.has_openvino = 'openvino' in sys.modules
        # Qualcomm AI Engine support
        self.has_qualcomm = 'qti' in sys.modules or 'qnn_wrapper' in sys.modules
        # WebNN/WebGPU support
        self.has_webnn = False  # Will be set by WebNN bridge if available
        self.has_webgpu = False  # Will be set by WebGPU bridge if available
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        logger.info(f"Using device: {self.device}")
        
    def get_model(self):
        """Load model from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Get tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get model
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(self.device)
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    
    {{custom_model_loading}}
    
    def test_basic_inference(self):
        """Run a basic inference test with the model."""
        model, tokenizer = self.get_model()
        
        if model is None or tokenizer is None:
            logger.error("Failed to load model or tokenizer")
            return False
        
        try:
            # Prepare input
            {{model_input_code}}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Check outputs
            {{output_check_code}}
            
            logger.info("Basic inference test passed")
            return True
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return False
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware platforms."""
        devices_to_test = []
        
        if self.has_cuda:
            devices_to_test.append('cuda')
        if self.has_mps:
            devices_to_test.append('mps')
        if self.has_rocm:
            devices_to_test.append('cuda')  # ROCm uses CUDA compatibility layer
        if self.has_openvino:
            devices_to_test.append('openvino')
        if self.has_qualcomm:
            devices_to_test.append('qualcomm')
        
        # Always test CPU
        if 'cpu' not in devices_to_test:
            devices_to_test.append('cpu')
        
        results = {}
        
        for device in devices_to_test:
            try:
                logger.info(f"Testing on {device}...")
                original_device = self.device
                self.device = device
                
                # Run a simple test
                success = self.test_basic_inference()
                results[device] = success
                
                # Restore original device
                self.device = original_device
            except Exception as e:
                logger.error(f"Error testing on {device}: {e}")
                results[device] = False
        
        return results
    
    def run(self):
        """Run all tests."""
        logger.info(f"Testing {self.model_name} on {self.device}")
        
        # Run basic inference test
        basic_result = self.test_basic_inference()
        
        # Run hardware compatibility test
        hw_results = self.test_hardware_compatibility()
        
        # Summarize results
        logger.info("Test Results:")
        logger.info(f"- Basic inference: {'PASS' if basic_result else 'FAIL'}")
        logger.info("- Hardware compatibility:")
        for device, result in hw_results.items():
            logger.info(f"  - {device}: {'PASS' if result else 'FAIL'}")
        
        return basic_result and all(hw_results.values())


{{model_specific_code}}


if __name__ == "__main__":
    # Create and run the test
    test = Test{{model_class_name}}()
    test.run()
'''

# Model type specific templates with proper indentation

MODEL_INPUT_TEMPLATES = {
    "text_embedding": '''            # Prepare text input
            text = "This is a sample text for testing the {{model_name}} model."
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}''',
    
    "vision": '''            # Prepare image input
            from PIL import Image
            import requests
            from io import BytesIO
            from transformers import AutoImageProcessor
            
            # Create a test image if none exists
            test_image_path = "test_image.jpg"
            if not os.path.exists(test_image_path):
                # Create a simple test image (black and white gradient)
                import numpy as np
                from PIL import Image
                size = 224
                img_array = np.zeros((size, size, 3), dtype=np.uint8)
                for i in range(size):
                    for j in range(size):
                        img_array[i, j, :] = (i + j) % 256
                img = Image.fromarray(img_array)
                img.save(test_image_path)
            
            # Load the image
            image = Image.open(test_image_path)
            
            # Get image processor
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}''',
    
    "audio": '''            # Prepare audio input
            import torch
            import numpy as np
            from transformers import AutoFeatureExtractor
            
            # Create a test audio if none exists
            test_audio_path = "test_audio.wav"
            if not os.path.exists(test_audio_path):
                # Generate a simple sine wave
                import scipy.io.wavfile as wav
                sample_rate = 16000
                duration = 3  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
                wav.write(test_audio_path, sample_rate, audio.astype(np.float32))
            
            # Load audio file
            sample_rate = 16000
            audio = np.zeros(sample_rate * 3)  # 3 seconds of silence as fallback
            try:
                import soundfile as sf
                audio, sample_rate = sf.read(test_audio_path)
            except:
                logger.warning("Could not load audio, using zeros array")
            
            # Get feature extractor
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}''',
    
    "multimodal": '''            # Prepare multimodal input (text and image)
            from PIL import Image
            from transformers import AutoProcessor
            
            # Create a test image if none exists
            test_image_path = "test_image.jpg"
            if not os.path.exists(test_image_path):
                # Create a simple test image
                import numpy as np
                from PIL import Image
                size = 224
                img_array = np.zeros((size, size, 3), dtype=np.uint8)
                for i in range(size):
                    for j in range(size):
                        img_array[i, j, :] = (i + j) % 256
                img = Image.fromarray(img_array)
                img.save(test_image_path)
            
            # Load the image
            image = Image.open(test_image_path)
            
            # Prepare text
            text = "What's in this image?"
            
            # Get processor
            processor = AutoProcessor.from_pretrained(self.model_name)
            inputs = processor(text=text, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}'''
}

# Model specific code for different model types
MODEL_SPECIFIC_CODE_TEMPLATES = {
    "text_embedding": '''# Additional methods for text embedding models
def test_embedding_similarity(self):
    """Test embedding similarity functionality."""
    model, tokenizer = self.get_model()
    
    if model is None or tokenizer is None:
        logger.error("Failed to load model or tokenizer")
        return False
    
    try:
        # Prepare input texts
        texts = [
            "This is a sample text for testing embeddings.",
            "Another example text that is somewhat similar.",
            "This text is completely different from the others."
        ]
        
        # Get embeddings
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Use mean pooling to get sentence embedding
            embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embedding)
        
        # Calculate similarities
        import torch.nn.functional as F
        
        sim_0_1 = F.cosine_similarity(embeddings[0], embeddings[1])
        sim_0_2 = F.cosine_similarity(embeddings[0], embeddings[2])
        
        logger.info(f"Similarity between text 0 and 1: {sim_0_1.item():.4f}")
        logger.info(f"Similarity between text 0 and 2: {sim_0_2.item():.4f}")
        
        # First two should be more similar than first and third
        assert sim_0_1 > sim_0_2, "Expected similarity between similar texts to be higher"
        
        return True
    except Exception as e:
        logger.error(f"Error during embedding similarity test: {e}")
        return False''',
    
    "text_generation": '''# Additional methods for text generation models
def test_text_generation(self):
    """Test text generation functionality."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        # Use the specialized model class for generation
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = model.to(self.device)
        
        # Prepare input
        prompt = "Once upon a time, there was a"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        
        logger.info(f"Generated text: {generated_text}")
        
        # Basic validation
        assert len(generated_text) > len(prompt), "Generated text should be longer than prompt"
        
        return True
    except Exception as e:
        logger.error(f"Error during text generation test: {e}")
        return False''',
    
    "vision": '''# Additional methods for vision models
def test_image_classification(self):
    """Test image classification functionality."""
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    
    try:
        # Create a test image if none exists
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            # Create a simple test image
            import numpy as np
            from PIL import Image
            size = 224
            img_array = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    img_array[i, j, :] = (i + j) % 256
            img = Image.fromarray(img_array)
            img.save(test_image_path)
            
        # Load specialized model and processor
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = AutoModelForImageClassification.from_pretrained(self.model_name)
        except:
            # Fallback to general model
            from transformers import AutoModel, AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            
        model = model.to(self.device)
        
        # Load and process the image
        from PIL import Image
        image = Image.open(test_image_path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Check outputs
        assert outputs is not None, "Outputs should not be None"
        
        # If it's a classification model, try to get class probabilities
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            logger.info(f"Top probability: {probabilities.max().item():.4f}")
            
        return True
    except Exception as e:
        logger.error(f"Error during image classification test: {e}")
        return False''',
    
    "audio": '''# Additional methods for audio models
def test_audio_processing(self):
    """Test audio processing functionality."""
    try:
        # Create a test audio if none exists
        test_audio_path = "test_audio.wav"
        if not os.path.exists(test_audio_path):
            # Generate a simple sine wave
            import scipy.io.wavfile as wav
            sample_rate = 16000
            duration = 3  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            wav.write(test_audio_path, sample_rate, audio.astype(np.float32))
            
        # Load audio file
        sample_rate = 16000
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(test_audio_path)
        except:
            logger.warning("Could not load audio, using zeros array")
            audio = np.zeros(sample_rate * 3)  # 3 seconds of silence
            
        # Try different model classes
        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        except:
            try:
                # Try speech recognition model
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                processor = AutoProcessor.from_pretrained(self.model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            except:
                # Fallback to generic model
                from transformers import AutoModel, AutoFeatureExtractor
                processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                
        model = model.to(self.device)
        
        # Process audio
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Check outputs
        assert outputs is not None, "Outputs should not be None"
        
        # If it's a classification model, try to get class probabilities
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            logger.info(f"Logits shape: {logits.shape}")
            
        return True
    except Exception as e:
        logger.error(f"Error during audio processing test: {e}")
        return False''',
    
    "multimodal": '''# Additional methods for multimodal models
def test_multimodal_processing(self):
    """Test multimodal processing functionality."""
    try:
        # Create a test image if none exists
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            # Create a simple test image
            import numpy as np
            from PIL import Image
            size = 224
            img_array = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(size):
                for j in range(size):
                    img_array[i, j, :] = (i + j) % 256
            img = Image.fromarray(img_array)
            img.save(test_image_path)
            
        # Prepare text
        text = "What's in this image?"
            
        # Try different model classes
        try:
            from transformers import AutoProcessor, AutoModel
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
        except:
            try:
                # Try CLIP model
                from transformers import CLIPProcessor, CLIPModel
                processor = CLIPProcessor.from_pretrained(self.model_name)
                model = CLIPModel.from_pretrained(self.model_name)
            except:
                # Fallback
                from transformers import AutoProcessor, AutoModel
                processor = AutoProcessor.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                
        model = model.to(self.device)
        
        # Load and process the inputs
        from PIL import Image
        image = Image.open(test_image_path)
        
        # Process multimodal input
        try:
            inputs = processor(text=text, images=image, return_tensors="pt")
        except:
            try:
                # Try CLIP-style
                inputs = processor(text=[text], images=image, return_tensors="pt")
            except:
                # Try another method
                text_inputs = processor.tokenizer(text, return_tensors="pt")
                image_inputs = processor.image_processor(image, return_tensors="pt")
                inputs = {**text_inputs, **image_inputs}
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Check outputs
        assert outputs is not None, "Outputs should not be None"
        
        # If it's a classification/similarity model, check for specific outputs
        if hasattr(outputs, "logits_per_image") or hasattr(outputs, "logits_per_text"):
            logger.info("Model produces image-text similarity scores")
            
        return True
    except Exception as e:
        logger.error(f"Error during multimodal processing test: {e}")
        return False'''
}

# Custom model loading templates
CUSTOM_MODEL_LOADING_TEMPLATES = {
    "text_embedding": '''def get_model_specific(self):
        """Load model with specialized configuration."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Get tokenizer with specific settings
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                truncation_side="right",
                use_fast=True
            )
            
            # Get model with specific settings
            model = AutoModel.from_pretrained(
                self.model_name,
                torchscript=True if self.device == 'cpu' else False
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model with specific settings: {e}")
            return None, None''',
    
    "text_generation": '''def get_model_specific(self):
        """Load model with specialized configuration for text generation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Get tokenizer with specific settings
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                truncation_side="left",
                use_fast=True
            )
            
            # Get model with specific settings for generation
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == 'cuda' else None
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model with specific settings: {e}")
            return None, None''',
    
    "vision": '''def get_model_specific(self):
        """Load model with specialized configuration for vision tasks."""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            
            # Get image processor
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Get model with vision-specific settings
            model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                torchscript=True if self.device == 'cpu' else False
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, processor
        except Exception as e:
            logger.error(f"Error loading vision model with specific settings: {e}")
            
            # Fallback to generic model
            from transformers import AutoModel, AutoFeatureExtractor
            try:
                processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                model = model.to(self.device)
                model.eval()
                return model, processor
            except Exception as e2:
                logger.error(f"Error in fallback loading: {e2}")
                return None, None''',
    
    "audio": '''def get_model_specific(self):
        """Load model with specialized configuration for audio processing."""
        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            
            # Get feature extractor
            processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            
            # Get model with audio-specific settings
            model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                torchscript=True if self.device == 'cpu' else False
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, processor
        except Exception as e:
            logger.error(f"Error loading audio model with specific settings: {e}")
            
            # Try alternative model type (speech recognition)
            try:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                processor = AutoProcessor.from_pretrained(self.model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
                model = model.to(self.device)
                model.eval()
                return model, processor
            except Exception as e2:
                logger.error(f"Error in alternative loading: {e2}")
                
                # Fallback to generic model
                try:
                    from transformers import AutoModel, AutoFeatureExtractor
                    processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                    model = AutoModel.from_pretrained(self.model_name)
                    model = model.to(self.device)
                    model.eval()
                    return model, processor
                except Exception as e3:
                    logger.error(f"Error in fallback loading: {e3}")
                    return None, None''',
    
    "multimodal": '''def get_model_specific(self):
        """Load model with specialized configuration for multimodal tasks."""
        try:
            from transformers import AutoProcessor, AutoModel
            
            # Get processor for multimodal inputs
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Get model with multimodal-specific settings
            model = AutoModel.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == 'cuda' else None
            )
            model = model.to(self.device)
            
            # Put model in evaluation mode
            model.eval()
            
            return model, processor
        except Exception as e:
            logger.error(f"Error loading multimodal model with specific settings: {e}")
            
            # Try alternative model class
            try:
                from transformers import CLIPModel, CLIPProcessor
                processor = CLIPProcessor.from_pretrained(self.model_name)
                model = CLIPModel.from_pretrained(self.model_name)
                model = model.to(self.device)
                model.eval()
                return model, processor
            except Exception as e2:
                logger.error(f"Error in alternative loading: {e2}")
                return None, None'''
}

OUTPUT_CHECK_TEMPLATES = {
    "text_embedding": '''            # Check output shape and values
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
            assert outputs.last_hidden_state.shape[1] > 0, "Sequence length should be positive"
            logger.info(f"Output shape: {outputs.last_hidden_state.shape}")''',
    
    "vision": '''            # Check output shape and values
            assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
            assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
            logger.info(f"Output shape: {outputs.last_hidden_state.shape}")''',
    
    "audio": '''            # Check output shape and values
            assert outputs is not None, "Outputs should not be None"
            if hasattr(outputs, "last_hidden_state"):
                assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
                logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            else:
                # Some audio models have different output structures
                logger.info(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")''',
    
    "multimodal": '''            # Check output shape and values
            assert outputs is not None, "Outputs should not be None"
            if hasattr(outputs, "last_hidden_state"):
                assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
                logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            elif hasattr(outputs, "logits"):
                assert outputs.logits.shape[0] == 1, "Batch size should be 1"
                logger.info(f"Logits shape: {outputs.logits.shape}")
            else:
                # Some multimodal models have different output structures
                logger.info(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")'''
}

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import os
    from datetime import datetime
    
    # Add parent directory to path to import validator
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from generators.validators.template_validator_integration import validate_template_for_generator
    
    # Create a complete example template for a model
    def create_complete_template(model_type):
        """Create a complete template with specific model type components."""
        model_name = {
            "text_embedding": "bert-base-uncased",
            "vision": "google/vit-base-patch16-224",
            "audio": "openai/whisper-tiny",
            "multimodal": "openai/clip-vit-base-patch32"
        }.get(model_type, "bert-base-uncased")
        
        model_class_name = model_name.split('/')[-1].replace('-', '')
        model_input_code = MODEL_INPUT_TEMPLATES.get(model_type, MODEL_INPUT_TEMPLATES["text_embedding"])
        output_check_code = OUTPUT_CHECK_TEMPLATES.get(model_type, OUTPUT_CHECK_TEMPLATES["text_embedding"])
        
        template = STANDARD_TEMPLATE
        template = template.replace("{{model_name}}", model_name)
        template = template.replace("{{model_class_name}}", model_class_name)
        template = template.replace("{{model_type}}", model_type)
        template = template.replace("{{generation_date}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        template = template.replace("{{model_input_code}}", model_input_code)
        template = template.replace("{{output_check_code}}", output_check_code)
        template = template.replace("{{custom_model_loading}}", "def custom_method(self):\n        pass")
        template = template.replace("{{model_specific_code}}", "# No model specific code")
        
        return template
    
    # Create test dir
    test_dir = os.path.join(os.path.dirname(__file__), "test_templates")
    os.makedirs(test_dir, exist_ok=True)
    
    # Validate a complete template for each model type
    model_types = ["text_embedding", "vision", "audio", "multimodal"]
    for model_type in model_types:
        print(f"\nValidating {model_type} template...")
        template = create_complete_template(model_type)
        
        # Save to file for inspection
        template_file = os.path.join(test_dir, f"test_{model_type}.py")
        with open(template_file, 'w') as f:
            f.write(template)
        
        # Validate
        is_valid, errors = validate_template_for_generator(
            template,
            generator_type="fixed_template_example",
            validate_hardware=True,
            strict_indentation=False  # Be lenient with indentation in templates
        )
        
        if is_valid:
            print(f"✅ {model_type} template is valid!")
        else:
            print(f"❌ {model_type} template has errors:")
            for error in errors:
                print(f"  - {error}")
                
    print(f"\nTemplate test files written to {test_dir}")