#!/usr/bin/env python3
"""
Reference model generator module for the refactored generator suite.

This module provides functionality to generate model-specific code
based on the reference implementation pattern found in ipfs_accelerate_py/worker/skillset.
"""

import os
import re
import sys
import time
import json
import logging
import importlib
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from generators.architecture_detector import (
    get_architecture_type,
    get_model_metadata,
    get_default_model_id
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReferenceModelGenerator:
    """Generator for model skillset files using the reference implementation pattern."""
    
    def __init__(self, template_dir: str = None, output_dir: str = None):
        """
        Initialize the reference model generator.
        
        Args:
            template_dir: Directory containing templates. If None, uses default.
            output_dir: Directory for generated files. If None, uses default.
        """
        # Set directories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.template_dir = template_dir or os.path.join(base_dir, "templates")
        
        # Set output directory to the target worker/skillset directory or local if not found
        worker_skillset_dir = os.path.join(base_dir, "..", "ipfs_accelerate_py", "worker", "skillset")
        if os.path.exists(worker_skillset_dir):
            self.output_dir = output_dir or worker_skillset_dir
        else:
            self.output_dir = output_dir or os.path.join(base_dir, "skillsets")
        
        # Ensure directories exist
        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Reference template file
        self.reference_template = "hf_reference_template.py"
        
        logger.info(f"Initialized ReferenceModelGenerator with templates from {self.template_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Architecture-specific task types and model classes
        self.arch_task_mappings = {
            "encoder-only": {
                "task_type": "text_embedding",
                "task_class": "MaskedLM",
                "description": "This is a transformer-based language model designed to understand context in text by looking at words bidirectionally. It's commonly used for text embedding generation, which can be used for tasks like semantic search, text classification, and more.",
                "hidden_size": 768,
                "automodel_class": "self.transformers.AutoModelForMaskedLM",
                "test_input": "The quick brown fox jumps over the lazy dog."
            },
            "decoder-only": {
                "task_type": "text_generation",
                "task_class": "CausalLM",
                "description": "This is a generative language model that predicts the next token in a sequence. It's used for text generation tasks like story writing, code completion, and conversational AI.",
                "hidden_size": 768,
                "automodel_class": "self.transformers.AutoModelForCausalLM",
                "test_input": "Once upon a time, there was a"
            },
            "encoder-decoder": {
                "task_type": "text2text_generation",
                "task_class": "Seq2SeqLM",
                "description": "This is a sequence-to-sequence model that can transform input text into output text, useful for tasks like translation, summarization, and question answering.",
                "hidden_size": 768,
                "automodel_class": "self.transformers.AutoModelForSeq2SeqLM",
                "test_input": "Translate to French: Hello, how are you?"
            },
            "vision": {
                "task_type": "image_classification",
                "task_class": "ImageClassification",
                "description": "This model processes images using a vision transformer architecture to extract visual features and classify images into categories.",
                "hidden_size": 768,
                "automodel_class": "self.transformers.AutoModelForImageClassification",
                "test_input": "[IMAGE PLACEHOLDER]"
            },
            "vision-encoder-text-decoder": {
                "task_type": "multimodal_embedding",
                "task_class": "VisionTextDualEncoder",
                "description": "This model processes both images and text, aligning them in a shared embedding space. It's used for tasks like image-text matching, visual reasoning, and multimodal search.",
                "hidden_size": 768,
                "automodel_class": "self.transformers.VisionTextDualEncoderModel",
                "test_input": "A photograph of a beautiful mountain landscape"
            },
            "speech": {
                "task_type": "speech_recognition",
                "task_class": "SpeechSeq2Seq",
                "description": "This model processes audio input and converts it to text. It's used for automatic speech recognition, transcription, and audio understanding tasks.",
                "hidden_size": 768,
                "automodel_class": "self.transformers.AutoModelForSpeechSeq2Seq",
                "test_input": "[AUDIO PLACEHOLDER]"
            }
        }
        
        # Initialize template snippets
        self._initialize_template_snippets()
    
    def _initialize_template_snippets(self):
        """Initialize code snippets for different architecture types."""
        # Encoder-only model mock tokenizer output
        self.arch_snippets = {
            "encoder-only": {
                "mock_tokenize_output": """return {
                    "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
                    "token_type_ids": torch.zeros((batch_size, 10), dtype=torch.long)
                }""",
                "mock_forward_output": """result = MagicMock()
                result.last_hidden_state = torch.rand((batch_size, sequence_length, hidden_size))
                return result""",
                "cpu_inference_code": """outputs = endpoint(**inputs)
                    # Get embeddings from the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()""",
                "cpu_result_format": """{"success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label}""",
                "cuda_inference_code": """outputs = endpoint(**inputs)
                    # Get embeddings from the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()""",
                "cuda_result_format": """{"success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label}""",
                "rocm_inference_code": """outputs = endpoint(**inputs)
                    # Get embeddings from the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()""",
                "rocm_result_format": """{"success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label}""",
                "openvino_inference_code": """outputs = endpoint(**inputs)
                    # Get embeddings from the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()""",
                "openvino_result_format": """{"success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label}""",
                "apple_inference_code": """outputs = endpoint(**inputs)
                    # Get embeddings from the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()""",
                "apple_result_format": """{"success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label}""",
                "qualcomm_result_format": """"embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]"""
            },
            "decoder-only": {
                "mock_tokenize_output": """return {
                    "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
                }""",
                "mock_forward_output": """result = MagicMock()
                result.logits = torch.rand((batch_size, sequence_length, 50000))  # Vocab size
                return result""",
                "cpu_inference_code": """# For text generation, we need to use the generate method
                    output_ids = endpoint.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode the generated text
                    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "cpu_result_format": """{"success": True,
                    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
                    "all_texts": generated_texts,
                    "device": device,
                    "hardware": hardware_label}""",
                "cuda_inference_code": """# For text generation, we need to use the generate method
                    output_ids = endpoint.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode the generated text
                    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "cuda_result_format": """{"success": True,
                    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
                    "all_texts": generated_texts,
                    "device": device,
                    "hardware": hardware_label}""",
                "openvino_inference_code": """# For text generation, we need to use the generate method
                    output_ids = endpoint.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1
                    )
                    
                    # Decode the generated text
                    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "openvino_result_format": """{"success": True,
                    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
                    "all_texts": generated_texts,
                    "device": device,
                    "hardware": hardware_label}""",
                "apple_inference_code": """# For text generation, we need to use the generate method
                    output_ids = endpoint.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode the generated text
                    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "apple_result_format": """{"success": True,
                    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
                    "all_texts": generated_texts,
                    "device": device,
                    "hardware": hardware_label}""",
                "qualcomm_result_format": """"generated_text": f"Mock Qualcomm generated text for: {batch[0][:30]}...",
                    "all_texts": [f"Mock Qualcomm generated text for: {text[:30]}..." for text in batch]"""
            },
            "encoder-decoder": {
                "mock_tokenize_output": """return {
                    "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
                }""",
                "mock_forward_output": """result = MagicMock()
                result.logits = torch.rand((batch_size, sequence_length, 50000))  # Vocab size
                return result""",
                "cpu_inference_code": """# For seq2seq models, we need to use the generate method
                    output_ids = endpoint.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1
                    )
                    
                    # Decode the generated text
                    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "cpu_result_format": """{"success": True,
                    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
                    "all_texts": generated_texts,
                    "device": device,
                    "hardware": hardware_label}""",
                "cuda_inference_code": """# For seq2seq models, we need to use the generate method
                    output_ids = endpoint.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1
                    )
                    
                    # Decode the generated text
                    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "cuda_result_format": """{"success": True,
                    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
                    "all_texts": generated_texts,
                    "device": device,
                    "hardware": hardware_label}""",
                "openvino_inference_code": """# For seq2seq models, we need to use the generate method
                    output_ids = endpoint.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1
                    )
                    
                    # Decode the generated text
                    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "openvino_result_format": """{"success": True,
                    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
                    "all_texts": generated_texts,
                    "device": device,
                    "hardware": hardware_label}""",
                "apple_inference_code": """# For seq2seq models, we need to use the generate method
                    output_ids = endpoint.generate(
                        **inputs,
                        max_length=100,
                        num_return_sequences=1
                    )
                    
                    # Decode the generated text
                    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "apple_result_format": """{"success": True,
                    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
                    "all_texts": generated_texts,
                    "device": device,
                    "hardware": hardware_label}""",
                "qualcomm_result_format": """"generated_text": f"Mock Qualcomm generated text for: {batch[0][:30]}...",
                    "all_texts": [f"Mock Qualcomm generated text for: {text[:30]}..." for text in batch]"""
            },
            "vision": {
                "mock_tokenize_output": """# Vision models use image processors, not tokenizers
                import numpy as np
                dummy_images = np.random.randint(0, 255, size=(batch_size, 3, 224, 224), dtype=np.uint8)
                return {"pixel_values": torch.tensor(dummy_images, dtype=torch.float32)}""",
                "mock_forward_output": """result = MagicMock()
                num_classes = 1000  # Typical ImageNet classes
                result.logits = torch.rand((batch_size, num_classes))
                return result""",
                "cpu_inference_code": """outputs = endpoint(**inputs)
                    # Get the predicted class IDs
                    probs = outputs.logits.softmax(dim=-1)
                    predicted_class_idx = probs.argmax(-1).cpu().numpy().tolist()
                    
                    # In a real implementation, we would map these to class labels
                    # For now, we just return the indices
                    class_labels = [f"class_{idx}" for idx in predicted_class_idx]
                    confidences = probs.max(dim=-1).values.cpu().numpy().tolist()""",
                "cpu_result_format": """{"success": True,
                    "predictions": [{"label": label, "confidence": conf} for label, conf in zip(class_labels, confidences)],
                    "device": device,
                    "hardware": hardware_label}""",
                "cuda_inference_code": """outputs = endpoint(**inputs)
                    # Get the predicted class IDs
                    probs = outputs.logits.softmax(dim=-1)
                    predicted_class_idx = probs.argmax(-1).cpu().numpy().tolist()
                    
                    # In a real implementation, we would map these to class labels
                    # For now, we just return the indices
                    class_labels = [f"class_{idx}" for idx in predicted_class_idx]
                    confidences = probs.max(dim=-1).values.cpu().numpy().tolist()""",
                "cuda_result_format": """{"success": True,
                    "predictions": [{"label": label, "confidence": conf} for label, conf in zip(class_labels, confidences)],
                    "device": device,
                    "hardware": hardware_label}""",
                "openvino_inference_code": """outputs = endpoint(**inputs)
                    # Get the predicted class IDs
                    probs = outputs.logits.softmax(dim=-1)
                    predicted_class_idx = probs.argmax(-1).cpu().numpy().tolist()
                    
                    # In a real implementation, we would map these to class labels
                    # For now, we just return the indices
                    class_labels = [f"class_{idx}" for idx in predicted_class_idx]
                    confidences = probs.max(dim=-1).values.cpu().numpy().tolist()""",
                "openvino_result_format": """{"success": True,
                    "predictions": [{"label": label, "confidence": conf} for label, conf in zip(class_labels, confidences)],
                    "device": device,
                    "hardware": hardware_label}""",
                "apple_inference_code": """outputs = endpoint(**inputs)
                    # Get the predicted class IDs
                    probs = outputs.logits.softmax(dim=-1)
                    predicted_class_idx = probs.argmax(-1).cpu().numpy().tolist()
                    
                    # In a real implementation, we would map these to class labels
                    # For now, we just return the indices
                    class_labels = [f"class_{idx}" for idx in predicted_class_idx]
                    confidences = probs.max(dim=-1).values.cpu().numpy().tolist()""",
                "apple_result_format": """{"success": True,
                    "predictions": [{"label": label, "confidence": conf} for label, conf in zip(class_labels, confidences)],
                    "device": device,
                    "hardware": hardware_label}""",
                "qualcomm_result_format": """"predictions": [{"label": f"class_{i % 1000}", "confidence": 0.9 - (i * 0.1 % 0.9)} for i in range(len(batch))]"""
            },
            "vision-encoder-text-decoder": {
                "mock_tokenize_output": """# Vision-text models have complex preprocessing
                # Just return dummy tensors for both modalities
                import numpy as np
                dummy_images = np.random.randint(0, 255, size=(batch_size, 3, 224, 224), dtype=np.uint8)
                return {
                    "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
                    "pixel_values": torch.tensor(dummy_images, dtype=torch.float32)
                }""",
                "mock_forward_output": """result = MagicMock()
                result.logits_per_image = torch.rand((batch_size, batch_size))
                result.logits_per_text = torch.rand((batch_size, batch_size))
                return result""",
                "cpu_inference_code": """# Most vision-text models compute similarity scores
                    # between images and text
                    outputs = endpoint(**inputs)
                    
                    # Extract similarity scores
                    if hasattr(outputs, 'logits_per_image'):
                        # CLIP-style model
                        similarity_scores = outputs.logits_per_image.cpu().numpy().tolist()
                    else:
                        # For other models, this would need to be adapted
                        similarity_scores = [[0.5] * len(batch)]""",
                "cpu_result_format": """{"success": True,
                    "similarity_scores": similarity_scores[0] if similarity_scores else [],
                    "device": device,
                    "hardware": hardware_label}""",
                "cuda_inference_code": """# Most vision-text models compute similarity scores
                    # between images and text
                    outputs = endpoint(**inputs)
                    
                    # Extract similarity scores
                    if hasattr(outputs, 'logits_per_image'):
                        # CLIP-style model
                        similarity_scores = outputs.logits_per_image.cpu().numpy().tolist()
                    else:
                        # For other models, this would need to be adapted
                        similarity_scores = [[0.5] * len(batch)]""",
                "cuda_result_format": """{"success": True,
                    "similarity_scores": similarity_scores[0] if similarity_scores else [],
                    "device": device,
                    "hardware": hardware_label}""",
                "openvino_inference_code": """# Most vision-text models compute similarity scores
                    # between images and text
                    outputs = endpoint(**inputs)
                    
                    # Extract similarity scores
                    if hasattr(outputs, 'logits_per_image'):
                        # CLIP-style model
                        similarity_scores = outputs.logits_per_image.cpu().numpy().tolist()
                    else:
                        # For other models, this would need to be adapted
                        similarity_scores = [[0.5] * len(batch)]""",
                "openvino_result_format": """{"success": True,
                    "similarity_scores": similarity_scores[0] if similarity_scores else [],
                    "device": device,
                    "hardware": hardware_label}""",
                "apple_inference_code": """# Most vision-text models compute similarity scores
                    # between images and text
                    outputs = endpoint(**inputs)
                    
                    # Extract similarity scores
                    if hasattr(outputs, 'logits_per_image'):
                        # CLIP-style model
                        similarity_scores = outputs.logits_per_image.cpu().numpy().tolist()
                    else:
                        # For other models, this would need to be adapted
                        similarity_scores = [[0.5] * len(batch)]""",
                "apple_result_format": """{"success": True,
                    "similarity_scores": similarity_scores[0] if similarity_scores else [],
                    "device": device,
                    "hardware": hardware_label}""",
                "qualcomm_result_format": """"similarity_scores": [0.9, 0.8, 0.7, 0.6, 0.5]"""
            },
            "speech": {
                "mock_tokenize_output": """# Speech models use feature extractors, not tokenizers
                import numpy as np
                # Create dummy audio features (typically spectrogram or waveform)
                dummy_audio = np.random.randn(batch_size, 16000)  # 1 second of audio at 16kHz
                return {"input_features": torch.tensor(dummy_audio, dtype=torch.float32)}""",
                "mock_forward_output": """result = MagicMock()
                result.logits = torch.rand((batch_size, sequence_length, 50000))  # Vocab size
                return result""",
                "cpu_inference_code": """# For speech recognition, we typically use generate
                    output_ids = endpoint.generate(
                        inputs.get("input_features")
                    )
                    
                    # Decode the predicted text
                    transcriptions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "cpu_result_format": """{"success": True,
                    "text": transcriptions[0] if transcriptions else "",
                    "all_texts": transcriptions,
                    "device": device,
                    "hardware": hardware_label}""",
                "cuda_inference_code": """# For speech recognition, we typically use generate
                    output_ids = endpoint.generate(
                        inputs.get("input_features").to(device)
                    )
                    
                    # Decode the predicted text
                    transcriptions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "cuda_result_format": """{"success": True,
                    "text": transcriptions[0] if transcriptions else "",
                    "all_texts": transcriptions,
                    "device": device,
                    "hardware": hardware_label}""",
                "openvino_inference_code": """# For speech recognition, we typically use generate
                    output_ids = endpoint.generate(
                        inputs.get("input_features")
                    )
                    
                    # Decode the predicted text
                    transcriptions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "openvino_result_format": """{"success": True,
                    "text": transcriptions[0] if transcriptions else "",
                    "all_texts": transcriptions,
                    "device": device,
                    "hardware": hardware_label}""",
                "apple_inference_code": """# For speech recognition, we typically use generate
                    output_ids = endpoint.generate(
                        inputs.get("input_features").to(device)
                    )
                    
                    # Decode the predicted text
                    transcriptions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)""",
                "apple_result_format": """{"success": True,
                    "text": transcriptions[0] if transcriptions else "",
                    "all_texts": transcriptions,
                    "device": device,
                    "hardware": hardware_label}""",
                "qualcomm_result_format": """"text": "This is a mocked transcription result.", 
                    "all_texts": ["This is a mocked transcription result." for _ in range(len(batch))]"""
            }
        }
    
    def generate_reference_implementation(self, model_name: str, force: bool = False, verify: bool = True) -> Tuple[bool, List[str]]:
        """
        Generate a reference implementation file for a specific model.
        
        Args:
            model_name: The model name.
            force: Whether to overwrite existing files.
            verify: Whether to verify generated files with Python syntax check.
            
        Returns:
            Tuple of (success, list of generated file paths).
        """
        logger.info(f"Generating reference implementation for model: {model_name}")
        
        # Determine architecture type
        arch_type = get_architecture_type(model_name)
        logger.info(f"Detected architecture type: {arch_type}")
        
        # Get model metadata
        metadata = get_model_metadata(model_name)
        model_type = metadata["model_type"]
        model_type_upper = model_type.upper()
        
        # Check if architecture type is supported
        if arch_type not in self.arch_task_mappings:
            logger.error(f"Architecture type {arch_type} not supported for reference implementation.")
            return False, []
        
        # Get task mapping for this architecture
        task_mapping = self.arch_task_mappings[arch_type]
        
        # Get snippets for this architecture
        snippets = self.arch_snippets[arch_type]
        
        # Output file path uses hf_ prefix followed by model_type
        output_filename = f"hf_{model_type}.py"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Check if file exists
        if os.path.exists(output_path) and not force:
            logger.warning(f"File already exists: {output_path}. Use force=True to overwrite.")
            return True, []  # Return True since the file already exists
        
        # Read reference template
        template_path = os.path.join(self.template_dir, self.reference_template)
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
        except Exception as e:
            logger.error(f"Error reading template {template_path}: {e}")
            return False, []
        
        # Add indentation to code snippets
        def add_indentation(code, indent_level=16):
            indent = ' ' * indent_level
            lines = code.splitlines()
            indented_lines = [indent + line for line in lines]
            return '\n'.join(indented_lines)
        
        # Replace placeholders in template
        replacements = {
            "{model_type}": model_type,
            "{model_type_upper}": model_type_upper,
            "{model_description}": task_mapping["description"],
            "{task_type}": task_mapping["task_type"],
            "{task_class}": task_mapping["task_class"],
            "{hidden_size}": str(task_mapping["hidden_size"]),
            "{automodel_class}": task_mapping["automodel_class"],
            "{test_input}": task_mapping["test_input"],
            "{mock_tokenize_output}": add_indentation(snippets["mock_tokenize_output"]),
            "{mock_forward_output}": add_indentation(snippets["mock_forward_output"]),
            "{cpu_inference_code}": add_indentation(snippets["cpu_inference_code"]),
            "{cpu_result_format}": add_indentation(snippets["cpu_result_format"]),
            "{cuda_inference_code}": add_indentation(snippets["cuda_inference_code"]),
            "{cuda_result_format}": add_indentation(snippets["cuda_result_format"]),
            "{rocm_inference_code}": add_indentation(snippets["rocm_inference_code"]),
            "{rocm_result_format}": add_indentation(snippets["rocm_result_format"]),
            "{openvino_inference_code}": add_indentation(snippets["openvino_inference_code"]),
            "{openvino_result_format}": add_indentation(snippets["openvino_result_format"]),
            "{apple_inference_code}": add_indentation(snippets["apple_inference_code"]),
            "{apple_result_format}": add_indentation(snippets["apple_result_format"]),
            "{qualcomm_result_format}": add_indentation(snippets["qualcomm_result_format"])
        }
        
        filled_content = template_content
        for placeholder, value in replacements.items():
            filled_content = filled_content.replace(placeholder, value)
        
        # Write to file
        try:
            with open(output_path, 'w') as f:
                f.write(filled_content)
            logger.info(f"Generated file: {output_path}")
        except Exception as e:
            logger.error(f"Error writing file {output_path}: {e}")
            return False, []
        
        # Verify file
        if verify and not self.verify_reference_file(output_path):
            logger.error(f"Verification failed for {output_path}")
            return False, []
        
        return True, [output_path]
    
    def verify_reference_file(self, file_path: str) -> bool:
        """
        Verify a generated reference file for syntax correctness.
        
        Args:
            file_path: Path to the file to verify.
            
        Returns:
            True if verification succeeded, False otherwise.
        """
        try:
            # Use Python's builtin compile function to check syntax
            with open(file_path, 'r') as f:
                content = f.read()
            
            compile(content, file_path, 'exec')
            logger.info(f"Syntax verification passed for {file_path}")
            return True
        
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False
    
    def generate_for_priority(self, priority: str, force: bool = False, verify: bool = True) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Generate reference implementations for models in a priority level.
        
        Args:
            priority: Priority level ("critical", "high", "medium", "low").
            force: Whether to overwrite existing files.
            verify: Whether to verify generated files.
            
        Returns:
            Tuple of (success, dict mapping model names to file paths).
        """
        # Import priority models
        from generators.model_generator import PRIORITY_MODELS
        
        if priority not in PRIORITY_MODELS:
            logger.error(f"Unknown priority level: {priority}")
            return False, {}
        
        models = PRIORITY_MODELS[priority]
        logger.info(f"Generating reference implementations for {len(models)} models with priority '{priority}'")
        
        results = {}
        overall_success = True
        
        for model in models:
            success, files = self.generate_reference_implementation(model, force, verify)
            results[model] = files
            
            if not success:
                overall_success = False
        
        return overall_success, results


# Direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reference model implementations")
    parser.add_argument("--model", "-m", type=str, help="Model to generate implementation for")
    parser.add_argument("--priority", "-p", type=str, default="critical",
                       choices=["critical", "high", "medium", "low"],
                       help="Priority level to generate implementations for")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing files")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    parser.add_argument("--template-dir", "-t", type=str, help="Template directory")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    generator = ReferenceModelGenerator(args.template_dir, args.output_dir)
    
    if args.model:
        # Generate for specific model
        success, files = generator.generate_reference_implementation(
            args.model, args.force, not args.no_verify
        )
        
        if success:
            print(f"Successfully generated reference implementation for {args.model}")
            for file in files:
                print(f"  - {file}")
            sys.exit(0)
        else:
            print(f"Failed to generate reference implementation for {args.model}")
            sys.exit(1)
    else:
        # Generate for priority level
        success, results = generator.generate_for_priority(
            args.priority, args.force, not args.no_verify
        )
        
        if success:
            print(f"Successfully generated reference implementations for {args.priority} priority models")
            for model, files in results.items():
                if files:  # Only show models with generated files
                    print(f"  - {model}: {len(files)} file(s)")
            sys.exit(0)
        else:
            print(f"Failed to generate some reference implementations for {args.priority} priority models")
            sys.exit(1)