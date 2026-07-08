"""
Text model adapters for benchmarking.
"""

import logging
from typing import Dict, Any, Optional

import torch
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForQuestionAnswering, AutoTokenizer
)

from . import ModelAdapter

logger = logging.getLogger("benchmark.models.text")

class TextModelAdapter(ModelAdapter):
    """
    Adapter for text-based models.
    
    Handles loading and input preparation for various text model types.
    """
    
    def __init__(self, model_id: str, task: Optional[str] = None):
        """
        Initialize a text model adapter.
        
        Args:
            model_id: HuggingFace model ID
            task: Model task
        """
        super().__init__(model_id, task)
        
        # Model type detection based on ID
        self.model_id_lower = self.model_id.lower()
        
        # Detect modern LLM families
        self.is_llama = "llama" in self.model_id_lower
        self.is_mistral = "mistral" in self.model_id_lower
        self.is_falcon = "falcon" in self.model_id_lower
        self.is_mpt = "mpt" in self.model_id_lower
        self.is_phi = "phi" in self.model_id_lower
        
        # Detect traditional model families
        self.is_bert = "bert" in self.model_id_lower
        self.is_gpt = "gpt" in self.model_id_lower
        self.is_t5 = "t5" in self.model_id_lower
        self.is_bart = "bart" in self.model_id_lower
        
        # Default task based on model type if not provided
        if self.task is None:
            if any([self.is_llama, self.is_mistral, self.is_falcon, self.is_mpt, self.is_phi, self.is_gpt]):
                self.task = "text-generation"
            elif self.is_bert:
                self.task = "fill-mask"
            elif self.is_t5 or self.is_bart:
                self.task = "text2text-generation"
            else:
                self.task = "text-generation"  # Default
        
        # Initialize tokenizer
        self.tokenizer = None
    
    def load_model(self, device: torch.device, use_4bit: bool = False, use_8bit: bool = False) -> torch.nn.Module:
        """
        Load the text model on the specified device.
        
        Args:
            device: Device to load the model on
            use_4bit: Whether to use 4-bit quantization (for modern LLMs)
            use_8bit: Whether to use 8-bit quantization (for modern LLMs)
            
        Returns:
            Loaded model
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Check if we need quantization for large models
        quantization_config = None
        
        # Modern LLM-specific optimizations
        is_large_model = any([self.is_llama, self.is_mistral, self.is_falcon, self.is_mpt])
        
        if is_large_model and (use_4bit or use_8bit):
            try:
                from transformers import BitsAndBytesConfig
                
                # Configure quantization based on requested precision
                if use_4bit:
                    logger.info(f"Using 4-bit quantization for model {self.model_id}")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4"
                    )
                elif use_8bit:
                    logger.info(f"Using 8-bit quantization for model {self.model_id}")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
            except ImportError:
                logger.warning("BitsAndBytes not available. Install with 'pip install bitsandbytes transformers>=4.30.0'")
                
        # Set default model loading kwargs
        model_kwargs = {}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        
        # Load appropriate model class based on task
        try:
            if self.task == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            elif self.task == "fill-mask":
                model = AutoModelForMaskedLM.from_pretrained(self.model_id, **model_kwargs)
            elif self.task == "text-classification":
                model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **model_kwargs)
            elif self.task == "token-classification":
                model = AutoModelForTokenClassification.from_pretrained(self.model_id, **model_kwargs)
            elif self.task == "question-answering":
                model = AutoModelForQuestionAnswering.from_pretrained(self.model_id, **model_kwargs)
            elif self.task == "text2text-generation":
                # Handle sequence-to-sequence models like T5, BART
                from transformers import AutoModelForSeq2SeqLM
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, **model_kwargs)
            else:
                # Default to base model
                logger.warning(f"Unknown task '{self.task}' for text model, using AutoModel")
                model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}")
            # Fallback to base model
            logger.warning(f"Falling back to AutoModel for {self.model_id}")
            model = AutoModel.from_pretrained(self.model_id)
        
        # Move model to device and set to evaluation mode
        if not use_4bit and not use_8bit:  # Quantized models are already on the right device
            model = model.to(device)
        model.eval()
        
        return model
    
    def prepare_inputs(self, batch_size: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the text model.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Dictionary of input tensors
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call load_model first.")
        
        # Generate appropriate input text based on task
        if self.task == "fill-mask":
            # For masked language models
            if self.tokenizer.mask_token:
                text = f"The quick {self.tokenizer.mask_token} fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
                text = text[:sequence_length]
            else:
                # Fallback if mask token not available
                text = "The quick brown fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
                text = text[:sequence_length]
        
        elif self.task == "question-answering":
            # For question answering models
            question = "What color is the fox?"
            context = "The quick brown fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
            context = context[:sequence_length]
            
            # Special handling for QA inputs
            return self.tokenizer(
                [question] * batch_size,
                [context] * batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=sequence_length
            )
        
        else:
            # Default text input for other tasks
            text = "The quick brown fox jumps over the lazy dog. " * (sequence_length // 10 + 1)
            text = text[:sequence_length]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            [text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=sequence_length
        )
        
        return inputs