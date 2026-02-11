#!/usr/bin/env python3
"""
Template for State Space Models (SSM) such as Mamba, RWKV, etc.

This template is designed for models that use state space representations
instead of traditional attention mechanisms. These models often have
linear scaling with sequence length.
"""

import os
import time
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from transformers import {model_class_name}, {processor_class_name}
    import transformers
except ImportError:
    raise ImportError(
        "The transformers package is required to use this model. "
        "Please install it with `pip install transformers`."
    )

logger = logging.getLogger(__name__)

class {skillset_class_name}:
    """
    Skillset for {model_type_upper} - a state space model (SSM) that processes
    sequences with linear time and memory complexity.
    """
    
    def __init__(self, model_id: str = "{default_model_id}", device: str = "cpu", **kwargs):
        """
        Initialize the {model_type_upper} model.
        
        Args:
            model_id: HuggingFace model ID or path
            device: Device to run the model on ('cpu', 'cuda', 'mps', etc.)
            **kwargs: Additional arguments to pass to the model
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
        # Track hardware info for reporting
        self.hardware_info = {
            "device": device,
            "device_name": None,
            "memory_available": None,
            "supports_half_precision": False,
            "ssm_specific": {
                "state_size": None,
                "hidden_size": None,
                "max_sequence_length": None
            }
        }
        
        # Optional configuration
        self.low_memory_mode = kwargs.get("low_memory_mode", False)
        self.max_memory = kwargs.get("max_memory", None)
        self.torch_dtype = kwargs.get("torch_dtype", None)
        
        # Initialize the model if auto_init is True
        auto_init = kwargs.get("auto_init", True)
        if auto_init:
            self.initialize()
    
    def initialize(self):
        """Initialize the model and tokenizer."""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing {self.model_id} on {self.device}")
        start_time = time.time()
        
        try:
            # Check if CUDA is available when device is cuda
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if MPS is available when device is mps
            if self.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if ROCm/HIP is available when device is rocm
            if self.device == "rocm":
                rocm_available = False
                try:
                    if hasattr(torch, 'hip') and torch.hip.is_available():
                        rocm_available = True
                    elif torch.cuda.is_available():
                        # Could be ROCm using CUDA API
                        device_name = torch.cuda.get_device_name(0)
                        if "AMD" in device_name or "Radeon" in device_name:
                            rocm_available = True
                            self.device = "cuda"  # ROCm uses CUDA device map
                except:
                    pass
                
                if not rocm_available:
                    logger.warning("ROCm requested but not available, falling back to CPU")
                    self.device = "cpu"
            
            # Record hardware info
            if self.device.startswith("cuda"):
                self.hardware_info["device_name"] = torch.cuda.get_device_name(0)
                self.hardware_info["memory_available"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.hardware_info["supports_half_precision"] = True
            
            # Device-specific initialization
            {device_init_code}
            
            # Load tokenizer
            self.tokenizer = {processor_class_name}.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Extract model-specific parameters
            if hasattr(self.model.config, "hidden_size"):
                self.hardware_info["ssm_specific"]["hidden_size"] = self.model.config.hidden_size
            if hasattr(self.model.config, "state_size") or hasattr(self.model.config, "d_state"):
                state_size = getattr(self.model.config, "state_size", None) or getattr(self.model.config, "d_state", None)
                self.hardware_info["ssm_specific"]["state_size"] = state_size
            if hasattr(self.model.config, "max_seq_len") or hasattr(self.model.config, "max_position_embeddings"):
                max_len = getattr(self.model.config, "max_seq_len", None) or getattr(self.model.config, "max_position_embeddings", None)
                self.hardware_info["ssm_specific"]["max_sequence_length"] = max_len
            
            self.is_initialized = True
            logger.info(f"Initialized {self.model_id} in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_p=0.9, **kwargs):
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter (lower = less diverse)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated text
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # SSM models like Mamba can benefit from stateful generation
            # where the state is preserved across chunks, but we implement
            # a basic version here that works for all models
            
            # Prepare input
            if isinstance(prompt, str):
                prompts = [prompt]
            else:
                prompts = prompt
            
            # Tokenize input
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Set up generation parameters
            generation_config = transformers.GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Generate text
            start_time = time.time()
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            generation_time = time.time() - start_time
            
            # Decode output
            generated_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Get just the newly generated text (not including the prompt)
            new_texts = []
            for i, text in enumerate(generated_texts):
                if text.startswith(prompts[i]):
                    new_texts.append(text[len(prompts[i]):])
                else:
                    new_texts.append(text)
            
            # Return results with metadata
            return {
                "full_text": generated_texts,
                "generated_text": new_texts,
                "prompt": prompts,
                "generation_time": generation_time,
                "tokens_per_second": max_new_tokens / generation_time if generation_time > 0 else 0,
                "model_id": self.model_id,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return {"error": str(e)}
    
    def embed(self, text, pooling="mean", **kwargs):
        """
        Generate embeddings for text input.
        
        Args:
            text: Text input or list of text inputs
            pooling: Pooling method ('mean', 'cls', 'max')
            **kwargs: Additional embedding parameters
            
        Returns:
            Dictionary containing embeddings
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Prepare input
            if isinstance(text, str):
                batch = [text]
            else:
                batch = text
            
            # Tokenize input
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, **kwargs)
            
            # Get embeddings from the last hidden state
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            else:
                # Fallback to last_hidden_state
                hidden_states = outputs.last_hidden_state
            
            # Apply pooling
            if pooling == "cls":
                embeddings = hidden_states[:, 0].cpu().numpy()  # First token
            elif pooling == "mean":
                # Apply mean pooling - take attention mask into account for averaging
                attention_mask = inputs["attention_mask"]
                embeddings = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
                embeddings = embeddings.cpu().numpy()
            elif pooling == "max":
                # Apply max pooling
                embeddings = torch.max(hidden_states, dim=1)[0].cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            
            # Convert to Python list
            embeddings_list = embeddings.tolist()
            
            # Return results with metadata
            return {
                "embeddings": embeddings_list,
                "dimension": embeddings.shape[1],
                "text": batch,
                "model_id": self.model_id,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {"error": str(e)}
    
    def state_tracking_generation(self, prompt, max_tokens=1000, chunk_size=100, **kwargs):
        """
        Generate text using state tracking for more efficient long-form generation.
        This is especially efficient for SSM models with linear complexity.
        
        Args:
            prompt: Initial prompt
            max_tokens: Maximum total tokens to generate
            chunk_size: Number of tokens to generate in each iteration
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated text
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Initial tokenization
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            input_len = input_ids.shape[1]
            
            # Generation parameters
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)
            do_sample = kwargs.get("do_sample", True if temperature > 0 else False)
            
            # Set up for generation
            generated_tokens = input_ids.clone()
            total_time = 0
            
            # Check if model has a specific stateful generation method (e.g., some Mamba implementations)
            has_custom_stateful = hasattr(self.model, "stateful_generate") or hasattr(self.model, "generate_with_state")
            
            if has_custom_stateful:
                # Use model's custom stateful generation
                stateful_method = getattr(self.model, "stateful_generate", None) or getattr(self.model, "generate_with_state")
                
                start_time = time.time()
                outputs = stateful_method(
                    input_ids,
                    max_length=input_len + max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    **kwargs
                )
                total_time = time.time() - start_time
                
                generated_tokens = outputs
                
            else:
                # Use chunked generation for standard models
                state = None  # Some SSM models support passing state between calls
                
                # Generate text in chunks
                for i in range(0, max_tokens, chunk_size):
                    chunk_size = min(chunk_size, max_tokens - i)
                    
                    # Generate chunk
                    start_time = time.time()
                    
                    with torch.no_grad():
                        if state is not None and hasattr(self.model, "forward_with_state"):
                            # SSM-specific stateful forward call
                            outputs, state = self.model.forward_with_state(
                                input_ids=generated_tokens[:, -chunk_size:] if i > 0 else generated_tokens,
                                state=state,
                                **kwargs
                            )
                            
                            # Get next token using sampling
                            logits = outputs.logits[:, -1, :]
                            if do_sample:
                                logits = logits / (temperature if temperature > 0 else 1.0)
                                filtered_logits = transformers.top_p_filtering(logits, top_p=top_p)
                                probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
                                next_token = torch.multinomial(probabilities, 1)
                            else:
                                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                                
                            # Append to generated tokens
                            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                            
                        else:
                            # Standard HF generation for current chunk
                            outputs = self.model.generate(
                                generated_tokens,
                                max_new_tokens=chunk_size,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=do_sample,
                                **kwargs
                            )
                            
                            # Update generated tokens
                            generated_tokens = outputs
                    
                    chunk_time = time.time() - start_time
                    total_time += chunk_time
                    
                    # Early stopping if end of text token is generated
                    if generated_tokens[0, -1].item() == self.tokenizer.eos_token_id:
                        break
            
            # Decode the generated text
            full_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            generated_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            
            # Return results with metadata
            return {
                "full_text": full_text,
                "generated_text": generated_text,
                "prompt": prompt,
                "generation_time": total_time,
                "tokens_per_second": (generated_tokens.shape[1] - input_len) / total_time if total_time > 0 else 0,
                "model_id": self.model_id,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error in state tracking generation: {e}")
            return {"error": str(e)}
    
    def get_hardware_info(self):
        """Get information about the hardware being used."""
        return self.hardware_info


# For testing
class {test_class_name}:
    """Test class for {model_type_upper} model."""
    
    @staticmethod
    def run_tests():
        """Run basic tests for the model."""
        model_id = "{default_model_id}"
        
        try:
            # Initialize with default settings
            skillset = {skillset_class_name}(
                model_id=model_id, 
                device="cpu",
                torch_dtype=torch.float32
            )
            
            # Test text generation
            result = skillset.generate(
                "State space models are efficient because",
                max_new_tokens=20,
                temperature=0.7
            )
            
            if "error" in result:
                print(f"Generation test failed: {result['error']}")
                return False
                
            print(f"Generation test complete. Generated text: {result['generated_text'][0][:100]}...")
            print(f"Generation time: {result['generation_time']:.2f}s")
            
            # Test embedding
            embed_result = skillset.embed("This is a test sentence for embedding.")
            
            if "error" in embed_result:
                print(f"Embedding test failed: {embed_result['error']}")
            else:
                print(f"Embedding test complete. Dimension: {embed_result['dimension']}")
            
            print("All tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"Test failed: {e}")
            return False


if __name__ == "__main__":
    # Run tests when the script is executed directly
    {test_class_name}.run_tests()