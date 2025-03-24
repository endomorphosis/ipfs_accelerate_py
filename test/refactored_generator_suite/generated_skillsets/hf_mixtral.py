#!/usr/bin/env python3
"""
Template for Mixture of Experts (MoE) models (Mixtral, Switch Transformer, etc.)

This template is designed for models that use the Mixture of Experts architecture,
where multiple specialized "expert" networks are conditionally activated
based on the input.
"""

import os
import time
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from transformers import Mixtral, {processor_class_name}
    import transformers
except ImportError:
    raise ImportError(
        "The transformers package is required to use this model. "
        "Please install it with `pip install transformers`."
    )

logger = logging.getLogger(__name__)

class Mixtral:
    """
    Skillset for MIXTRAL - a Mixture of Experts (MoE) language model
    that utilizes conditional computation for efficient scaling.
    """
    
    def __init__(self, model_id: str = "mixtral-base", device: str = "cpu", **kwargs):
        """
        Initialize the MIXTRAL model.
        
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
            "moe_specific": {
                "num_experts": None,
                "num_experts_per_token": None,
                "expert_capacity": None
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
            
            # Set up quantization parameters if low memory mode is enabled
            quantization_config = None
            if self.low_memory_mode:
                import transformers
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            
            # Device-specific initialization
            # Device-specific initialization will be added automatically
            
            # Load tokenizer
            self.tokenizer = {processor_class_name}.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine MoE-specific parameters by inspecting the model
            if hasattr(self.model.config, "num_experts"):
                self.hardware_info["moe_specific"]["num_experts"] = self.model.config.num_experts
            if hasattr(self.model.config, "num_experts_per_token"):
                self.hardware_info["moe_specific"]["num_experts_per_token"] = self.model.config.num_experts_per_token
            if hasattr(self.model.config, "expert_capacity"):
                self.hardware_info["moe_specific"]["expert_capacity"] = self.model.config.expert_capacity
            
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
            hidden_states = outputs.hidden_states[-1]
            
            # Apply pooling
            if pooling == "cls":
                embeddings = hidden_states[:, 0].cpu().numpy()  # [CLS] token
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
    
    def chat(self, messages, **kwargs):
        """
        Generate chat completions based on messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional chat parameters
            
        Returns:
            Dictionary containing generated response
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Check if the model supports chat
            if not hasattr(self.tokenizer, "apply_chat_template"):
                logger.warning("This model may not support chat format. Using basic text generation instead.")
                # Extract the last user message as the prompt
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                if user_messages:
                    return self.generate(user_messages[-1]["content"], **kwargs)
                else:
                    return {"error": "No user message found in chat history"}
            
            # Format chat messages using the tokenizer's chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate response
            result = self.generate(prompt, **kwargs)
            
            # Process the result to extract only the assistant's response
            if "full_text" in result:
                # Try to extract just the assistant's message
                full_text = result["full_text"][0]
                
                # Add chat-specific metadata
                result["messages"] = messages
                result["response_role"] = "assistant"
                
                return result
            else:
                return result
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return {"error": str(e)}
    
    def get_expert_routing(self, text):
        """
        Analyze expert routing patterns for given input.
        (This is a MoE-specific feature that shows which experts are activated.)
        
        Args:
            text: Text input to analyze
            
        Returns:
            Dictionary containing expert routing information
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Check if the model has router layers
            has_routers = False
            for name, module in self.model.named_modules():
                if "router" in name.lower() or "expert" in name.lower() or "moe" in name.lower():
                    has_routers = True
                    break
            
            if not has_routers:
                return {"error": "This model does not appear to have MoE routing layers"}
            
            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Setup hooks to capture routing decisions
            routing_info = []
            
            def hook_fn(module, input, output):
                # This is a simplified version - actual implementation would depend on the model
                if hasattr(output, "router_probs"):
                    router_probs = output.router_probs.cpu().detach().numpy()
                    routing_info.append({
                        "layer": module.__class__.__name__,
                        "router_probabilities": router_probs.tolist()
                    })
            
            # Register hooks on router layers
            hooks = []
            for name, module in self.model.named_modules():
                if "router" in name.lower() or "expert" in name.lower() or "moe" in name.lower():
                    hooks.append(module.register_forward_hook(hook_fn))
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Return expert routing information
            return {
                "text": text,
                "routing_info": routing_info,
                "num_experts": self.hardware_info["moe_specific"]["num_experts"],
                "num_experts_per_token": self.hardware_info["moe_specific"]["num_experts_per_token"],
                "expert_capacity": self.hardware_info["moe_specific"]["expert_capacity"],
                "model_id": self.model_id
            }
            
        except Exception as e:
            logger.error(f"Error analyzing expert routing: {e}")
            return {"error": f"Could not analyze expert routing: {str(e)}"}
    
    def get_hardware_info(self):
        """Get information about the hardware being used."""
        return self.hardware_info


# For testing
class Mixtral:
    """Test class for MIXTRAL model."""
    
    @staticmethod
    def run_tests():
        """Run basic tests for the model."""
        model_id = "mixtral-base"
        
        try:
            # Initialize with low memory mode for testing
            skillset = {skillset_class_name}(
                model_id=model_id, 
                device="cpu",
                low_memory_mode=True,
                torch_dtype=torch.float32
            )
            
            # Test text generation
            result = skillset.generate(
                "Explain the concept of mixture of experts in machine learning in one paragraph:",
                max_new_tokens=50,
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