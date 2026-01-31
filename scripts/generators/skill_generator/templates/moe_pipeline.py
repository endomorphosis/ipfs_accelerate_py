#!/usr/bin/env python3
"""
Mixture-of-Experts Pipeline Template for IPFS Accelerate Python.

This module implements a pipeline template for Mixture-of-Experts models like
Mixtral, Switch Transformers, etc. It handles MoE-specific processing and 
optimizations for efficient sparse activation.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipelineTemplate


class MoEPipelineTemplate(BasePipelineTemplate):
    """Template for Mixture-of-Experts model pipelines."""
    
    def __init__(self):
        """Initialize the MoE pipeline template."""
        super().__init__()
        self.pipeline_type = "moe"
        self.input_type = "text"
        self.output_type = "text"
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = True
        self.max_batch_size = 1  # MoE models are memory-intensive
    
    def get_import_statements(self) -> str:
        """Get MoE pipeline import statements."""
        return """
# MoE pipeline imports
import os
import json
import numpy as np
from typing import List, Dict, Union, Any, Optional, Tuple
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get MoE preprocessing code for specific task types."""
        if task_type == "text_generation":
            return """
# Preprocess for MoE text generation
# Parse input
if isinstance(text, dict):
    # Advanced input with parameters
    if "prompt" in text:
        prompt = text["prompt"]
    else:
        prompt = text.get("text", "")
    
    # Get generation parameters
    max_new_tokens = text.get("max_new_tokens", 128)
    temperature = text.get("temperature", 0.7)
    top_p = text.get("top_p", 0.9)
    top_k = text.get("top_k", 50)
    repetition_penalty = text.get("repetition_penalty", 1.0)
    do_sample = text.get("do_sample", True)
    
    # MoE-specific parameters
    num_active_experts = text.get("num_active_experts", None)  # Number of experts to activate per token
    expert_routing = text.get("expert_routing", None)  # Expert routing strategy
    
elif isinstance(text, str):
    # Simple prompt
    prompt = text
    max_new_tokens = 128
    temperature = 0.7
    top_p = 0.9
    top_k = 50
    repetition_penalty = 1.0
    do_sample = True
    
    # Default MoE parameters (using model's default settings)
    num_active_experts = None
    expert_routing = None
    
else:
    # Default fallback
    prompt = "Hello, I am a Mixture-of-Experts language model."
    max_new_tokens = 128
    temperature = 0.7
    top_p = 0.9
    top_k = 50
    repetition_penalty = 1.0
    do_sample = True
    
    # Default MoE parameters
    num_active_experts = None
    expert_routing = None

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Prepare generation parameters
generation_config = {
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "repetition_penalty": repetition_penalty,
    "do_sample": do_sample
}

# Add MoE-specific parameters if provided
if num_active_experts is not None:
    # This would be used in model-specific ways
    generation_config["num_active_experts"] = num_active_experts
    
if expert_routing is not None:
    # This would be used in model-specific ways
    generation_config["expert_routing"] = expert_routing

# Merge with any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in generation_config:
        generation_config[param_name] = param_value
"""
        elif task_type == "text_classification":
            return """
# Preprocess for MoE text classification
# Parse input
if isinstance(text, dict):
    if "text" in text:
        input_text = text["text"]
    else:
        input_text = str(text)
elif isinstance(text, str):
    input_text = text
elif isinstance(text, list) and all(isinstance(item, str) for item in text):
    # List of strings for batch processing
    input_text = text
else:
    # Default fallback
    input_text = "Hello, I am a Mixture-of-Experts language model."

# MoE-specific parameters
if isinstance(text, dict):
    num_active_experts = text.get("num_active_experts", None)
    expert_routing = text.get("expert_routing", None)
else:
    num_active_experts = None
    expert_routing = None

# Tokenize the input
inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Add MoE-specific parameters if provided
moe_config = {}
if num_active_experts is not None:
    moe_config["num_active_experts"] = num_active_experts
    
if expert_routing is not None:
    moe_config["expert_routing"] = expert_routing

# Add any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in moe_config:
        moe_config[param_name] = param_value
"""
        else:
            # Default preprocessing for other MoE tasks
            return """
# Default preprocessing for MoE models
# Parse input
if isinstance(text, dict):
    if "text" in text:
        input_text = text["text"]
    elif "prompt" in text:
        input_text = text["prompt"]
    else:
        input_text = str(text)
elif isinstance(text, str):
    input_text = text
elif isinstance(text, list) and all(isinstance(item, str) for item in text):
    # List of strings for batch processing
    input_text = text
else:
    # Default fallback
    input_text = "Hello, I am a Mixture-of-Experts language model."

# MoE-specific parameters
if isinstance(text, dict):
    num_active_experts = text.get("num_active_experts", None)
    expert_routing = text.get("expert_routing", None)
    task_specific_params = {k: v for k, v in text.items() 
                           if k not in ["text", "prompt", "num_active_experts", "expert_routing"]}
else:
    num_active_experts = None
    expert_routing = None
    task_specific_params = {}

# Tokenize the input
inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Prepare configuration
config = task_specific_params.copy()

# Add MoE-specific parameters if provided
if num_active_experts is not None:
    config["num_active_experts"] = num_active_experts
    
if expert_routing is not None:
    config["expert_routing"] = expert_routing

# Add any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in config:
        config[param_name] = param_value
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get MoE postprocessing code for specific task types."""
        if task_type == "text_generation":
            return """
# Process outputs from MoE text generation
with self.torch.no_grad():
    # Run generation with the configured parameters
    output_ids = endpoint.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        **generation_config
    )
    
    # Decode the generated text
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    # Try to extract expert routing information if available
    expert_usage = {}
    try:
        # For models that expose this information
        if hasattr(endpoint, "last_expert_selection") and endpoint.last_expert_selection is not None:
            expert_selection = endpoint.last_expert_selection
            expert_usage["expert_selection"] = expert_selection.cpu().numpy().tolist()
            
        # For models that return router logits
        if hasattr(endpoint, "last_router_logits") and endpoint.last_router_logits is not None:
            router_logits = endpoint.last_router_logits
            expert_usage["router_logits"] = router_logits.cpu().numpy().tolist()
    except:
        # If unable to extract expert information
        pass
    
    # Create results dictionary
    results = {
        "generated_text": generated_texts[0] if generated_texts else "",
        "all_texts": generated_texts,
        "expert_usage": expert_usage
    }
    
    # Add generation parameters used
    results["parameters"] = {
        "max_new_tokens": generation_config.get("max_new_tokens", 128),
        "temperature": generation_config.get("temperature", 0.7),
        "top_p": generation_config.get("top_p", 0.9),
        "top_k": generation_config.get("top_k", 50),
        "repetition_penalty": generation_config.get("repetition_penalty", 1.0),
        "do_sample": generation_config.get("do_sample", True)
    }
    
    # Add MoE-specific parameters if used
    if "num_active_experts" in generation_config:
        results["parameters"]["num_active_experts"] = generation_config["num_active_experts"]
        
    if "expert_routing" in generation_config:
        results["parameters"]["expert_routing"] = generation_config["expert_routing"]
"""
        elif task_type == "text_classification":
            return """
# Process outputs from MoE text classification
with self.torch.no_grad():
    # Run classification
    outputs = endpoint(**inputs)
    
    # Get logits
    logits = outputs.logits
    
    # Apply softmax to get probabilities
    probs = self.torch.nn.functional.softmax(logits, dim=-1)
    
    # Convert to numpy and then to lists
    probs_list = probs.cpu().numpy().tolist()
    
    # Get the predicted class indices
    predicted_class_ids = self.torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    
    # Try to map to class labels if available
    predicted_labels = []
    if hasattr(endpoint.config, "id2label"):
        for class_id in predicted_class_ids:
            label = endpoint.config.id2label.get(class_id, f"CLASS_{class_id}")
            predicted_labels.append(label)
    else:
        predicted_labels = [f"CLASS_{class_id}" for class_id in predicted_class_ids]
    
    # Try to extract expert routing information if available
    expert_usage = {}
    try:
        # For models that expose this information
        if hasattr(endpoint, "last_expert_selection") and endpoint.last_expert_selection is not None:
            expert_selection = endpoint.last_expert_selection
            expert_usage["expert_selection"] = expert_selection.cpu().numpy().tolist()
            
        # For models that return router logits
        if hasattr(endpoint, "last_router_logits") and endpoint.last_router_logits is not None:
            router_logits = endpoint.last_router_logits
            expert_usage["router_logits"] = router_logits.cpu().numpy().tolist()
    except:
        # If unable to extract expert information
        pass
    
    # Create results dictionary
    results = {
        "predictions": [],
        "expert_usage": expert_usage
    }
    
    # Format predictions with labels and scores
    for i, (label, probs) in enumerate(zip(predicted_labels, probs_list)):
        prediction = {
            "label": label,
            "score": max(probs),
            "all_scores": probs
        }
        results["predictions"].append(prediction)
"""
        else:
            # Default postprocessing for other MoE tasks
            return """
# Default postprocessing for MoE models
with self.torch.no_grad():
    # Run the model with inputs
    outputs = endpoint(**inputs)
    
    # Process based on output type
    if hasattr(outputs, "logits"):
        # Classification-like output
        logits = outputs.logits
        
        # Check dimensionality to determine processing
        if len(logits.shape) >= 2 and logits.shape[-1] > 1:
            # Multi-class classification
            probs = self.torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_ids = self.torch.argmax(logits, dim=-1)
            
            # Convert to lists for API response
            probs_list = probs.cpu().numpy().tolist()
            predicted_ids = predicted_class_ids.cpu().numpy().tolist()
            
            # Create results
            results = {
                "logits": logits.cpu().numpy().tolist(),
                "probabilities": probs_list,
                "predicted_ids": predicted_ids
            }
        else:
            # Scalar or sequence output
            results = {
                "logits": logits.cpu().numpy().tolist()
            }
    elif hasattr(outputs, "last_hidden_state"):
        # Embedding-like output
        hidden_states = outputs.last_hidden_state
        
        # Use mean pooling as a simple aggregation
        embeddings = hidden_states.mean(dim=1)
        
        # Convert to lists for API response
        results = {
            "embeddings": embeddings.cpu().numpy().tolist()
        }
    else:
        # Generic output handling
        results = {
            "outputs": str(outputs)
        }
    
    # Try to extract expert routing information if available
    expert_usage = {}
    try:
        # For models that expose this information
        if hasattr(endpoint, "last_expert_selection") and endpoint.last_expert_selection is not None:
            expert_selection = endpoint.last_expert_selection
            expert_usage["expert_selection"] = expert_selection.cpu().numpy().tolist()
            
        # For models that return router logits
        if hasattr(endpoint, "last_router_logits") and endpoint.last_router_logits is not None:
            router_logits = endpoint.last_router_logits
            expert_usage["router_logits"] = router_logits.cpu().numpy().tolist()
    except:
        # If unable to extract expert information
        pass
    
    # Add expert usage to results
    results["expert_usage"] = expert_usage
    
    # Add parameter information
    results["moe_config"] = {
        "num_active_experts": config.get("num_active_experts", "default"),
        "expert_routing": config.get("expert_routing", "default")
    }
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get MoE result formatting code for specific task types."""
        if task_type == "text_generation":
            return """
# Format results for MoE text generation
return {
    "success": True,
    "moe_generation": {
        "text": results["generated_text"],
        "all_texts": results["all_texts"],
        "parameters": results["parameters"],
        "expert_info": results.get("expert_usage", {})
    },
    "device": device,
    "hardware": hardware_label
}
"""
        elif task_type == "text_classification":
            return """
# Format results for MoE text classification
return {
    "success": True,
    "moe_classification": {
        "predictions": results["predictions"],
        "expert_info": results.get("expert_usage", {})
    },
    "device": device,
    "hardware": hardware_label
}
"""
        else:
            # Default result formatting for MoE tasks
            return """
# Default format for MoE model results
return {
    "success": True,
    "moe_output": {
        "results": results,
        "moe_config": results.get("moe_config", {})
    },
    "device": device,
    "hardware": hardware_label
}
"""
    
    def get_mock_input_code(self) -> str:
        """Get MoE mock input code."""
        return """
# Mock MoE input
mock_input = {
    "prompt": "Write a short story about a robot learning to paint",
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_p": 0.92,
    "num_active_experts": 2,  # MoE-specific parameter
    "expert_routing": "tokens"  # MoE-specific parameter
}
"""
    
    def get_mock_output_code(self) -> str:
        """Get MoE mock output code."""
        return """
# Mock MoE output
num_tokens = 10
num_experts = 8  # Typical for MoE models

# Create mock router logits and expert selection
mock_router_logits = self.torch.randn((1, num_tokens, num_experts))
mock_expert_selection = self.torch.zeros((1, num_tokens, 2), dtype=self.torch.long)
# Randomly select 2 experts per token
for i in range(num_tokens):
    selected_experts = self.torch.randperm(num_experts)[:2]
    mock_expert_selection[0, i] = selected_experts

# Create appropriate mock output based on task
if "generation" in task_type:
    # Mock text generation output
    mock_output = type('MockMoEOutput', (), {})()
    mock_output.sequences = self.torch.randint(0, 50000, (1, num_tokens))
    # Add MoE specific attributes
    mock_output.last_expert_selection = mock_expert_selection
    mock_output.last_router_logits = mock_router_logits
    
elif "classification" in task_type:
    # Mock classification output
    mock_output = type('MockMoEOutput', (), {})()
    mock_output.logits = self.torch.randn((1, 3))  # 3 classes
    # Add MoE specific attributes
    mock_output.last_expert_selection = mock_expert_selection
    mock_output.last_router_logits = mock_router_logits
    
else:
    # Default mock output
    mock_output = type('MockMoEOutput', (), {})()
    mock_output.last_hidden_state = self.torch.randn((1, num_tokens, hidden_size))
    # Add MoE specific attributes
    mock_output.last_expert_selection = mock_expert_selection
    mock_output.last_router_logits = mock_router_logits

return mock_output
"""
    
    def get_pipeline_utilities(self) -> str:
        """Get MoE utility functions."""
        return """
# MoE pipeline utilities
def analyze_expert_usage(expert_selection, router_logits=None):
    \"\"\"Analyze expert usage patterns from model outputs.
    
    Args:
        expert_selection: Tensor of selected experts
        router_logits: Optional tensor of router logits
        
    Returns:
        Dictionary with expert usage statistics
    \"\"\"
    if not isinstance(expert_selection, list):
        # Convert to list if it's a tensor
        if hasattr(expert_selection, "cpu"):
            expert_selection = expert_selection.cpu().numpy().tolist()
    
    # Analyze which experts were selected most often
    expert_counts = {}
    for batch in expert_selection:
        for token in batch:
            for expert in token:
                expert_id = str(expert)
                if expert_id not in expert_counts:
                    expert_counts[expert_id] = 0
                expert_counts[expert_id] += 1
    
    # Sort experts by usage
    sorted_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "expert_counts": expert_counts,
        "top_experts": sorted_experts[:3],  # Top 3 most used experts
        "total_routing_decisions": sum(expert_counts.values())
    }

def extract_expert_patterns(expert_selection):
    \"\"\"Extract patterns in expert selection across tokens.
    
    Args:
        expert_selection: Tensor or list of selected experts
        
    Returns:
        Dictionary with pattern analysis
    \"\"\"
    if not isinstance(expert_selection, list):
        # Convert to list if it's a tensor
        if hasattr(expert_selection, "cpu"):
            expert_selection = expert_selection.cpu().numpy().tolist()
    
    # Look for common expert combinations
    expert_combos = {}
    for batch in expert_selection:
        for token in batch:
            # Sort the experts to treat [0,1] and [1,0] as the same combination
            combo = tuple(sorted(token))
            if combo not in expert_combos:
                expert_combos[combo] = 0
            expert_combos[combo] += 1
    
    # Sort combinations by frequency
    sorted_combos = sorted(expert_combos.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "expert_combinations": expert_combos,
        "top_combinations": sorted_combos[:3]  # Top 3 most common combinations
    }
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check MoE pipeline compatibility with architecture type."""
        # MoE pipeline is compatible with MoE-based architectures
        return arch_type in [
            "mixture-of-experts",
            "moe",
            "sparse"  # Some sparse models use MoE-like routing
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check MoE pipeline compatibility with task type."""
        # MoE pipeline is compatible with these tasks
        return task_type in [
            "text_generation",
            "text_classification",
            "feature_extraction",
            "question_answering",  # Many MoE models can do QA
            "summarization"        # Many MoE models can do summarization
        ]


# Define alias for compatibility with the verification script
MixOfExpertsPipelineTemplate = MoEPipelineTemplate