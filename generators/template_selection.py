"""
Template selection logic for the Hugging Face Transformers skillset generator.
This module selects the most appropriate template based on model analysis.
"""

import os
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Model family patterns for template matching
MODEL_FAMILY_PATTERNS = {
    # Embedding models
    'hf_embedding_template.py': [
        'bert', 'roberta', 'distilbert', 'mpnet', 'albert', 'xlm', 'electra', 
        'deberta', 'camembert', 'xlnet', 'ernie', 'luke', 'flaubert', 'sentence'
    ],
    # Text generation models
    'hf_text_generation_template.py': [
        't5', 'gpt', 'llama', 'opt', 'bloom', 'mistral', 'falcon', 'phi', 
        'mixtral', 'gemma', 'bart', 'pegasus', 'mt5', 'mbart', 'longt5', 'flan',
        'palm', 'qwen', 'mpt', 'starcoder', 'codellama', 'dolphin'
    ],
    # Vision models
    'hf_vision_template.py': [
        'vit', 'clip', 'deit', 'beit', 'dino', 'swin', 'detr', 'segformer',
        'convnext', 'resnet', 'yolos', 'maskformer', 'owlvit', 'dinov2',
        'efficientnet', 'beit', 'siglip', 'sam', 'yolo'
    ],
    # Audio models
    'hf_audio_template.py': [
        'whisper', 'wav2vec2', 'hubert', 'unispeech', 'wavlm', 'speecht5',
        'mctct', 'musicgen', 'encodec', 'audio', 'clap', 'bark'
    ],
    # Multimodal models
    'hf_multimodal_template.py': [
        'llava', 'blip', 'flava', 'git', 'pali', 'fuyu', 'instructblip',
        'siglip', 'flamingo', 'idefics', 'kosmos', 'vision_encoder_decoder'
    ]
}

# Method name patterns that indicate model type
METHOD_PATTERNS = {
    'hf_embedding_template.py': ['embed', 'encode', 'get_embedding', 'get_sentence_embedding', 'get_vector'],
    'hf_text_generation_template.py': ['generate', 'complete', 'predict_next', 'chat', 'summarize', 'predict'],
    'hf_vision_template.py': ['image_to_text', 'image_classification', 'object_detection', 'segmentation', 'detect'],
    'hf_audio_template.py': ['transcribe', 'audio_classification', 'speech_recognition', 'audio_to_text'],
    'hf_multimodal_template.py': ['vision_to_text', 'image_text_generation', 'visual_question_answering', 'describe_image']
}

# Hardware requirements that might indicate model type
HARDWARE_INDICATORS = {
    'large_memory': {
        'template': 'hf_text_generation_template.py',
        'threshold': 2000  # MB
    },
    'multiple_modalities': {
        'template': 'hf_multimodal_template.py',
        'indicators': ['image', 'text']
    },
    'specialized_hardware': {
        'template': 'hf_multimodal_template.py',
        'incompatible_platforms': ['mps', 'webnn']
    }
}

def load_model_metadata(metadata_file):
    """Load model metadata from JSON file"""
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning(f"Could not load metadata from {metadata_file}")
        return {}

def match_by_name(model_name):
    """Match model to a template based on name patterns"""
    # Normalize model name
    normalized_name = model_name.lower().split('-')[0]  # Handle variants like 'bert-base-uncased'
    
    # Check for explicit matches in each template family
    for template, patterns in MODEL_FAMILY_PATTERNS.items():
        for pattern in patterns:
            if pattern in normalized_name:
                logger.info(f"Matched {model_name} to {template} by name pattern '{pattern}'")
                return template
    
    return None

def match_by_methods(methods):
    """Match model to a template based on method patterns"""
    if not methods:
        return None
        
    # Count matches for each template
    template_scores = defaultdict(int)
    
    for method_name in methods.keys():
        for template, patterns in METHOD_PATTERNS.items():
            if any(pattern in method_name for pattern in patterns):
                template_scores[template] += 1
    
    # Return template with highest score, if any
    if template_scores:
        best_template = max(template_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Matched template {best_template} based on method patterns with score {template_scores[best_template]}")
        return best_template
    
    return None

def match_by_hardware(hardware_support):
    """Match model to a template based on hardware requirements"""
    if not hardware_support:
        return None
        
    # Check for large memory requirements
    for hw_type in ['cuda', 'rocm']:
        if hw_type in hardware_support:
            memory_usage = hardware_support[hw_type].get('memory_usage', {}).get('peak', 0)
            if memory_usage > HARDWARE_INDICATORS['large_memory']['threshold']:
                logger.info(f"Matched large memory model ({memory_usage}MB) to text generation template")
                return HARDWARE_INDICATORS['large_memory']['template']
    
    # Check for platform incompatibilities that suggest multimodal models
    unsupported_platforms = []
    for platform in ['mps', 'webnn', 'webgpu']:
        if platform in hardware_support and not hardware_support[platform].get('supported', True):
            unsupported_platforms.append(platform)
    
    if set(unsupported_platforms).intersection(set(HARDWARE_INDICATORS['specialized_hardware']['incompatible_platforms'])):
        logger.info(f"Matched model with platform incompatibilities {unsupported_platforms} to multimodal template")
        return HARDWARE_INDICATORS['specialized_hardware']['template']
    
    return None

def select_template_for_model(model_name, requirements, model_db_path=None):
    """
    Select the most appropriate template for the model
    
    Args:
        model_name: The model name (e.g., 'bert', 't5')
        requirements: Dict containing model requirements from test analysis
        model_db_path: Optional path to model metadata database
        
    Returns:
        String with the template filename to use
    """
    logger.info(f"Selecting template for model: {model_name}")
    
    # Try to match by model name first (most reliable)
    template = match_by_name(model_name)
    if template:
        return template
    
    # Check model database if provided
    if model_db_path and os.path.exists(model_db_path):
        model_metadata = load_model_metadata(model_db_path)
        if model_name in model_metadata:
            model_family = model_metadata[model_name].get('family')
            if model_family:
                for template, families in MODEL_FAMILY_PATTERNS.items():
                    if any(family in model_family.lower() for family in families):
                        logger.info(f"Matched {model_name} to {template} via metadata family: {model_family}")
                        return template
    
    # Try to match by methods
    methods = requirements.get('methods', {})
    template = match_by_methods(methods)
    if template:
        return template
    
    # Try to match by hardware requirements
    hardware_support = requirements.get('hardware_support', {})
    template = match_by_hardware(hardware_support)
    if template:
        return template
    
    # Default to combination template for unknown models
    logger.info(f"No specific match found for {model_name}, using combination template")
    return "hf_combination_template.py"

def get_template_combinations(model_name, requirements):
    """
    For complex models that fit multiple categories, return a list of templates to combine
    
    Args:
        model_name: The model name
        requirements: Dict containing model requirements
        
    Returns:
        List of template names to combine
    """
    templates = []
    matches = []
    
    # Check for multimodal capabilities first
    multimodal_indicators = ['image', 'text', 'audio', 'vision', 'visual']
    methods = requirements.get('methods', {})
    method_inputs = []
    
    # Extract input types from method parameters
    for method_info in methods.values():
        for param in method_info.get('required_parameters', []):
            method_inputs.append(param)
    
    # Count multimodal indicators
    modality_count = sum(1 for indicator in multimodal_indicators if any(indicator in input_name for input_name in method_inputs))
    
    if modality_count >= 2:
        templates.append('hf_multimodal_template.py')
        matches.append(('multimodal', modality_count))
    
    # Check for other capabilities
    for template, patterns in METHOD_PATTERNS.items():
        if template in templates:
            continue
            
        match_count = 0
        for method_name in methods.keys():
            if any(pattern in method_name for pattern in patterns):
                match_count += 1
        
        if match_count > 0:
            templates.append(template)
            matches.append((template, match_count))
    
    # Sort by match strength
    templates = [template for template, _ in sorted(matches, key=lambda x: x[1], reverse=True)]
    
    if not templates:
        # Fallback to basic name matching
        template = match_by_name(model_name)
        if template:
            templates = [template]
        else:
            # Ultimate fallback
            templates = ['hf_template.py']
    
    logger.info(f"Selected template combinations for {model_name}: {templates}")
    return templates