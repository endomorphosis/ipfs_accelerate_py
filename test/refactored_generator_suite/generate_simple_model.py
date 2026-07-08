#!/usr/bin/env python3
"""
Generate a simple reference model implementation using the simplified template.
"""

import os
import sys
import time
import string

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_model(model_type, output_dir=None):
    """Generate a model implementation from the simple template."""
    # Default output dir is test/skillset
    if output_dir is None:
        # Get the path to the refactored_generator_suite directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to 'test' directory and then into 'skillset'
        output_dir = os.path.normpath(os.path.join(current_dir, "../../test/skillset"))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to template
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "templates", "simple_reference_template.py")
    
    # Read template
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Architecture-specific settings
    architecture_settings = {
        # Encoder-only models
        'bert': {
            'architecture': 'encoder-only',
            'model_type_upper': 'BERT',
            'model_description': 'This is a bidirectional encoder model used for text embeddings.',
            'task_type': 'text_embedding',
            'task_class': 'MaskedLM',
            'automodel_class': 'self.transformers.AutoModel',
            'test_input': 'This is a test input for BERT.'
        },
        'roberta': {
            'architecture': 'encoder-only',
            'model_type_upper': 'RoBERTa',
            'model_description': 'This is an optimized BERT model with different pre-training.',
            'task_type': 'text_embedding',
            'task_class': 'MaskedLM',
            'automodel_class': 'self.transformers.AutoModel',
            'test_input': 'This is a test input for RoBERTa.'
        },
        'albert': {
            'architecture': 'encoder-only',
            'model_type_upper': 'ALBERT',
            'model_description': 'This is a light version of BERT with parameter sharing.',
            'task_type': 'text_embedding',
            'task_class': 'MaskedLM',
            'automodel_class': 'self.transformers.AutoModel',
            'test_input': 'This is a test input for ALBERT.'
        },
        'deberta': {
            'architecture': 'encoder-only',
            'model_type_upper': 'DeBERTa',
            'model_description': 'This is an enhanced BERT model with disentangled attention.',
            'task_type': 'text_embedding',
            'task_class': 'MaskedLM',
            'automodel_class': 'self.transformers.AutoModel',
            'test_input': 'This is a test input for DeBERTa.'
        },
        'distilbert': {
            'architecture': 'encoder-only',
            'model_type_upper': 'DistilBERT',
            'model_description': 'This is a distilled version of BERT that is smaller and faster.',
            'task_type': 'text_embedding',
            'task_class': 'MaskedLM',
            'automodel_class': 'self.transformers.AutoModel',
            'test_input': 'This is a test input for DistilBERT.'
        },
        'electra': {
            'architecture': 'encoder-only',
            'model_type_upper': 'ELECTRA',
            'model_description': 'This model uses replaced token detection instead of masked language modeling.',
            'task_type': 'text_embedding',
            'task_class': 'ForPreTraining',
            'automodel_class': 'self.transformers.AutoModel',
            'test_input': 'This is a test input for ELECTRA.'
        },
        'xlm-roberta': {
            'architecture': 'encoder-only',
            'model_type_upper': 'XLM-RoBERTa',
            'model_description': 'This is a multilingual version of RoBERTa trained on 100 languages.',
            'task_type': 'text_embedding',
            'task_class': 'MaskedLM',
            'automodel_class': 'self.transformers.AutoModel',
            'test_input': 'This is a test input for XLM-RoBERTa.'
        },
        
        # Decoder-only models
        'gpt2': {
            'architecture': 'decoder-only',
            'model_type_upper': 'GPT2',
            'model_description': 'This is an autoregressive decoder model used for text generation.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for GPT2.'
        },
        'llama': {
            'architecture': 'decoder-only',
            'model_type_upper': 'LLaMA',
            'model_description': 'This is an efficient decoder-only language model.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for LLaMA.'
        },
        'mistral': {
            'architecture': 'decoder-only',
            'model_type_upper': 'Mistral',
            'model_description': 'This is an efficient decoder-only language model with sliding window attention.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for Mistral.'
        },
        'bloom': {
            'architecture': 'decoder-only',
            'model_type_upper': 'BLOOM',
            'model_description': 'This is a multilingual decoder-only model for text generation.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for BLOOM.'
        },
        'codellama': {
            'architecture': 'decoder-only',
            'model_type_upper': 'CodeLlama',
            'model_description': 'This is a specialized code generation model based on LLaMA architecture.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'def factorial(n):'
        },
        'falcon': {
            'architecture': 'decoder-only',
            'model_type_upper': 'Falcon',
            'model_description': 'This is an efficient decoder-only language model for text generation.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for Falcon.'
        },
        'gemma': {
            'architecture': 'decoder-only',
            'model_type_upper': 'Gemma',
            'model_description': 'This is a lightweight decoder-only language model from Google.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for Gemma.'
        },
        'phi': {
            'architecture': 'decoder-only',
            'model_type_upper': 'Phi',
            'model_description': 'This is a small but powerful decoder-only language model from Microsoft.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for Phi.'
        },
        
        # Encoder-decoder models
        't5': {
            'architecture': 'encoder-decoder',
            'model_type_upper': 'T5',
            'model_description': 'This is an encoder-decoder model used for text-to-text generation.',
            'task_type': 'text2text_generation',
            'task_class': 'Seq2SeqLM',
            'automodel_class': 'self.transformers.AutoModelForSeq2SeqLM',
            'test_input': 'Translate to French: Hello, how are you?'
        },
        'bart': {
            'architecture': 'encoder-decoder',
            'model_type_upper': 'BART',
            'model_description': 'This is an encoder-decoder model for sequence-to-sequence tasks.',
            'task_type': 'text2text_generation',
            'task_class': 'Seq2SeqLM',
            'automodel_class': 'self.transformers.AutoModelForSeq2SeqLM',
            'test_input': 'Summarize: The quick brown fox jumps over the lazy dog. The dog was not amused.'
        },
        'mbart': {
            'architecture': 'encoder-decoder',
            'model_type_upper': 'mBART',
            'model_description': 'This is a multilingual encoder-decoder model for translation.',
            'task_type': 'text2text_generation',
            'task_class': 'Seq2SeqLM',
            'automodel_class': 'self.transformers.AutoModelForSeq2SeqLM',
            'test_input': 'Translate to Spanish: Hello, how are you today?'
        },
        'flan-t5': {
            'architecture': 'encoder-decoder',
            'model_type_upper': 'FLAN-T5',
            'model_description': 'This is an instruction-tuned version of T5.',
            'task_type': 'text2text_generation',
            'task_class': 'Seq2SeqLM',
            'automodel_class': 'self.transformers.AutoModelForSeq2SeqLM',
            'test_input': 'Explain what is machine learning:'
        },
        'pegasus': {
            'architecture': 'encoder-decoder',
            'model_type_upper': 'Pegasus',
            'model_description': 'This is an encoder-decoder model optimized for text summarization.',
            'task_type': 'text2text_generation',
            'task_class': 'Seq2SeqLM',
            'automodel_class': 'self.transformers.AutoModelForSeq2SeqLM',
            'test_input': 'Summarize: The researchers discovered a new species of fish in the deep ocean that can glow in the dark and survive extreme pressure.'
        },
        
        # Vision models
        'vit': {
            'architecture': 'vision',
            'model_type_upper': 'ViT',
            'model_description': 'This is a vision transformer model used for image classification.',
            'task_type': 'image_classification',
            'task_class': 'ImageClassification',
            'automodel_class': 'self.transformers.AutoModelForImageClassification',
            'test_input': '[IMAGE TENSOR]'
        },
        'beit': {
            'architecture': 'vision',
            'model_type_upper': 'BEiT',
            'model_description': 'This is a BERT-style pre-trained vision transformer.',
            'task_type': 'image_classification',
            'task_class': 'ImageClassification',
            'automodel_class': 'self.transformers.AutoModelForImageClassification',
            'test_input': '[IMAGE TENSOR]'
        },
        'convnext': {
            'architecture': 'vision',
            'model_type_upper': 'ConvNeXt',
            'model_description': 'This is a convolutional neural network for image classification.',
            'task_type': 'image_classification',
            'task_class': 'ImageClassification',
            'automodel_class': 'self.transformers.AutoModelForImageClassification',
            'test_input': '[IMAGE TENSOR]'
        },
        'deit': {
            'architecture': 'vision',
            'model_type_upper': 'DeiT',
            'model_description': 'This is a data-efficient image transformer.',
            'task_type': 'image_classification',
            'task_class': 'ImageClassification',
            'automodel_class': 'self.transformers.AutoModelForImageClassification',
            'test_input': '[IMAGE TENSOR]'
        },
        'swin': {
            'architecture': 'vision',
            'model_type_upper': 'Swin',
            'model_description': 'This is a hierarchical vision transformer with shifted windows.',
            'task_type': 'image_classification',
            'task_class': 'ImageClassification',
            'automodel_class': 'self.transformers.AutoModelForImageClassification',
            'test_input': '[IMAGE TENSOR]'
        },
        
        # Vision-text models
        'clip': {
            'architecture': 'vision-text',
            'model_type_upper': 'CLIP',
            'model_description': 'This is a vision-text model used for image-text matching.',
            'task_type': 'vision_text_dual_encoding',
            'task_class': 'VisionTextDualEncoder',
            'automodel_class': 'self.transformers.CLIPModel',
            'test_input': 'A photograph of a cat.'
        },
        'blip': {
            'architecture': 'vision-text',
            'model_type_upper': 'BLIP',
            'model_description': 'This is a model for vision-language understanding and generation.',
            'task_type': 'vision_text_dual_encoding',
            'task_class': 'VisionTextDualEncoder',
            'automodel_class': 'self.transformers.BlipModel',
            'test_input': 'A photo of a dog running in a park.'
        },
        'blip-2': {
            'architecture': 'vision-text',
            'model_type_upper': 'BLIP-2',
            'model_description': 'This is a vision-language model for understanding and generation.',
            'task_type': 'vision_text_dual_encoding',
            'task_class': 'VisionTextDualEncoder',
            'automodel_class': 'self.transformers.Blip2Model',
            'test_input': 'A photo of mountains at sunset.'
        },
        'git': {
            'architecture': 'vision-text',
            'model_type_upper': 'GIT',
            'model_description': 'This is a generative image-to-text model.',
            'task_type': 'image_to_text',
            'task_class': 'VisionTextDualEncoder',
            'automodel_class': 'self.transformers.GitModel',
            'test_input': '[IMAGE TENSOR]'
        },
        'llava': {
            'architecture': 'vision-text',
            'model_type_upper': 'LLaVA',
            'model_description': 'This is a large language and vision assistant model.',
            'task_type': 'vision_text_dual_encoding',
            'task_class': 'VisionTextDualEncoder',
            'automodel_class': 'self.transformers.LlavaModel',
            'test_input': '[IMAGE TENSOR] What is shown in this image?'
        },
        
        # Speech models
        'whisper': {
            'architecture': 'speech',
            'model_type_upper': 'Whisper',
            'model_description': 'This is a speech-to-text model used for automatic speech recognition.',
            'task_type': 'speech_recognition',
            'task_class': 'SpeechSeq2Seq',
            'automodel_class': 'self.transformers.AutoModelForSpeechSeq2Seq',
            'test_input': '[AUDIO TENSOR]'
        },
        'hubert': {
            'architecture': 'speech',
            'model_type_upper': 'Hubert',
            'model_description': 'This is a speech model for audio representation learning.',
            'task_type': 'audio_classification',
            'task_class': 'AudioClassification',
            'automodel_class': 'self.transformers.HubertModel',
            'test_input': '[AUDIO TENSOR]'
        },
        'wav2vec2': {
            'architecture': 'speech',
            'model_type_upper': 'Wav2Vec2',
            'model_description': 'This is a model for speech recognition and audio representation.',
            'task_type': 'audio_classification',
            'task_class': 'AudioClassification',
            'automodel_class': 'self.transformers.Wav2Vec2Model',
            'test_input': '[AUDIO TENSOR]'
        },
        'bark': {
            'architecture': 'speech',
            'model_type_upper': 'Bark',
            'model_description': 'This is a text-to-audio model for speech generation.',
            'task_type': 'text_to_audio',
            'task_class': 'TextToAudio',
            'automodel_class': 'self.transformers.BarkModel',
            'test_input': 'Hello, how are you doing today?'
        },
        
        # Text-to-image models
        'stable-diffusion': {
            'architecture': 'text-to-image',
            'model_type_upper': 'StableDiffusion',
            'model_description': 'This is a text-to-image diffusion model.',
            'task_type': 'text_to_image',
            'task_class': 'DiffusionPipeline',
            'automodel_class': 'self.transformers.StableDiffusionPipeline',
            'test_input': 'A beautiful sunset over mountains'
        },
        
        # Mixture of experts models
        'mixtral': {
            'architecture': 'mixture-of-experts',
            'model_type_upper': 'Mixtral',
            'model_description': 'This is a mixture of experts model for text generation.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for Mixtral.'
        },
        
        # State space models
        'mamba': {
            'architecture': 'state-space-model',
            'model_type_upper': 'Mamba',
            'model_description': 'This is a state space model for efficient text generation.',
            'task_type': 'text_generation',
            'task_class': 'CausalLM',
            'automodel_class': 'self.transformers.AutoModelForCausalLM',
            'test_input': 'This is a test input for Mamba.'
        }
    }
    
    # Default settings for unknown models
    default_settings = {
        'architecture': 'encoder-only',
        'model_type_upper': model_type.upper(),
        'model_description': f'This is a HuggingFace {model_type.upper()} model.',
        'task_type': 'text_embedding',
        'task_class': 'MaskedLM',
        'automodel_class': 'self.transformers.AutoModel',
        'test_input': f'This is a test input for {model_type}.'
    }
    
    # Get settings for the model
    if model_type not in architecture_settings:
        print(f"Model type {model_type} not recognized. Using default settings.")
        settings = default_settings
    else:
        settings = architecture_settings[model_type]
    
    # Replace placeholders in template
    model_code = template_content
    for key, value in settings.items():
        placeholder = '{' + key + '}'
        model_code = model_code.replace(placeholder, value)
    
    # Replace model_type (handle hyphens in class names)
    # Python class names can't have hyphens, so convert them to underscores
    class_name = model_type.replace('-', '_')
    model_code = model_code.replace('{model_type}', class_name)
    
    # Write to file
    output_file = os.path.join(output_dir, f"hf_{model_type}.py")
    with open(output_file, 'w') as f:
        f.write(model_code)
    
    print(f"Generated model implementation: {output_file}")
    return output_file

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a model implementation using a simple template")
    parser.add_argument("model_type", help="Model type to generate (e.g., bert, gpt2)")
    parser.add_argument("--output-dir", "-o", help="Directory to write generated model to")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite if file exists")
    
    args = parser.parse_args()
    
    try:
        output_file = generate_model(args.model_type, args.output_dir)
        print(f"Successfully generated: {output_file}")
        return 0
    except Exception as e:
        print(f"Error generating model: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())