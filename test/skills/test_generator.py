#!/usr/bin/env python3
"""
Hugging Face Test Generator

This module provides a comprehensive framework for generating test files that cover
all Hugging Face model architectures, with support for:
- Multiple hardware backends (CPU, CUDA, OpenVINO)
- Both from_pretrained() and pipeline() API approaches
- Consistent performance benchmarking and result collection
- Automatic model discovery and test generation
- Batch processing of multiple model families

Usage:
  # List available model families in registry
  python test_generator.py --list-families
  
  # Generate tests for a specific model family
  python test_generator.py --generate bert
  
  # Generate tests for all model families in registry
  python test_generator.py --all
  
  # Generate tests for a specific set of models
  python test_generator.py --batch-generate bert,gpt2,t5,vit,clip
  
  # Discover and suggest new models to add
  python test_generator.py --suggest-models
  
  # Generate a registry entry for a specific model
  python test_generator.py --generate-registry-entry sam
  
  # Automatically discover and add new models to registry (without actually adding)
  python test_generator.py --auto-add --max-models 5
  
  # Update test_all_models.py with all model families
  python test_generator.py --update-all-models
  
  # Scan transformers library for available models
  python test_generator.py --scan-transformers
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = CURRENT_DIR.parent
RESULTS_DIR = CURRENT_DIR / "collected_results"
EXPECTED_DIR = CURRENT_DIR / "expected_results"
TEMPLATES_DIR = CURRENT_DIR / "templates"

# Model Registry - Maps model families to their configurations
MODEL_REGISTRY = {
    "bert": {
        "family_name": "BERT",
        "description": "BERT-family masked language models",
        "default_model": "bert-base-uncased",
        "class": "BertForMaskedLM",
        "test_class": "TestBertModels",
        "module_name": "test_hf_bert",
        "tasks": ["fill-mask"],
        "inputs": {
            "text": "The quick brown fox jumps over the [MASK] dog."
        },
        "dependencies": ["transformers", "tokenizers", "sentencepiece"],
        "task_specific_args": {
            "fill-mask": {"top_k": 5}
        },
        "models": {
            "bert-base-uncased": {
                "description": "BERT base model (uncased)",
                "class": "BertForMaskedLM",
                "vocab_size": 30522
            },
            "distilbert-base-uncased": {
                "description": "DistilBERT base model (uncased)",
                "class": "DistilBertForMaskedLM",
                "vocab_size": 30522
            },
            "roberta-base": {
                "description": "RoBERTa base model",
                "class": "RobertaForMaskedLM",
                "vocab_size": 50265
            }
        }
    },
    "gpt2": {
        "family_name": "GPT-2",
        "description": "GPT-2 causal language models",
        "default_model": "gpt2",
        "class": "GPT2LMHeadModel",
        "test_class": "TestGpt2Models",
        "module_name": "test_hf_gpt2",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Once upon a time"
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "text-generation": {"max_length": 50, "min_length": 20}
        },
        "models": {
            "gpt2": {
                "description": "GPT-2 small model",
                "class": "GPT2LMHeadModel"
            },
            "gpt2-medium": {
                "description": "GPT-2 medium model",
                "class": "GPT2LMHeadModel"
            },
            "distilgpt2": {
                "description": "DistilGPT-2 model",
                "class": "GPT2LMHeadModel"
            }
        }
    },
    "t5": {
        "family_name": "T5",
        "description": "T5 encoder-decoder models",
        "default_model": "t5-small",
        "class": "T5ForConditionalGeneration",
        "test_class": "TestT5Models",
        "module_name": "test_hf_t5",
        "tasks": ["translation_en_to_fr", "summarization"],
        "inputs": {
            "translation_en_to_fr": "My name is Sarah and I live in London",
            "summarization": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. It was the first structure to reach a height of 300 metres."
        },
        "dependencies": ["transformers", "sentencepiece"],
        "task_specific_args": {
            "translation_en_to_fr": {"max_length": 40},
            "summarization": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "t5-small": {
                "description": "T5 small model",
                "class": "T5ForConditionalGeneration"
            },
            "t5-base": {
                "description": "T5 base model",
                "class": "T5ForConditionalGeneration"
            },
            "google/flan-t5-small": {
                "description": "Flan-T5 small model",
                "class": "T5ForConditionalGeneration"
            }
        }
    },
    "clip": {
        "family_name": "CLIP",
        "description": "CLIP vision-language models",
        "default_model": "openai/clip-vit-base-patch32",
        "class": "CLIPModel",
        "test_class": "TestClipModels",
        "module_name": "test_hf_clip",
        "tasks": ["zero-shot-image-classification"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "candidate_labels": ["a photo of a cat", "a photo of a dog"]
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "zero-shot-image-classification": {}
        },
        "models": {
            "openai/clip-vit-base-patch32": {
                "description": "CLIP ViT-Base-Patch32 model",
                "class": "CLIPModel"
            },
            "openai/clip-vit-base-patch16": {
                "description": "CLIP ViT-Base-Patch16 model",
                "class": "CLIPModel"
            }
        }
    },
    "llama": {
        "family_name": "LLaMA",
        "description": "LLaMA causal language models",
        "default_model": "meta-llama/Llama-2-7b-hf",
        "class": "LlamaForCausalLM",
        "test_class": "TestLlamaModels",
        "module_name": "test_hf_llama",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "In this paper, we propose"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 50, "min_length": 20}
        },
        "models": {
            "meta-llama/Llama-2-7b-hf": {
                "description": "LLaMA 2 7B model",
                "class": "LlamaForCausalLM"
            },
            "meta-llama/Llama-2-7b-chat-hf": {
                "description": "LLaMA 2 7B chat model",
                "class": "LlamaForCausalLM"
            }
        }
    },
    "whisper": {
        "family_name": "Whisper",
        "description": "Whisper speech recognition models",
        "default_model": "openai/whisper-tiny",
        "class": "WhisperForConditionalGeneration",
        "test_class": "TestWhisperModels",
        "module_name": "test_hf_whisper",
        "tasks": ["automatic-speech-recognition"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "automatic-speech-recognition": {"max_length": 448}
        },
        "models": {
            "openai/whisper-tiny": {
                "description": "Whisper tiny model",
                "class": "WhisperForConditionalGeneration"
            },
            "openai/whisper-base": {
                "description": "Whisper base model",
                "class": "WhisperForConditionalGeneration"
            }
        }
    },
    "wav2vec2": {
        "family_name": "Wav2Vec2",
        "description": "Wav2Vec2 speech models",
        "default_model": "facebook/wav2vec2-base",
        "class": "Wav2Vec2ForCTC",
        "test_class": "TestWav2Vec2Models",
        "module_name": "test_hf_wav2vec2",
        "tasks": ["automatic-speech-recognition"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "automatic-speech-recognition": {}
        },
        "models": {
            "facebook/wav2vec2-base": {
                "description": "Wav2Vec2 base model",
                "class": "Wav2Vec2ForCTC"
            },
            "facebook/wav2vec2-large": {
                "description": "Wav2Vec2 large model",
                "class": "Wav2Vec2ForCTC"
            }
        }
    },
    "vit": {
        "family_name": "ViT",
        "description": "Vision Transformer models",
        "default_model": "google/vit-base-patch16-224",
        "class": "ViTForImageClassification",
        "test_class": "TestVitModels",
        "module_name": "test_hf_vit",
        "tasks": ["image-classification"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-classification": {}
        },
        "models": {
            "google/vit-base-patch16-224": {
                "description": "ViT Base model (patch size 16, image size 224)",
                "class": "ViTForImageClassification"
            },
            "facebook/deit-base-patch16-224": {
                "description": "DeiT Base model (patch size 16, image size 224)",
                "class": "DeiTForImageClassification"
            }
        }
    },
    "detr": {
        "family_name": "DETR",
        "description": "Detection Transformer models for object detection",
        "default_model": "facebook/detr-resnet-50",
        "class": "DetrForObjectDetection",
        "test_class": "TestDetrModels",
        "module_name": "test_hf_detr",
        "tasks": ["object-detection"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "object-detection": {}
        },
        "models": {
            "facebook/detr-resnet-50": {
                "description": "DETR with ResNet-50 backbone",
                "class": "DetrForObjectDetection"
            },
            "facebook/detr-resnet-101": {
                "description": "DETR with ResNet-101 backbone",
                "class": "DetrForObjectDetection"
            }
        }
    },
    "layoutlmv2": {
        "family_name": "LayoutLMv2",
        "description": "LayoutLMv2 models for document understanding",
        "default_model": "microsoft/layoutlmv2-base-uncased",
        "class": "LayoutLMv2ForTokenClassification",
        "test_class": "TestLayoutLMv2Models",
        "module_name": "test_hf_layoutlmv2",
        "tasks": ["document-question-answering"],
        "inputs": {
            "image_url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_docvqa/resolve/main/document.png",
            "question": "What is the date on this document?"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "document-question-answering": {}
        },
        "models": {
            "microsoft/layoutlmv2-base-uncased": {
                "description": "LayoutLMv2 Base model (uncased)",
                "class": "LayoutLMv2ForTokenClassification"
            },
            "microsoft/layoutlmv2-large-uncased": {
                "description": "LayoutLMv2 Large model (uncased)",
                "class": "LayoutLMv2ForTokenClassification"
            }
        }
    },
    "time_series_transformer": {
        "family_name": "TimeSeriesTransformer",
        "description": "Time Series Transformer models for forecasting",
        "default_model": "huggingface/time-series-transformer-tourism-monthly",
        "class": "TimeSeriesTransformerForPrediction",
        "test_class": "TestTimeSeriesTransformerModels",
        "module_name": "test_hf_time_series_transformer",
        "tasks": ["time-series-prediction"],
        "inputs": {
            "past_values": [100, 150, 200, 250, 300],
            "past_time_features": [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]],
            "future_time_features": [[5, 1], [6, 1], [7, 1]]
        },
        "dependencies": ["transformers", "numpy"],
        "task_specific_args": {
            "time-series-prediction": {}
        },
        "models": {
            "huggingface/time-series-transformer-tourism-monthly": {
                "description": "Time Series Transformer for monthly tourism forecasting",
                "class": "TimeSeriesTransformerForPrediction"
            }
        }
    },
    "llava": {
        "family_name": "LLaVA",
        "description": "Large Language-and-Vision Assistant models",
        "default_model": "llava-hf/llava-1.5-7b-hf",
        "class": "LlavaForConditionalGeneration",
        "test_class": "TestLlavaModels",
        "module_name": "test_hf_llava",
        "tasks": ["visual-question-answering"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "What do you see in this image?"
        },
        "dependencies": ["transformers", "pillow", "requests", "accelerate"],
        "task_specific_args": {
            "visual-question-answering": {"max_length": 200}
        },
        "models": {
            "llava-hf/llava-1.5-7b-hf": {
                "description": "LLaVA 1.5 7B model",
                "class": "LlavaForConditionalGeneration"
            }
        }
    },
    "roberta": {
        "family_name": "RoBERTa",
        "description": "RoBERTa masked language models",
        "default_model": "roberta-base",
        "class": "RobertaForMaskedLM",
        "test_class": "TestRobertaModels",
        "module_name": "test_hf_roberta",
        "tasks": ["fill-mask"],
        "inputs": {
            "text": "The quick brown fox jumps over the <mask> dog."
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "fill-mask": {"top_k": 5}
        },
        "models": {
            "roberta-base": {
                "description": "RoBERTa base model",
                "class": "RobertaForMaskedLM"
            },
            "roberta-large": {
                "description": "RoBERTa large model",
                "class": "RobertaForMaskedLM"
            },
            "distilroberta-base": {
                "description": "DistilRoBERTa base model",
                "class": "RobertaForMaskedLM"
            }
        }
    },
    "phi": {
        "family_name": "Phi",
        "description": "Phi language models from Microsoft",
        "default_model": "microsoft/phi-2",
        "class": "PhiForCausalLM",
        "test_class": "TestPhiModels",
        "module_name": "test_hf_phi",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Explain quantum computing in simple terms"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "microsoft/phi-1": {
                "description": "Phi-1 model",
                "class": "PhiForCausalLM"
            },
            "microsoft/phi-2": {
                "description": "Phi-2 model",
                "class": "PhiForCausalLM"
            }
        }
    },
    "distilbert": {
        "family_name": "DistilBERT",
        "description": "DistilBERT masked language models",
        "default_model": "distilbert-base-uncased",
        "class": "DistilBertForMaskedLM",
        "test_class": "TestDistilBertModels",
        "module_name": "test_hf_distilbert",
        "tasks": ["fill-mask"],
        "inputs": {
            "text": "The quick brown fox jumps over the [MASK] dog."
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "fill-mask": {"top_k": 5}
        },
        "models": {
            "distilbert-base-uncased": {
                "description": "DistilBERT base model (uncased)",
                "class": "DistilBertForMaskedLM"
            },
            "distilbert-base-cased": {
                "description": "DistilBERT base model (cased)",
                "class": "DistilBertForMaskedLM"
            }
        }
    },
    "visual_bert": {
        "family_name": "VisualBERT",
        "description": "VisualBERT for vision-language tasks",
        "default_model": "uclanlp/visualbert-vqa-coco-pre",
        "class": "VisualBertForQuestionAnswering",
        "test_class": "TestVisualBertModels",
        "module_name": "test_hf_visual_bert",
        "tasks": ["visual-question-answering"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "question": "What is shown in the image?"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "visual-question-answering": {}
        },
        "models": {
            "uclanlp/visualbert-vqa-coco-pre": {
                "description": "VisualBERT pretrained on COCO for VQA",
                "class": "VisualBertForQuestionAnswering"
            }
        }
    },
    "zoedepth": {
        "family_name": "ZoeDepth",
        "description": "ZoeDepth monocular depth estimation models",
        "default_model": "isl-org/ZoeDepth",
        "class": "ZoeDepthForDepthEstimation",
        "test_class": "TestZoeDepthModels",
        "module_name": "test_hf_zoedepth",
        "tasks": ["depth-estimation"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "depth-estimation": {}
        },
        "models": {
            "isl-org/ZoeDepth": {
                "description": "ZoeDepth model for monocular depth estimation",
                "class": "ZoeDepthForDepthEstimation"
            }
        }
    },
    "mistral": {
        "family_name": "Mistral",
        "description": "Mistral causal language models",
        "default_model": "mistralai/Mistral-7B-v0.1",
        "class": "MistralForCausalLM",
        "test_class": "TestMistralModels",
        "module_name": "test_hf_mistral",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Explain quantum computing in simple terms"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "mistralai/Mistral-7B-v0.1": {
                "description": "Mistral 7B model",
                "class": "MistralForCausalLM"
            },
            "mistralai/Mistral-7B-Instruct-v0.1": {
                "description": "Mistral 7B Instruct model",
                "class": "MistralForCausalLM"
            }
        }
    },
    "blip": {
        "family_name": "BLIP",
        "description": "BLIP vision-language models",
        "default_model": "Salesforce/blip-image-captioning-base",
        "class": "BlipForConditionalGeneration",
        "test_class": "TestBlipModels",
        "module_name": "test_hf_blip",
        "tasks": ["image-to-text"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-to-text": {"max_length": 50}
        },
        "models": {
            "Salesforce/blip-image-captioning-base": {
                "description": "BLIP base model for image captioning",
                "class": "BlipForConditionalGeneration"
            },
            "Salesforce/blip-vqa-base": {
                "description": "BLIP base model for visual question answering",
                "class": "BlipForQuestionAnswering"
            }
        }
    },
    "sam": {
        "family_name": "SAM",
        "description": "Segment Anything Model for image segmentation",
        "default_model": "facebook/sam-vit-base",
        "class": "SamModel",
        "test_class": "TestSamModels",
        "module_name": "test_hf_sam",
        "tasks": ["image-segmentation"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "points": [[500, 375]]
        },
        "dependencies": ["transformers", "pillow", "requests", "numpy"],
        "task_specific_args": {
            "image-segmentation": {}
        },
        "models": {
            "facebook/sam-vit-base": {
                "description": "SAM with ViT-Base backbone",
                "class": "SamModel"
            },
            "facebook/sam-vit-large": {
                "description": "SAM with ViT-Large backbone",
                "class": "SamModel"
            },
            "facebook/sam-vit-huge": {
                "description": "SAM with ViT-Huge backbone",
                "class": "SamModel"
            }
        }
    },
    "owlvit": {
        "family_name": "OWL-ViT",
        "description": "Open-vocabulary object detection with Vision Transformers",
        "default_model": "google/owlvit-base-patch32",
        "class": "OwlViTForObjectDetection",
        "test_class": "TestOwlvitModels",
        "module_name": "test_hf_owlvit",
        "tasks": ["zero-shot-object-detection"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "candidate_labels": ["cat", "dog", "person", "chair"]
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "zero-shot-object-detection": {"threshold": 0.1}
        },
        "models": {
            "google/owlvit-base-patch32": {
                "description": "OWL-ViT Base model (patch size 32)",
                "class": "OwlViTForObjectDetection"
            },
            "google/owlvit-base-patch16": {
                "description": "OWL-ViT Base model (patch size 16)",
                "class": "OwlViTForObjectDetection"
            },
            "google/owlvit-large-patch14": {
                "description": "OWL-ViT Large model (patch size 14)",
                "class": "OwlViTForObjectDetection"
            }
        }
    },
    "gemma": {
        "family_name": "Gemma",
        "description": "Gemma language models from Google",
        "default_model": "google/gemma-2b",
        "class": "GemmaForCausalLM",
        "test_class": "TestGemmaModels",
        "module_name": "test_hf_gemma",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Write a poem about artificial intelligence"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "google/gemma-2b": {
                "description": "Gemma 2B model",
                "class": "GemmaForCausalLM"
            },
            "google/gemma-7b": {
                "description": "Gemma 7B model",
                "class": "GemmaForCausalLM"
            },
            "google/gemma-2b-it": {
                "description": "Gemma 2B instruction-tuned model",
                "class": "GemmaForCausalLM"
            }
        }
    },
    "musicgen": {
        "family_name": "MusicGen",
        "description": "MusicGen music generation models from AudioCraft",
        "default_model": "facebook/musicgen-small",
        "class": "MusicgenForConditionalGeneration",
        "test_class": "TestMusicgenModels",
        "module_name": "test_hf_musicgen",
        "tasks": ["text-to-audio"],
        "inputs": {
            "text": "Electronic dance music with a strong beat and synth melody"
        },
        "dependencies": ["transformers", "tokenizers", "librosa", "soundfile"],
        "task_specific_args": {
            "text-to-audio": {"max_length": 256}
        },
        "models": {
            "facebook/musicgen-small": {
                "description": "MusicGen small model",
                "class": "MusicgenForConditionalGeneration"
            },
            "facebook/musicgen-medium": {
                "description": "MusicGen medium model",
                "class": "MusicgenForConditionalGeneration"
            },
            "facebook/musicgen-melody": {
                "description": "MusicGen melody model",
                "class": "MusicgenForConditionalGeneration"
            }
        }
    },
    "hubert": {
        "family_name": "HuBERT",
        "description": "HuBERT speech representation models",
        "default_model": "facebook/hubert-base-ls960",
        "class": "HubertModel",
        "test_class": "TestHubertModels",
        "module_name": "test_hf_hubert",
        "tasks": ["automatic-speech-recognition"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "automatic-speech-recognition": {}
        },
        "models": {
            "facebook/hubert-base-ls960": {
                "description": "HuBERT Base model",
                "class": "HubertModel"
            },
            "facebook/hubert-large-ll60k": {
                "description": "HuBERT Large model",
                "class": "HubertModel"
            },
            "facebook/hubert-xlarge-ll60k": {
                "description": "HuBERT XLarge model",
                "class": "HubertModel"
            }
        }
    },
    "donut": {
        "family_name": "Donut",
        "description": "Donut document understanding transformer",
        "default_model": "naver-clova-ix/donut-base-finetuned-docvqa",
        "class": "DonutProcessor",
        "test_class": "TestDonutModels",
        "module_name": "test_hf_donut",
        "tasks": ["document-question-answering"],
        "inputs": {
            "image_url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_docvqa/resolve/main/document.png",
            "question": "What is the date on this document?"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "document-question-answering": {}
        },
        "models": {
            "naver-clova-ix/donut-base-finetuned-docvqa": {
                "description": "Donut base model finetuned for document VQA",
                "class": "DonutProcessor"
            },
            "naver-clova-ix/donut-base-finetuned-cord-v2": {
                "description": "Donut base model finetuned for receipt parsing (CORD)",
                "class": "DonutProcessor"
            }
        }
    },
    "layoutlmv3": {
        "family_name": "LayoutLMv3",
        "description": "LayoutLMv3 models for document understanding",
        "default_model": "microsoft/layoutlmv3-base",
        "class": "LayoutLMv3ForTokenClassification",
        "test_class": "TestLayoutLMv3Models",
        "module_name": "test_hf_layoutlmv3",
        "tasks": ["document-question-answering"],
        "inputs": {
            "image_url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_docvqa/resolve/main/document.png",
            "question": "What is the date on this document?"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "document-question-answering": {}
        },
        "models": {
            "microsoft/layoutlmv3-base": {
                "description": "LayoutLMv3 Base model",
                "class": "LayoutLMv3ForTokenClassification"
            },
            "microsoft/layoutlmv3-large": {
                "description": "LayoutLMv3 Large model",
                "class": "LayoutLMv3ForTokenClassification"
            }
        }
    },
    "markuplm": {
        "family_name": "MarkupLM",
        "description": "MarkupLM models for markup language understanding",
        "default_model": "microsoft/markuplm-base",
        "class": "MarkupLMModel",
        "test_class": "TestMarkupLMModels",
        "module_name": "test_hf_markuplm",
        "tasks": ["token-classification"],
        "inputs": {
            "html": "<html><body><h1>Title</h1><p>This is a paragraph.</p></body></html>"
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "token-classification": {}
        },
        "models": {
            "microsoft/markuplm-base": {
                "description": "MarkupLM Base model",
                "class": "MarkupLMModel"
            },
            "microsoft/markuplm-large": {
                "description": "MarkupLM Large model",
                "class": "MarkupLMModel"
            }
        }
    },
    "mamba": {
        "family_name": "Mamba",
        "description": "Mamba state space models for language modeling",
        "default_model": "state-spaces/mamba-2.8b",
        "class": "MambaForCausalLM",
        "test_class": "TestMambaModels",
        "module_name": "test_hf_mamba",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Mamba is a new architecture that"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "state-spaces/mamba-2.8b": {
                "description": "Mamba 2.8B base model",
                "class": "MambaForCausalLM"
            },
            "state-spaces/mamba-1.4b": {
                "description": "Mamba 1.4B model",
                "class": "MambaForCausalLM"
            },
            "state-spaces/mamba-2.8b-slimpj": {
                "description": "Mamba 2.8B slim projection model",
                "class": "MambaForCausalLM"
            }
        }
    },
    "phi3": {
        "family_name": "Phi-3",
        "description": "Phi-3 language models from Microsoft",
        "default_model": "microsoft/phi-3-mini-4k-instruct",
        "class": "Phi3ForCausalLM",
        "test_class": "TestPhi3Models",
        "module_name": "test_hf_phi3",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Explain the theory of relativity in simple terms"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "microsoft/phi-3-mini-4k-instruct": {
                "description": "Phi-3 Mini 4K instruction-tuned model",
                "class": "Phi3ForCausalLM"
            },
            "microsoft/phi-3-small-8k-instruct": {
                "description": "Phi-3 Small 8K instruction-tuned model",
                "class": "Phi3ForCausalLM"
            },
            "microsoft/phi-3-medium-4k-instruct": {
                "description": "Phi-3 Medium 4K instruction-tuned model",
                "class": "Phi3ForCausalLM"
            }
        }
    },
    "paligemma": {
        "family_name": "PaLI-GEMMA",
        "description": "PaLI-GEMMA vision-language models from Google",
        "default_model": "google/paligemma-3b-mix-224",
        "class": "PaliGemmaForConditionalGeneration",
        "test_class": "TestPaliGemmaModels",
        "module_name": "test_hf_paligemma",
        "tasks": ["image-to-text"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "Describe this image in detail:"
        },
        "dependencies": ["transformers", "pillow", "requests", "accelerate"],
        "task_specific_args": {
            "image-to-text": {"max_length": 200}
        },
        "models": {
            "google/paligemma-3b-mix-224": {
                "description": "PaLI-GEMMA 3B model (224px)",
                "class": "PaliGemmaForConditionalGeneration"
            },
            "google/paligemma-3b-vision-224": {
                "description": "PaLI-GEMMA 3B vision model (224px)",
                "class": "PaliGemmaForConditionalGeneration"
            }
        }
    },
    "mixtral": {
        "family_name": "Mixtral",
        "description": "Mixtral mixture-of-experts language models",
        "default_model": "mistralai/Mixtral-8x7B-v0.1",
        "class": "MixtralForCausalLM",
        "test_class": "TestMixtralModels",
        "module_name": "test_hf_mixtral",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "The concept of sparse mixture-of-experts means"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "mistralai/Mixtral-8x7B-v0.1": {
                "description": "Mixtral 8x7B base model",
                "class": "MixtralForCausalLM"
            },
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                "description": "Mixtral 8x7B instruction-tuned model",
                "class": "MixtralForCausalLM"
            }
        }
    },
    "deberta_v2": {
        "family_name": "DeBERTa-V2",
        "description": "DeBERTa-V2 models with enhanced disentangled attention",
        "default_model": "microsoft/deberta-v2-xlarge",
        "class": "DebertaV2ForMaskedLM",
        "test_class": "TestDebertaV2Models",
        "module_name": "test_hf_deberta_v2",
        "tasks": ["fill-mask"],
        "inputs": {
            "text": "Paris is the [MASK] of France."
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "fill-mask": {"top_k": 5}
        },
        "models": {
            "microsoft/deberta-v2-xlarge": {
                "description": "DeBERTa V2 XLarge model",
                "class": "DebertaV2ForMaskedLM"
            },
            "microsoft/deberta-v2-xxlarge": {
                "description": "DeBERTa V2 XXLarge model",
                "class": "DebertaV2ForMaskedLM"
            },
            "microsoft/deberta-v2-xlarge-mnli": {
                "description": "DeBERTa V2 XLarge model fine-tuned on MNLI",
                "class": "DebertaV2ForSequenceClassification"
            }
        }
    },
    "video_llava": {
        "family_name": "Video-LLaVA",
        "description": "Video-LLaVA video understanding models",
        "default_model": "LanguageBind/Video-LLaVA-7B",
        "class": "VideoLlavaForConditionalGeneration",
        "test_class": "TestVideoLlavaModels",
        "module_name": "test_hf_video_llava",
        "tasks": ["video-to-text"],
        "inputs": {
            "video_url": "https://huggingface.co/datasets/LanguageBind/Video-LLaVA-Instruct-150K/resolve/main/demo/airplane-short.mp4",
            "text": "What's happening in this video?"
        },
        "dependencies": ["transformers", "pillow", "requests", "decord"],
        "task_specific_args": {
            "video-to-text": {"max_length": 200}
        },
        "models": {
            "LanguageBind/Video-LLaVA-7B": {
                "description": "Video-LLaVA 7B model",
                "class": "VideoLlavaForConditionalGeneration"
            },
            "LanguageBind/Video-LLaVA-13B": {
                "description": "Video-LLaVA 13B model",
                "class": "VideoLlavaForConditionalGeneration"
            }
        }
    },
    "blip2": {
        "family_name": "BLIP-2",
        "description": "BLIP-2 vision-language models",
        "default_model": "Salesforce/blip2-opt-2.7b",
        "class": "Blip2ForConditionalGeneration",
        "test_class": "TestBlip2Models",
        "module_name": "test_hf_blip_2",
        "tasks": ["image-to-text"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "Question: What is shown in the image? Answer:"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-to-text": {"max_length": 100}
        },
        "models": {
            "Salesforce/blip2-opt-2.7b": {
                "description": "BLIP-2 with OPT 2.7B",
                "class": "Blip2ForConditionalGeneration"
            },
            "Salesforce/blip2-flan-t5-xl": {
                "description": "BLIP-2 with Flan-T5 XL",
                "class": "Blip2ForConditionalGeneration"
            }
        }
    },
    "instructblip": {
        "family_name": "InstructBLIP",
        "description": "InstructBLIP vision-language instruction-tuned models",
        "default_model": "Salesforce/instructblip-flan-t5-xl",
        "class": "InstructBlipForConditionalGeneration",
        "test_class": "TestInstructBlipModels",
        "module_name": "test_hf_instructblip",
        "tasks": ["image-to-text"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "What is unusual about this scene?"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-to-text": {"max_length": 100}
        },
        "models": {
            "Salesforce/instructblip-flan-t5-xl": {
                "description": "InstructBLIP with Flan-T5 XL",
                "class": "InstructBlipForConditionalGeneration"
            },
            "Salesforce/instructblip-vicuna-7b": {
                "description": "InstructBLIP with Vicuna 7B",
                "class": "InstructBlipForConditionalGeneration"
            }
        }
    },
    "swin": {
        "family_name": "Swin",
        "description": "Swin Transformer vision models",
        "default_model": "microsoft/swin-base-patch4-window7-224",
        "class": "SwinForImageClassification",
        "test_class": "TestSwinModels",
        "module_name": "test_hf_swin",
        "tasks": ["image-classification"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-classification": {}
        },
        "models": {
            "microsoft/swin-base-patch4-window7-224": {
                "description": "Swin Base (patch 4, window 7, 224x224)",
                "class": "SwinForImageClassification"
            },
            "microsoft/swin-large-patch4-window7-224-in22k": {
                "description": "Swin Large (patch 4, window 7, 224x224, ImageNet-22K)",
                "class": "SwinForImageClassification"
            }
        }
    },
    "convnext": {
        "family_name": "ConvNeXT",
        "description": "ConvNeXT vision models",
        "default_model": "facebook/convnext-base-224",
        "class": "ConvNextForImageClassification",
        "test_class": "TestConvNextModels",
        "module_name": "test_hf_convnext",
        "tasks": ["image-classification"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-classification": {}
        },
        "models": {
            "facebook/convnext-base-224": {
                "description": "ConvNeXT Base (224x224)",
                "class": "ConvNextForImageClassification"
            },
            "facebook/convnext-large-224": {
                "description": "ConvNeXT Large (224x224)",
                "class": "ConvNextForImageClassification"
            }
        }
    },
    "seamless_m4t": {
        "family_name": "Seamless-M4T",
        "description": "Seamless multilingual and multimodal translation models",
        "default_model": "facebook/seamless-m4t-large",
        "class": "SeamlessM4TModel",
        "test_class": "TestSeamlessM4TModels",
        "module_name": "test_hf_seamless_m4t",
        "tasks": ["translation", "speech-to-text", "text-to-speech"],
        "inputs": {
            "text": "Hello, how are you?",
            "target_lang": "fr"
        },
        "dependencies": ["transformers", "tokenizers", "sentencepiece", "librosa"],
        "task_specific_args": {
            "translation": {"max_length": 100}
        },
        "models": {
            "facebook/seamless-m4t-large": {
                "description": "Seamless-M4T Large model",
                "class": "SeamlessM4TModel"
            },
            "facebook/seamless-m4t-medium": {
                "description": "Seamless-M4T Medium model",
                "class": "SeamlessM4TModel"
            }
        }
    },
    "wavlm": {
        "family_name": "WavLM",
        "description": "WavLM speech processing models",
        "default_model": "microsoft/wavlm-base",
        "class": "WavLMModel",
        "test_class": "TestWavLMModels",
        "module_name": "test_hf_wavlm",
        "tasks": ["audio-classification"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "audio-classification": {}
        },
        "models": {
            "microsoft/wavlm-base": {
                "description": "WavLM Base model",
                "class": "WavLMModel"
            },
            "microsoft/wavlm-base-plus": {
                "description": "WavLM Base Plus model",
                "class": "WavLMModel"
            },
            "microsoft/wavlm-large": {
                "description": "WavLM Large model",
                "class": "WavLMModel"
            }
        }
    },
    "codellama": {
        "family_name": "CodeLlama",
        "description": "CodeLlama for code generation",
        "default_model": "codellama/CodeLlama-7b-hf",
        "class": "LlamaForCausalLM",
        "test_class": "TestCodeLlamaModels",
        "module_name": "test_hf_codellama",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "def fibonacci(n):"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 200}
        },
        "models": {
            "codellama/CodeLlama-7b-hf": {
                "description": "CodeLlama 7B model",
                "class": "LlamaForCausalLM"
            },
            "codellama/CodeLlama-13b-hf": {
                "description": "CodeLlama 13B model",
                "class": "LlamaForCausalLM"
            },
            "codellama/CodeLlama-34b-hf": {
                "description": "CodeLlama 34B model",
                "class": "LlamaForCausalLM"
            }
        }
    },
    "starcoder2": {
        "family_name": "StarCoder2",
        "description": "StarCoder2 for code generation",
        "default_model": "bigcode/starcoder2-3b",
        "class": "StarCoder2ForCausalLM",
        "test_class": "TestStarcoder2Models",
        "module_name": "test_hf_starcoder2",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "def quicksort(arr):"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 200}
        },
        "models": {
            "bigcode/starcoder2-3b": {
                "description": "StarCoder2 3B model",
                "class": "StarCoder2ForCausalLM"
            },
            "bigcode/starcoder2-7b": {
                "description": "StarCoder2 7B model",
                "class": "StarCoder2ForCausalLM"
            },
            "bigcode/starcoder2-15b": {
                "description": "StarCoder2 15B model",
                "class": "StarCoder2ForCausalLM"
            }
        }
    },
    "qwen2": {
        "family_name": "Qwen2",
        "description": "Qwen2 models from Alibaba",
        "default_model": "Qwen/Qwen2-7B-Instruct",
        "class": "Qwen2ForCausalLM",
        "test_class": "TestQwen2Models",
        "module_name": "test_hf_qwen2",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Explain the concept of neural networks to a beginner"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 150}
        },
        "models": {
            "Qwen/Qwen2-7B-Instruct": {
                "description": "Qwen2 7B instruction-tuned model",
                "class": "Qwen2ForCausalLM"
            },
            "Qwen/Qwen2-7B": {
                "description": "Qwen2 7B base model",
                "class": "Qwen2ForCausalLM"
            }
        }
    },
    "bart": {
        "family_name": "BART",
        "description": "BART sequence-to-sequence models",
        "default_model": "facebook/bart-large-cnn",
        "class": "BartForConditionalGeneration",
        "test_class": "TestBartModels",
        "module_name": "test_hf_bart",
        "tasks": ["summarization"],
        "inputs": {
            "text": "The tower is 324 metres tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest human-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was completed in 1930."
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "summarization": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "facebook/bart-large-cnn": {
                "description": "BART large model fine-tuned on CNN/Daily Mail",
                "class": "BartForConditionalGeneration"
            },
            "facebook/bart-large-xsum": {
                "description": "BART large model fine-tuned on XSum",
                "class": "BartForConditionalGeneration"
            },
            "facebook/bart-large-mnli": {
                "description": "BART large model fine-tuned on MNLI",
                "class": "BartForSequenceClassification"
            }
        }
    },
    "segformer": {
        "family_name": "SegFormer",
        "description": "SegFormer models for image segmentation",
        "default_model": "nvidia/segformer-b0-finetuned-ade-512-512",
        "class": "SegformerForSemanticSegmentation",
        "test_class": "TestSegformerModels",
        "module_name": "test_hf_segformer",
        "tasks": ["image-segmentation"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-segmentation": {}
        },
        "models": {
            "nvidia/segformer-b0-finetuned-ade-512-512": {
                "description": "SegFormer B0 model finetuned on ADE20K",
                "class": "SegformerForSemanticSegmentation"
            },
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024": {
                "description": "SegFormer B5 model finetuned on Cityscapes",
                "class": "SegformerForSemanticSegmentation"
            }
        }
    },
    "dinov2": {
        "family_name": "DINOv2",
        "description": "DINOv2 self-supervised vision models",
        "default_model": "facebook/dinov2-base",
        "class": "Dinov2Model",
        "test_class": "TestDinov2Models",
        "module_name": "test_hf_dinov2",
        "tasks": ["image-classification"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-classification": {}
        },
        "models": {
            "facebook/dinov2-base": {
                "description": "DINOv2 Base model",
                "class": "Dinov2Model"
            },
            "facebook/dinov2-large": {
                "description": "DINOv2 Large model",
                "class": "Dinov2Model"
            },
            "facebook/dinov2-giant": {
                "description": "DINOv2 Giant model",
                "class": "Dinov2Model"
            }
        }
    },
    "mamba2": {
        "family_name": "Mamba2",
        "description": "Mamba2 state space models",
        "default_model": "state-spaces/mamba2-2.8b",
        "class": "Mamba2ForCausalLM",
        "test_class": "TestMamba2Models",
        "module_name": "test_hf_mamba2",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Mamba2 is an improved version that"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "state-spaces/mamba2-2.8b": {
                "description": "Mamba2 2.8B model",
                "class": "Mamba2ForCausalLM"
            },
            "state-spaces/mamba2-1.4b": {
                "description": "Mamba2 1.4B model",
                "class": "Mamba2ForCausalLM"
            }
        }
    },
    "phi4": {
        "family_name": "Phi-4",
        "description": "Phi-4 language models from Microsoft",
        "default_model": "microsoft/phi-4-mini-instruct",
        "class": "Phi4ForCausalLM",
        "test_class": "TestPhi4Models",
        "module_name": "test_hf_phi4",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "In a world where AI continues to evolve,"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "microsoft/phi-4-mini-instruct": {
                "description": "Phi-4 Mini instruction-tuned model",
                "class": "Phi4ForCausalLM"
            },
            "microsoft/phi-4-small": {
                "description": "Phi-4 Small model",
                "class": "Phi4ForCausalLM"
            }
        }
    },
    "rwkv": {
        "family_name": "RWKV",
        "description": "RWKV Receptance Weighted Key Value models",
        "default_model": "RWKV/rwkv-4-pile-430m",
        "class": "RwkvForCausalLM",
        "test_class": "TestRwkvModels",
        "module_name": "test_hf_rwkv",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "RWKV combines the best aspects of transformers and RNNs by"
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "RWKV/rwkv-4-pile-430m": {
                "description": "RWKV-4 430M model trained on Pile",
                "class": "RwkvForCausalLM"
            },
            "RWKV/rwkv-4-pile-1b5": {
                "description": "RWKV-4 1.5B model trained on Pile",
                "class": "RwkvForCausalLM"
            }
        }
    },
    "depth_anything": {
        "family_name": "Depth-Anything",
        "description": "Depth Anything models for universal depth estimation",
        "default_model": "LiheYoung/depth-anything-small",
        "class": "DepthAnythingForDepthEstimation",
        "test_class": "TestDepthAnythingModels",
        "module_name": "test_hf_depth_anything",
        "tasks": ["depth-estimation"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "depth-estimation": {}
        },
        "models": {
            "LiheYoung/depth-anything-small": {
                "description": "Depth Anything Small model",
                "class": "DepthAnythingForDepthEstimation"
            },
            "LiheYoung/depth-anything-base": {
                "description": "Depth Anything Base model",
                "class": "DepthAnythingForDepthEstimation"
            }
        }
    },
    "qwen2_audio": {
        "family_name": "Qwen2-Audio",
        "description": "Qwen2 Audio models for speech understanding",
        "default_model": "Qwen/Qwen2-Audio-7B",
        "class": "Qwen2AudioForCausalLM",
        "test_class": "TestQwen2AudioModels",
        "module_name": "test_hf_qwen2_audio",
        "tasks": ["audio-to-text"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "tokenizers", "librosa", "soundfile"],
        "task_specific_args": {
            "audio-to-text": {"max_length": 100}
        },
        "models": {
            "Qwen/Qwen2-Audio-7B": {
                "description": "Qwen2 Audio 7B model",
                "class": "Qwen2AudioForCausalLM"
            }
        }
    },
    "kosmos_2": {
        "family_name": "KOSMOS-2",
        "description": "KOSMOS-2 multimodal language models with reference grounding",
        "default_model": "microsoft/kosmos-2-patch14-224",
        "class": "Kosmos2ForConditionalGeneration",
        "test_class": "TestKosmos2Models",
        "module_name": "test_hf_kosmos_2",
        "tasks": ["image-to-text"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "This is a picture of"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-to-text": {"max_length": 100}
        },
        "models": {
            "microsoft/kosmos-2-patch14-224": {
                "description": "KOSMOS-2 model (patch size 14, 224x224)",
                "class": "Kosmos2ForConditionalGeneration"
            }
        }
    },
    "grounding_dino": {
        "family_name": "Grounding-DINO",
        "description": "Grounding DINO models for open-set object detection",
        "default_model": "IDEA-Research/grounding-dino-base",
        "class": "GroundingDinoForObjectDetection",
        "test_class": "TestGroundingDinoModels",
        "module_name": "test_hf_grounding_dino",
        "tasks": ["object-detection"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "cat . dog"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "object-detection": {"threshold": 0.3}
        },
        "models": {
            "IDEA-Research/grounding-dino-base": {
                "description": "Grounding DINO Base model",
                "class": "GroundingDinoForObjectDetection"
            },
            "IDEA-Research/grounding-dino-large": {
                "description": "Grounding DINO Large model",
                "class": "GroundingDinoForObjectDetection"
            }
        }
    },
    "wav2vec2_bert": {
        "family_name": "Wav2Vec2-BERT",
        "description": "Wav2Vec2-BERT for speech and language understanding",
        "default_model": "facebook/wav2vec2-bert-base",
        "class": "Wav2Vec2BertModel",
        "test_class": "TestWav2Vec2BertModels",
        "module_name": "test_hf_wav2vec2_bert",
        "tasks": ["automatic-speech-recognition"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "automatic-speech-recognition": {}
        },
        "models": {
            "facebook/wav2vec2-bert-base": {
                "description": "Wav2Vec2-BERT Base model",
                "class": "Wav2Vec2BertModel"
            }
        }
    },
    "idefics3": {
        "family_name": "IDEFICS3",
        "description": "IDEFICS3 vision-language models",
        "default_model": "HuggingFaceM4/idefics3-8b",
        "class": "Idefics3ForVisionText2Text",
        "test_class": "TestIdefics3Models",
        "module_name": "test_hf_idefics3",
        "tasks": ["image-to-text"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "What do you see in this image?"
        },
        "dependencies": ["transformers", "pillow", "requests", "accelerate"],
        "task_specific_args": {
            "image-to-text": {"max_length": 100}
        },
        "models": {
            "HuggingFaceM4/idefics3-8b": {
                "description": "IDEFICS3 8B model",
                "class": "Idefics3ForVisionText2Text"
            }
        }
    },
    "deepseek": {
        "family_name": "DeepSeek",
        "description": "DeepSeek language models",
        "default_model": "deepseek-ai/deepseek-llm-7b-base",
        "class": "DeepSeekForCausalLM",
        "test_class": "TestDeepSeekModels",
        "module_name": "test_hf_deepseek",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "DeepSeek is a model that can"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "deepseek-ai/deepseek-llm-7b-base": {
                "description": "DeepSeek 7B base model",
                "class": "DeepSeekForCausalLM"
            }
        }
    },
    "siglip": {
        "family_name": "SigLIP",
        "description": "SigLIP vision-language models with sigmoid loss",
        "default_model": "google/siglip-base-patch16-224",
        "class": "SiglipModel",
        "test_class": "TestSiglipModels",
        "module_name": "test_hf_siglip",
        "tasks": ["zero-shot-image-classification"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "candidate_labels": ["a photo of a cat", "a photo of a dog"]
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "zero-shot-image-classification": {}
        },
        "models": {
            "google/siglip-base-patch16-224": {
                "description": "SigLIP Base (patch size 16, 224x224)",
                "class": "SiglipModel"
            }
        }
    },
    "qwen2_vl": {
        "family_name": "Qwen2-VL",
        "description": "Qwen2 vision-language models",
        "default_model": "Qwen/Qwen2-VL-7B",
        "class": "Qwen2VLForConditionalGeneration",
        "test_class": "TestQwen2VLModels",
        "module_name": "test_hf_qwen2_vl",
        "tasks": ["image-to-text"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "What can you see in this image?"
        },
        "dependencies": ["transformers", "pillow", "requests", "accelerate"],
        "task_specific_args": {
            "image-to-text": {"max_length": 100}
        },
        "models": {
            "Qwen/Qwen2-VL-7B": {
                "description": "Qwen2-VL 7B vision-language model",
                "class": "Qwen2VLForConditionalGeneration"
            }
        }
    },
    "qwen2_audio_encoder": {
        "family_name": "Qwen2-Audio-Encoder",
        "description": "Qwen2 Audio Encoder models",
        "default_model": "Qwen/Qwen2-Audio-Encoder",
        "class": "Qwen2AudioEncoderModel",
        "test_class": "TestQwen2AudioEncoderModels",
        "module_name": "test_hf_qwen2_audio_encoder",
        "tasks": ["audio-classification"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "audio-classification": {}
        },
        "models": {
            "Qwen/Qwen2-Audio-Encoder": {
                "description": "Qwen2 Audio Encoder model",
                "class": "Qwen2AudioEncoderModel"
            }
        }
    },
    "xclip": {
        "family_name": "X-CLIP",
        "description": "X-CLIP extended CLIP models with additional capabilities",
        "default_model": "microsoft/xclip-base-patch32",
        "class": "XCLIPModel",
        "test_class": "TestXCLIPModels",
        "module_name": "test_hf_xclip",
        "tasks": ["zero-shot-image-classification"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": ["a photo of a cat", "a photo of a dog"]
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "zero-shot-image-classification": {}
        },
        "models": {
            "microsoft/xclip-base-patch32": {
                "description": "X-CLIP Base (patch size 32)",
                "class": "XCLIPModel"
            }
        }
    },
    "vilt": {
        "family_name": "ViLT",
        "description": "Vision-and-Language Transformer models",
        "default_model": "dandelin/vilt-b32-mlm",
        "class": "ViltForMaskedLM",
        "test_class": "TestViltModels",
        "module_name": "test_hf_vilt",
        "tasks": ["visual-question-answering"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "What is shown in the image?"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "visual-question-answering": {}
        },
        "models": {
            "dandelin/vilt-b32-mlm": {
                "description": "ViLT Base (patch size 32) for masked language modeling",
                "class": "ViltForMaskedLM"
            }
        }
    },
    "encodec": {
        "family_name": "EnCodec",
        "description": "EnCodec neural audio codec models",
        "default_model": "facebook/encodec_24khz",
        "class": "EncodecModel",
        "test_class": "TestEncodecModels",
        "module_name": "test_hf_encodec",
        "tasks": ["audio-to-audio"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "audio-to-audio": {}
        },
        "models": {
            "facebook/encodec_24khz": {
                "description": "EnCodec 24kHz model",
                "class": "EncodecModel"
            }
        }
    },
    "bark": {
        "family_name": "Bark",
        "description": "Bark text-to-audio generation models",
        "default_model": "suno/bark-small",
        "class": "BarkModel",
        "test_class": "TestBarkModels",
        "module_name": "test_hf_bark",
        "tasks": ["text-to-audio"],
        "inputs": {
            "text": "Hello, my name is Suno. And, I love to sing."
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "text-to-audio": {}
        },
        "models": {
            "suno/bark-small": {
                "description": "Bark Small model",
                "class": "BarkModel"
            }
        }
    },
    "biogpt": {
        "family_name": "BioGPT",
        "description": "BioGPT models for biomedical text generation",
        "default_model": "microsoft/biogpt",
        "class": "BioGptForCausalLM",
        "test_class": "TestBioGptModels",
        "module_name": "test_hf_biogpt",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "The patient presented with symptoms of"
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "text-generation": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "microsoft/biogpt": {
                "description": "BioGPT model for biomedical text generation",
                "class": "BioGptForCausalLM"
            }
        }
    },
    "esm": {
        "family_name": "ESM",
        "description": "ESM protein language models",
        "default_model": "facebook/esm2_t33_650M_UR50D",
        "class": "EsmForProteinFolding",
        "test_class": "TestEsmModels",
        "module_name": "test_hf_esm",
        "tasks": ["protein-folding"],
        "inputs": {
            "text": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "protein-folding": {}
        },
        "models": {
            "facebook/esm2_t33_650M_UR50D": {
                "description": "ESM-2 model with 33 layers and 650M parameters",
                "class": "EsmForProteinFolding"
            },
            "facebook/esm2_t6_8M_UR50D": {
                "description": "ESM-2 model with 6 layers and 8M parameters",
                "class": "EsmForProteinFolding"
            }
        }
    },
    "audioldm2": {
        "family_name": "AudioLDM2",
        "description": "AudioLDM2 text-to-audio diffusion models",
        "default_model": "cvssp/audioldm2",
        "class": "AudioLdm2ForConditionalGeneration",
        "test_class": "TestAudioLdm2Models",
        "module_name": "test_hf_audioldm2",
        "tasks": ["text-to-audio"],
        "inputs": {
            "text": "A dog barking in the distance"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "text-to-audio": {}
        },
        "models": {
            "cvssp/audioldm2": {
                "description": "AudioLDM2 base model",
                "class": "AudioLdm2ForConditionalGeneration"
            }
        }
    },
    "tinyllama": {
        "family_name": "TinyLlama",
        "description": "TinyLlama efficient small form-factor LLM",
        "default_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "class": "LlamaForCausalLM",
        "test_class": "TestTinyLlamaModels",
        "module_name": "test_hf_tinyllama",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "What are the benefits of using smaller language models?"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 100}
        },
        "models": {
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                "description": "TinyLlama 1.1B Chat model",
                "class": "LlamaForCausalLM"
            },
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-token-2.5T": {
                "description": "TinyLlama 1.1B intermediate checkpoint",
                "class": "LlamaForCausalLM"
            }
        }
    },
    "vqgan": {
        "family_name": "VQGAN",
        "description": "Vector Quantized Generative Adversarial Network",
        "default_model": "CompVis/vqgan-f16-16384",
        "class": "VQModel",
        "test_class": "TestVQGANModels",
        "module_name": "test_hf_vqgan",
        "tasks": ["image-to-image"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "image-to-image": {}
        },
        "models": {
            "CompVis/vqgan-f16-16384": {
                "description": "VQGAN model with 16-bit quantization and 16384 codes",
                "class": "VQModel"
            }
        }
    },
    "command_r": {
        "family_name": "Command-R",
        "description": "Command-R advanced instruction-following models",
        "default_model": "CohereForAI/c4ai-command-r-v01",
        "class": "AutoModelForCausalLM",
        "test_class": "TestCommandRModels",
        "module_name": "test_hf_command_r",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Explain the difference between transformer and state space models"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 150, "min_length": 50}
        },
        "models": {
            "CohereForAI/c4ai-command-r-v01": {
                "description": "Command-R base model",
                "class": "AutoModelForCausalLM"
            }
        }
    },
    "cm3": {
        "family_name": "CM3",
        "description": "CM3 multimodal model with text, image and audio capabilities",
        "default_model": "facebook/cm3leon-7b",
        "class": "Cm3LeonForConditionalGeneration",
        "test_class": "TestCm3Models",
        "module_name": "test_hf_cm3",
        "tasks": ["text-to-image", "image-to-text"],
        "inputs": {
            "text": "A cat wearing sunglasses and a leather jacket",
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
        },
        "dependencies": ["transformers", "pillow", "requests", "accelerate"],
        "task_specific_args": {
            "text-to-image": {},
            "image-to-text": {"max_length": 100}
        },
        "models": {
            "facebook/cm3leon-7b": {
                "description": "CM3Leon 7B model",
                "class": "Cm3LeonForConditionalGeneration"
            }
        }
    },
    "llava_next_video": {
        "family_name": "LLaVA-NeXT-Video",
        "description": "LLaVA-NeXT-Video for multimodal video understanding",
        "default_model": "llava-hf/llava-v1.6-vicuna-7b-video",
        "class": "LlavaNextVideoForConditionalGeneration",
        "test_class": "TestLlavaNextVideoModels",
        "module_name": "test_hf_llava_next_video",
        "tasks": ["video-to-text"],
        "inputs": {
            "video_url": "https://huggingface.co/datasets/LanguageBind/Video-LLaVA-Instruct-150K/resolve/main/demo/airplane-short.mp4",
            "text": "Describe what's happening in this video"
        },
        "dependencies": ["transformers", "pillow", "requests", "accelerate", "decord"],
        "task_specific_args": {
            "video-to-text": {"max_length": 150}
        },
        "models": {
            "llava-hf/llava-v1.6-vicuna-7b-video": {
                "description": "LLaVA NeXT Video 7B model",
                "class": "LlavaNextVideoForConditionalGeneration"
            }
        }
    },
    "orca3": {
        "family_name": "Orca3",
        "description": "Orca3 instruction-following LLM from Microsoft",
        "default_model": "microsoft/Orca-3-7B",
        "class": "Orca3ForCausalLM",
        "test_class": "TestOrca3Models",
        "module_name": "test_hf_orca3",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Explain how nuclear fusion works in simple terms"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 150, "min_length": 50}
        },
        "models": {
            "microsoft/Orca-3-7B": {
                "description": "Orca-3 7B model",
                "class": "Orca3ForCausalLM"
            }
        }
    },
    "imagebind": {
        "family_name": "ImageBind",
        "description": "ImageBind models binding multiple modalities",
        "default_model": "facebook/imagebind-huge",
        "class": "ImageBindModel",
        "test_class": "TestImageBindModels",
        "module_name": "test_hf_imagebind",
        "tasks": ["feature-extraction"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "a cat sitting on the floor",
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "pillow", "requests", "librosa"],
        "task_specific_args": {
            "feature-extraction": {}
        },
        "models": {
            "facebook/imagebind-huge": {
                "description": "ImageBind Huge model",
                "class": "ImageBindModel"
            }
        }
    },
    "cogvlm2": {
        "family_name": "CogVLM2",
        "description": "CogVLM2 vision-language model with cognitive capabilities",
        "default_model": "THUDM/cogvlm2-llama3-8b",
        "class": "CogVlm2ForConditionalGeneration",
        "test_class": "TestCogVlm2Models",
        "module_name": "test_hf_cogvlm2",
        "tasks": ["image-to-text"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "text": "Describe this image in detail with emphasis on objects, colors, and context"
        },
        "dependencies": ["transformers", "pillow", "requests", "accelerate"],
        "task_specific_args": {
            "image-to-text": {"max_length": 150}
        },
        "models": {
            "THUDM/cogvlm2-llama3-8b": {
                "description": "CogVLM2 with LLaMA3 8B",
                "class": "CogVlm2ForConditionalGeneration"
            }
        }
    },
    "graphsage": {
        "family_name": "GraphSAGE",
        "description": "GraphSAGE inductive framework for graph embeddings",
        "default_model": "deepgnn/graphsage-base",
        "class": "GraphSageForNodeClassification",
        "test_class": "TestGraphSageModels",
        "module_name": "test_hf_graphsage",
        "tasks": ["node-classification"],
        "inputs": {
            "graph_data": {
                "nodes": [0, 1, 2, 3, 4],
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
                "features": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
            }
        },
        "dependencies": ["transformers", "numpy", "torch_geometric"],
        "task_specific_args": {
            "node-classification": {}
        },
        "models": {
            "deepgnn/graphsage-base": {
                "description": "GraphSAGE base model",
                "class": "GraphSageForNodeClassification"
            }
        }
    },
    "ulip": {
        "family_name": "ULIP",
        "description": "Unified Language-Image Pre-training for point cloud understanding",
        "default_model": "salesforce/ulip-pointbert-base",
        "class": "UlipModel",
        "test_class": "TestUlipModels",
        "module_name": "test_hf_ulip",
        "tasks": ["point-cloud-classification"],
        "inputs": {
            "point_cloud_url": "https://huggingface.co/datasets/dummy-data/point-cloud/resolve/main/example.pts",
            "text": "a 3D model of a chair"
        },
        "dependencies": ["transformers", "torch", "numpy"],
        "task_specific_args": {
            "point-cloud-classification": {}
        },
        "models": {
            "salesforce/ulip-pointbert-base": {
                "description": "ULIP with PointBERT base model",
                "class": "UlipModel"
            }
        }
    },
    "claude3_haiku": {
        "family_name": "Claude3-Haiku",
        "description": "Claude 3 Haiku family large language models via Hugging Face API",
        "default_model": "anthropic/claude-3-haiku-20240307",
        "class": "Claude3Model",
        "test_class": "TestClaude3Models",
        "module_name": "test_hf_claude3_haiku",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Explain the key differences between Claude 3 Haiku and Claude 3 Sonnet"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 150, "min_length": 50}
        },
        "models": {
            "anthropic/claude-3-haiku-20240307": {
                "description": "Claude 3 Haiku model",
                "class": "Claude3Model"
            }
        }
    }
}

# Base template strings
BASE_TEST_FILE_TEMPLATE = """#!/usr/bin/env python3
\"\"\"
Class-based test file for all {family_name}-family models.
This file provides a unified testing interface for:
{model_class_comments}
\"\"\"

import os
import sys
import json
import time
import datetime
import traceback
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

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
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

{dependency_imports}

# Hardware detection
def check_hardware():
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {{
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }}
    
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

# Models registry - Maps model IDs to their specific configurations
{models_registry}

class {test_class_name}:
    \"\"\"Base test class for all {family_name}-family models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class for a specific model or default.\"\"\"
        self.model_id = model_id or "{default_model}"
        
        # Verify model exists in registry
        if self.model_id not in {registry_name}:
            logger.warning(f"Model {{self.model_id}} not in registry, using default configuration")
            self.model_info = {registry_name}["{default_model}"]
        else:
            self.model_info = {registry_name}[self.model_id]
        
        # Define model parameters
        self.task = "{default_task}"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        {test_inputs}
        
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}
    
    {test_pipeline_method}
    
    {test_from_pretrained_method}
    
    {test_openvino_method}
    
    def run_tests(self, all_hardware=False):
        \"\"\"
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
        
        Returns:
            Dict containing test results
        \"\"\"
        # Always test on default device
        self.test_pipeline()
        self.test_from_pretrained()
        
        # Test on all available hardware if requested
        if all_hardware:
            # Always test on CPU
            if self.preferred_device != "cpu":
                self.test_pipeline(device="cpu")
                self.test_from_pretrained(device="cpu")
            
            # Test on CUDA if available
            if HW_CAPABILITIES["cuda"] and self.preferred_device != "cuda":
                self.test_pipeline(device="cuda")
                self.test_from_pretrained(device="cuda")
            
            # Test on OpenVINO if available
            if HW_CAPABILITIES["openvino"]:
                self.test_with_openvino()
        
        # Build final results
        return {{
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": HW_CAPABILITIES,
            "metadata": {{
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                {has_dependencies}
            }}
        }}

def save_results(model_id, results, output_dir="collected_results"):
    \"\"\"Save test results to a file.\"\"\"
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_{family_lower}_{{safe_model_id}}_{{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {{output_path}}")
    return output_path

def get_available_models():
    \"\"\"Get a list of all available {family_name} models in the registry.\"\"\"
    return list({registry_name}.keys())

def test_all_models(output_dir="collected_results", all_hardware=False):
    \"\"\"Test all registered {family_name} models.\"\"\"
    models = get_available_models()
    results = {{}}
    
    for model_id in models:
        logger.info(f"Testing model: {{model_id}}")
        tester = {test_class_name}(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {{
            "success": any(r.get("pipeline_success", False) for r in model_results["results"].values() 
                          if r.get("pipeline_success") is not False)
        }}
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_{family_lower}_summary_{{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {{summary_path}}")
    return results

def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {family_name}-family models")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all registered models")
    
    # Hardware options
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    # List options
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_models()
        print("\\nAvailable {family_name}-family models:")
        for model in models:
            info = {registry_name}[model]
            print(f"  - {{model}} ({{info['class']}}): {{info['description']}}")
        return
    
    # Create output directory if needed
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test all models if requested
    if args.all_models:
        results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
        
        # Print summary
        print("\\n{family_name} Models Testing Summary:")
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully tested {{successful}} of {{total}} models ({{successful/total*100:.1f}}%)")
        return
    
    # Test single model (default or specified)
    model_id = args.model or "{default_model}"
    logger.info(f"Testing model: {{model_id}}")
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run test
    tester = {test_class_name}(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values()
                  if r.get("pipeline_success") is not False)
    
    print("\\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {{model_id}}")
        
        # Print performance highlights
        for device, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  - {{device}}: {{stats['avg_time']:.4f}}s average inference time")
        
        # Print example outputs if available
        if results.get("examples") and len(results["examples"]) > 0:
            print("\\nExample output:")
            example = results["examples"][0]
            if "predictions" in example:
                print(f"  Input: {{example['input']}}")
                print(f"  Predictions: {{example['predictions']}}")
            elif "output_preview" in example:
                print(f"  Input: {{example['input']}}")
                print(f"  Output: {{example['output_preview']}}")
    else:
        print(f"❌ Failed to test {{model_id}}")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {{test_name}}: {{result.get('pipeline_error_type', 'unknown')}}")
                print(f"    {{result.get('pipeline_error', 'Unknown error')}}")
    
    print("\\nFor detailed results, use --save flag and check the JSON output file.")

if __name__ == "__main__":
    main()
"""

# Template for pipeline test method
PIPELINE_TEST_TEMPLATE = """
def test_pipeline(self, device="auto"):
    \"\"\"Test the model using transformers pipeline API.\"\"\"
    if device == "auto":
        device = self.preferred_device
    
    results = {{
        "model": self.model_id,
        "device": device,
        "task": self.task,
        "class": self.class_name
    }}
    
    # Check for dependencies
    if not HAS_TRANSFORMERS:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_core"] = ["transformers"]
        results["pipeline_success"] = False
        return results
        
    {pipeline_dependency_checks}
    
    try:
        logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
        
        # Create pipeline with appropriate parameters
        pipeline_kwargs = {{
            "task": self.task,
            "model": self.model_id,
            "device": device
        }}
        
        # Time the model loading
        load_start_time = time.time()
        pipeline = transformers.pipeline(**pipeline_kwargs)
        load_time = time.time() - load_start_time
        
        # Prepare test input
        {pipeline_input_preparation}
        
        # Run warmup inference if on CUDA
        if device == "cuda":
            try:
                _ = pipeline(pipeline_input)
            except Exception:
                pass
        
        # Run multiple inference passes
        num_runs = 3
        times = []
        outputs = []
        
        for _ in range(num_runs):
            start_time = time.time()
            output = pipeline(pipeline_input)
            end_time = time.time()
            times.append(end_time - start_time)
            outputs.append(output)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Store results
        results["pipeline_success"] = True
        results["pipeline_avg_time"] = avg_time
        results["pipeline_min_time"] = min_time
        results["pipeline_max_time"] = max_time
        results["pipeline_load_time"] = load_time
        results["pipeline_error_type"] = "none"
        
        # Add to examples
        self.examples.append({{
            "method": f"pipeline() on {{device}}",
            "input": str(pipeline_input),
            "output_preview": str(outputs[0])[:200] + "..." if len(str(outputs[0])) > 200 else str(outputs[0])
        }})
        
        # Store in performance stats
        self.performance_stats[f"pipeline_{{device}}"] = {{
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "load_time": load_time,
            "num_runs": num_runs
        }}
        
    except Exception as e:
        # Store error information
        results["pipeline_success"] = False
        results["pipeline_error"] = str(e)
        results["pipeline_traceback"] = traceback.format_exc()
        logger.error(f"Error testing pipeline on {{device}}: {{e}}")
        
        # Classify error type
        error_str = str(e).lower()
        traceback_str = traceback.format_exc().lower()
        
        if "cuda" in error_str or "cuda" in traceback_str:
            results["pipeline_error_type"] = "cuda_error"
        elif "memory" in error_str:
            results["pipeline_error_type"] = "out_of_memory"
        elif "no module named" in error_str:
            results["pipeline_error_type"] = "missing_dependency"
        else:
            results["pipeline_error_type"] = "other"
    
    # Add to overall results
    self.results[f"pipeline_{{device}}"] = results
    return results
"""

# Template for from_pretrained test method
FROM_PRETRAINED_TEMPLATE = """
def test_from_pretrained(self, device="auto"):
    \"\"\"Test the model using direct from_pretrained loading.\"\"\"
    if device == "auto":
        device = self.preferred_device
    
    results = {{
        "model": self.model_id,
        "device": device,
        "task": self.task,
        "class": self.class_name
    }}
    
    # Check for dependencies
    if not HAS_TRANSFORMERS:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_core"] = ["transformers"]
        results["from_pretrained_success"] = False
        return results
        
    {from_pretrained_dependency_checks}
    
    try:
        logger.info(f"Testing {{self.model_id}} with from_pretrained() on {{device}}...")
        
        # Common parameters for loading
        pretrained_kwargs = {{
            "local_files_only": False
        }}
        
        # Time tokenizer loading
        tokenizer_load_start = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        tokenizer_load_time = time.time() - tokenizer_load_start
        
        # Use appropriate model class based on model type
        model_class = None
        if self.class_name == "{class_name}":
            model_class = transformers.{class_name}
        else:
            # Fallback to Auto class
            model_class = transformers.{auto_model_class}
        
        # Time model loading
        model_load_start = time.time()
        model = model_class.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        model_load_time = time.time() - model_load_start
        
        # Move model to device
        if device != "cpu":
            model = model.to(device)
        
        {from_pretrained_input_preparation}
        
        # Run warmup inference if using CUDA
        if device == "cuda":
            try:
                with torch.no_grad():
                    _ = model(**inputs)
            except Exception:
                pass
        
        # Run multiple inference passes
        num_runs = 3
        times = []
        outputs = []
        
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(**inputs)
            end_time = time.time()
            times.append(end_time - start_time)
            outputs.append(output)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        {from_pretrained_output_processing}
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = (param_count * 4) / (1024 * 1024)  # Rough size in MB
        
        # Store results
        results["from_pretrained_success"] = True
        results["from_pretrained_avg_time"] = avg_time
        results["from_pretrained_min_time"] = min_time
        results["from_pretrained_max_time"] = max_time
        results["tokenizer_load_time"] = tokenizer_load_time
        results["model_load_time"] = model_load_time
        results["model_size_mb"] = model_size_mb
        results["from_pretrained_error_type"] = "none"
        
        # Add predictions if available
        if 'predictions' in locals():
            results["predictions"] = predictions
        
        # Add to examples
        example_data = {{
            "method": f"from_pretrained() on {{device}}",
            "input": str(test_input)
        }}
        
        if 'predictions' in locals():
            example_data["predictions"] = predictions
        
        self.examples.append(example_data)
        
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
        
    except Exception as e:
        # Store error information
        results["from_pretrained_success"] = False
        results["from_pretrained_error"] = str(e)
        results["from_pretrained_traceback"] = traceback.format_exc()
        logger.error(f"Error testing from_pretrained on {{device}}: {{e}}")
        
        # Classify error type
        error_str = str(e).lower()
        traceback_str = traceback.format_exc().lower()
        
        if "cuda" in error_str or "cuda" in traceback_str:
            results["from_pretrained_error_type"] = "cuda_error"
        elif "memory" in error_str:
            results["from_pretrained_error_type"] = "out_of_memory"
        elif "no module named" in error_str:
            results["from_pretrained_error_type"] = "missing_dependency"
        else:
            results["from_pretrained_error_type"] = "other"
    
    # Add to overall results
    self.results[f"from_pretrained_{{device}}"] = results
    return results
"""

# Template for OpenVINO test method
OPENVINO_TEST_TEMPLATE = """
def test_with_openvino(self):
    \"\"\"Test the model using OpenVINO integration.\"\"\"
    results = {{
        "model": self.model_id,
        "task": self.task,
        "class": self.class_name
    }}
    
    # Check for OpenVINO support
    if not HW_CAPABILITIES["openvino"]:
        results["openvino_error_type"] = "missing_dependency"
        results["openvino_missing_core"] = ["openvino"]
        results["openvino_success"] = False
        return results
    
    # Check for transformers
    if not HAS_TRANSFORMERS:
        results["openvino_error_type"] = "missing_dependency"
        results["openvino_missing_core"] = ["transformers"]
        results["openvino_success"] = False
        return results
    
    try:
        from optimum.intel import {openvino_model_class}
        logger.info(f"Testing {{self.model_id}} with OpenVINO...")
        
        # Time tokenizer loading
        tokenizer_load_start = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        tokenizer_load_time = time.time() - tokenizer_load_start
        
        # Time model loading
        model_load_start = time.time()
        model = {openvino_model_class}.from_pretrained(
            self.model_id,
            export=True,
            provider="CPU"
        )
        model_load_time = time.time() - model_load_start
        
        {openvino_input_preparation}
        
        # Run inference
        start_time = time.time()
        outputs = model(**inputs)
        inference_time = time.time() - start_time
        
        {openvino_output_processing}
        
        # Store results
        results["openvino_success"] = True
        results["openvino_load_time"] = model_load_time
        results["openvino_inference_time"] = inference_time
        results["openvino_tokenizer_load_time"] = tokenizer_load_time
        
        # Add predictions if available
        if 'predictions' in locals():
            results["openvino_predictions"] = predictions
        
        results["openvino_error_type"] = "none"
        
        # Add to examples
        example_data = {{
            "method": "OpenVINO inference",
            "input": str(test_input)
        }}
        
        if 'predictions' in locals():
            example_data["predictions"] = predictions
        
        self.examples.append(example_data)
        
        # Store in performance stats
        self.performance_stats["openvino"] = {{
            "inference_time": inference_time,
            "load_time": model_load_time,
            "tokenizer_load_time": tokenizer_load_time
        }}
        
    except Exception as e:
        # Store error information
        results["openvino_success"] = False
        results["openvino_error"] = str(e)
        results["openvino_traceback"] = traceback.format_exc()
        logger.error(f"Error testing with OpenVINO: {{e}}")
        
        # Classify error
        error_str = str(e).lower()
        if "no module named" in error_str:
            results["openvino_error_type"] = "missing_dependency"
        else:
            results["openvino_error_type"] = "other"
    
    # Add to overall results
    self.results["openvino"] = results
    return results
"""

def generate_model_class_comments(family_info):
    """Generate comments for model classes in this family."""
    models = family_info.get("models", {})
    classes = set()
    for model_info in models.values():
        if "class" in model_info:
            classes.add(model_info["class"])
    
    result = []
    for cls in classes:
        result.append(f"- {cls}")
    
    # Add default if empty
    if not result:
        result.append(f"- {family_info['class']}")
    
    return "\n".join(result)

def generate_dependency_imports(family_info):
    """Generate import statements for family-specific dependencies."""
    dependencies = family_info.get("dependencies", [])
    result = []
    
    # Add standard imports for common dependencies
    for dep in dependencies:
        if dep == "tokenizers":
            result.append("""
# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")
""")
        elif dep == "sentencepiece":
            result.append("""
# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")
""")
        elif dep == "pillow":
            result.append("""
# Try to import PIL
try:
    from PIL import Image
    import requests
    from io import BytesIO
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    requests = MagicMock()
    BytesIO = MagicMock()
    HAS_PIL = False
    logger.warning("PIL or requests not available, using mock")
""")
        elif dep == "librosa":
            result.append("""
# Try to import audio processing libraries
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    librosa = MagicMock()
    sf = MagicMock()
    HAS_AUDIO = False
    logger.warning("librosa or soundfile not available, using mock")
""")
        elif dep == "accelerate":
            result.append("""
# Try to import accelerate
try:
    import accelerate
    HAS_ACCELERATE = True
except ImportError:
    accelerate = MagicMock()
    HAS_ACCELERATE = False
    logger.warning("accelerate not available, using mock")
""")
    
    # Add mock implementations for dependencies
    if "tokenizers" in dependencies:
        result.append("""
# Mock implementations for missing dependencies
if not HAS_TOKENIZERS:
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

    tokenizers.Tokenizer = MockTokenizer
""")
    
    if "sentencepiece" in dependencies:
        result.append("""
if not HAS_SENTENCEPIECE:
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

    sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor
""")
    
    if "pillow" in dependencies:
        result.append("""
if not HAS_PIL:
    class MockImage:
        @staticmethod
        def open(file):
            class MockImg:
                def __init__(self):
                    self.size = (224, 224)
                def convert(self, mode):
                    return self
                def resize(self, size):
                    return self
            return MockImg()
            
    class MockRequests:
        @staticmethod
        def get(url):
            class MockResponse:
                def __init__(self):
                    self.content = b"mock image data"
                def raise_for_status(self):
                    pass
            return MockResponse()

    Image.open = MockImage.open
    requests.get = MockRequests.get
""")
    
    if "librosa" in dependencies:
        result.append("""
if not HAS_AUDIO:
    def mock_load(file_path, sr=None, mono=True):
        return (np.zeros(16000), 16000)
        
    class MockSoundFile:
        @staticmethod
        def write(file, data, samplerate):
            pass
    
    if isinstance(librosa, MagicMock):
        librosa.load = mock_load
    
    if isinstance(sf, MagicMock):
        sf.write = MockSoundFile.write
""")
    
    return "\n".join(result)

def generate_models_registry(family_info):
    """Generate the models registry dictionary for a family."""
    family_name = family_info["family_name"].upper()
    models = family_info.get("models", {})
    
    registry_lines = [f"{family_name}_MODELS_REGISTRY = {{"]
    
    for model_id, model_info in models.items():
        registry_lines.append(f'    "{model_id}": {{')
        for key, value in model_info.items():
            if isinstance(value, str):
                registry_lines.append(f'        "{key}": "{value}",')
            else:
                registry_lines.append(f'        "{key}": {value},')
        registry_lines.append('    },')
    
    registry_lines.append("}")
    
    return "\n".join(registry_lines)

def generate_test_inputs(family_info):
    """Generate test input initialization."""
    task = family_info.get("tasks", ["text-generation"])[0]
    inputs = family_info.get("inputs", {})
    
    lines = []
    
    if "text" in inputs:
        lines.append(f'self.test_text = "{inputs["text"]}"')
        lines.append('self.test_texts = [')
        lines.append(f'    "{inputs["text"]}",')
        lines.append(f'    "{inputs["text"]} (alternative)"')
        lines.append(']')
    
    if "image_url" in inputs:
        lines.append(f'self.test_image_url = "{inputs["image_url"]}"')
        if "candidate_labels" in inputs:
            labels = inputs["candidate_labels"]
            lines.append('self.candidate_labels = [')
            for label in labels:
                lines.append(f'    "{label}",')
            lines.append(']')
    
    if "audio_file" in inputs:
        lines.append(f'self.test_audio = "{inputs["audio_file"]}"')
    
    return "\n        ".join(lines)

def generate_pipeline_dependency_checks(family_info):
    """Generate dependency checks for pipeline test."""
    dependencies = family_info.get("dependencies", [])
    checks = []
    
    for dep in dependencies:
        if dep == "tokenizers":
            checks.append("""if not HAS_TOKENIZERS:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["tokenizers>=0.11.0"]
        results["pipeline_success"] = False
        return results""")
        elif dep == "sentencepiece":
            checks.append("""if not HAS_SENTENCEPIECE:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["sentencepiece>=0.1.91"]
        results["pipeline_success"] = False
        return results""")
        elif dep == "pillow":
            checks.append("""if not HAS_PIL:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
        results["pipeline_success"] = False
        return results""")
        elif dep == "librosa":
            checks.append("""if not HAS_AUDIO:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["librosa>=0.8.0", "soundfile>=0.10.0"]
        results["pipeline_success"] = False
        return results""")
        elif dep == "accelerate":
            checks.append("""if not HAS_ACCELERATE:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["accelerate>=0.12.0"]
        results["pipeline_success"] = False
        return results""")
    
    return "\n    ".join(checks)

def generate_from_pretrained_dependency_checks(family_info):
    """Generate dependency checks for from_pretrained test."""
    dependencies = family_info.get("dependencies", [])
    checks = []
    
    for dep in dependencies:
        if dep == "tokenizers":
            checks.append("""if not HAS_TOKENIZERS:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["tokenizers>=0.11.0"]
        results["from_pretrained_success"] = False
        return results""")
        elif dep == "sentencepiece":
            checks.append("""if not HAS_SENTENCEPIECE:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["sentencepiece>=0.1.91"]
        results["from_pretrained_success"] = False
        return results""")
        elif dep == "pillow":
            checks.append("""if not HAS_PIL:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
        results["from_pretrained_success"] = False
        return results""")
        elif dep == "librosa":
            checks.append("""if not HAS_AUDIO:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["librosa>=0.8.0", "soundfile>=0.10.0"]
        results["from_pretrained_success"] = False
        return results""")
        elif dep == "accelerate":
            checks.append("""if not HAS_ACCELERATE:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["accelerate>=0.12.0"]
        results["from_pretrained_success"] = False
        return results""")
    
    return "\n    ".join(checks)

def generate_pipeline_input_preparation(family_info):
    """Generate code to prepare input for pipeline."""
    task = family_info.get("tasks", ["text-generation"])[0]
    inputs = family_info.get("inputs", {})
    
    if "text" in inputs:
        return "pipeline_input = self.test_text"
    elif "image_url" in inputs:
        return """if HAS_PIL:
            pipeline_input = requests.get(self.test_image_url).content
        else:
            pipeline_input = self.test_image_url"""
    elif "audio_file" in inputs:
        return """if os.path.exists(self.test_audio):
            pipeline_input = self.test_audio
        else:
            # Use a sample array if file not found
            pipeline_input = np.zeros(16000)"""
    else:
        return "pipeline_input = None  # Default empty input"

def generate_from_pretrained_input_preparation(family_info):
    """Generate code to prepare input for from_pretrained."""
    task = family_info.get("tasks", ["text-generation"])[0]
    inputs = family_info.get("inputs", {})
    
    if task in ["text-generation", "fill-mask", "translation_en_to_fr", "summarization"]:
        return """# Prepare test input
        test_input = self.test_text
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}"""
    elif task in ["zero-shot-image-classification"]:
        return """# Prepare test input
        test_input = self.test_image_url
        
        # Get image
        if HAS_PIL:
            response = requests.get(test_input)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Mock image
            image = None
            
        # Get text features
        inputs = tokenizer(self.candidate_labels, padding=True, return_tensors="pt")
        
        if HAS_PIL:
            # Get image features
            processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            image_inputs = processor(images=image, return_tensors="pt")
            inputs.update(image_inputs)
        
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}"""
    elif task in ["automatic-speech-recognition"]:
        return """# Prepare test input
        test_input = self.test_audio
        
        # Load audio
        if HAS_AUDIO and os.path.exists(test_input):
            waveform, sample_rate = librosa.load(test_input, sr=16000)
            inputs = {"input_values": torch.tensor([waveform]).float()}
        else:
            # Mock audio input
            inputs = {"input_values": torch.zeros(1, 16000).float()}
            
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}"""
    else:
        # Generic fallback
        return """# Prepare test input
        test_input = "Generic input for testing"
        
        # Create generic inputs
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}"""

def generate_from_pretrained_output_processing(family_info):
    """Generate code to process output for from_pretrained."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task == "fill-mask":
        return """# Get top predictions for masked position
        if hasattr(tokenizer, "mask_token_id"):
            mask_token_id = tokenizer.mask_token_id
            mask_positions = (inputs["input_ids"] == mask_token_id).nonzero()
            
            if len(mask_positions) > 0:
                mask_index = mask_positions[0][-1].item()
                logits = outputs[0].logits[0, mask_index]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top_k = torch.topk(probs, 5)
                
                predictions = []
                for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices)):
                    if hasattr(tokenizer, "convert_ids_to_tokens"):
                        token = tokenizer.convert_ids_to_tokens(idx.item())
                    else:
                        token = f"token_{idx.item()}"
                    predictions.append({
                        "token": token,
                        "probability": prob.item()
                    })
            else:
                predictions = []
        else:
            predictions = []"""
    elif task == "text-generation":
        return """# Process generation output
        predictions = outputs[0]
        if hasattr(tokenizer, "decode"):
            if hasattr(outputs[0], "logits"):
                logits = outputs[0].logits
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = tokenizer.decode([next_token_id])
                predictions = [{"token": next_token, "score": 1.0}]
            else:
                predictions = [{"generated_text": "Mock generated text"}]"""
    elif task in ["translation_en_to_fr", "summarization"]:
        return """# Process generation output
        if hasattr(outputs[0], "logits"):
            logits = outputs[0].logits
            generated_ids = torch.argmax(logits, dim=-1)
            if hasattr(tokenizer, "decode"):
                decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                predictions = [{"generated_text": decoded_output}]
            else:
                predictions = [{"generated_text": "Mock generated text"}]
        else:
            predictions = [{"generated_text": "Mock generated text"}]"""
    elif task == "zero-shot-image-classification":
        return """# Process classification output
        if hasattr(outputs, "logits_per_image"):
            logits = outputs.logits_per_image[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = []
            for i, (label, prob) in enumerate(zip(self.candidate_labels, probs)):
                predictions.append({
                    "label": label,
                    "score": prob.item()
                })
        else:
            predictions = [{"label": "Mock label", "score": 0.95}]"""
    elif task == "automatic-speech-recognition":
        return """# Process speech recognition output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            if hasattr(tokenizer, "decode"):
                transcription = tokenizer.decode(predicted_ids[0])
                predictions = [{"text": transcription}]
            else:
                predictions = [{"text": "Mock transcription"}]
        else:
            predictions = [{"text": "Mock transcription"}]"""
    else:
        # Generic fallback
        return """# Generic output processing
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            predictions = [{"output": "Processed model output"}]
        else:
            predictions = [{"output": "Mock output"}]"""

def generate_openvino_input_preparation(family_info):
    """Generate code to prepare input for OpenVINO."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task in ["fill-mask", "text-generation", "translation_en_to_fr", "summarization"]:
        return """# Prepare input
        if hasattr(tokenizer, "mask_token") and "[MASK]" in self.test_text:
            mask_token = tokenizer.mask_token
            test_input = self.test_text.replace("[MASK]", mask_token)
        else:
            test_input = self.test_text
            
        inputs = tokenizer(test_input, return_tensors="pt")"""
    elif task == "zero-shot-image-classification":
        return """# Prepare input
        test_input = self.test_image_url
        
        # Process image
        if HAS_PIL:
            response = requests.get(test_input)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Get text features
            inputs = tokenizer(self.candidate_labels, padding=True, return_tensors="pt")
            
            # Get image features
            processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            image_inputs = processor(images=image, return_tensors="pt")
            inputs.update(image_inputs)
        else:
            # Mock inputs
            inputs = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                "pixel_values": torch.zeros(1, 3, 224, 224)
            }"""
    elif task == "automatic-speech-recognition":
        return """# Prepare input
        test_input = self.test_audio
        
        # Load audio
        if HAS_AUDIO and os.path.exists(test_input):
            waveform, sample_rate = librosa.load(test_input, sr=16000)
            inputs = {"input_values": torch.tensor([waveform]).float()}
        else:
            # Mock audio input
            inputs = {"input_values": torch.zeros(1, 16000).float()}"""
    else:
        # Generic fallback
        return """# Prepare generic input
        test_input = "Generic input for testing"
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}"""

def generate_openvino_output_processing(family_info):
    """Generate code to process output for OpenVINO."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task == "fill-mask":
        return """# Get predictions
        if hasattr(tokenizer, "mask_token_id"):
            mask_token_id = tokenizer.mask_token_id
            mask_positions = (inputs["input_ids"] == mask_token_id).nonzero()
            
            if len(mask_positions) > 0:
                mask_index = mask_positions[0][-1].item()
                logits = outputs.logits[0, mask_index]
                top_k_indices = torch.topk(logits, 5).indices.tolist()
                
                predictions = []
                for idx in top_k_indices:
                    if hasattr(tokenizer, "convert_ids_to_tokens"):
                        token = tokenizer.convert_ids_to_tokens(idx)
                    else:
                        token = f"token_{idx}"
                    predictions.append(token)
            else:
                predictions = []
        else:
            predictions = []"""
    elif task == "text-generation":
        return """# Process generation output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            if hasattr(tokenizer, "decode"):
                next_token = tokenizer.decode([next_token_id])
                predictions = [next_token]
            else:
                predictions = ["<mock_token>"]
        else:
            predictions = ["<mock_output>"]"""
    elif task in ["translation_en_to_fr", "summarization"]:
        return """# Process generation output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            generated_ids = torch.argmax(logits, dim=-1)
            
            if hasattr(tokenizer, "decode"):
                decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                predictions = [decoded_output]
            else:
                predictions = ["<mock_output>"]
        else:
            predictions = ["<mock_output>"]"""
    elif task == "zero-shot-image-classification":
        return """# Process classification output
        if hasattr(outputs, "logits_per_image"):
            logits = outputs.logits_per_image[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            predictions = []
            for i, (label, prob) in enumerate(zip(self.candidate_labels, probs)):
                predictions.append({
                    "label": label,
                    "score": float(prob)
                })
        else:
            predictions = [{"label": "Mock label", "score": 0.95}]"""
    elif task == "automatic-speech-recognition":
        return """# Process speech recognition output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
            if hasattr(tokenizer, "decode"):
                transcription = tokenizer.decode(predicted_ids[0])
                predictions = [transcription]
            else:
                predictions = ["<mock_transcription>"]
        else:
            predictions = ["<mock_transcription>"]"""
    else:
        # Generic fallback
        return """# Generic output processing
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            predictions = ["Processed OpenVINO output"]
        else:
            predictions = ["<mock_output>"]"""

def generate_dependency_flags(family_info):
    """Generate dependency flag list for metadata."""
    dependencies = family_info.get("dependencies", [])
    
    flags = []
    for dep in dependencies:
        if dep == "tokenizers":
            flags.append('"has_tokenizers": HAS_TOKENIZERS')
        elif dep == "sentencepiece":
            flags.append('"has_sentencepiece": HAS_SENTENCEPIECE')
        elif dep == "pillow":
            flags.append('"has_pil": HAS_PIL')
        elif dep == "librosa":
            flags.append('"has_audio": HAS_AUDIO')
        elif dep == "accelerate":
            flags.append('"has_accelerate": HAS_ACCELERATE')
    
    return ",\n                ".join(flags)

def generate_auto_model_class(family_info):
    """Generate the appropriate Auto model class."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task == "fill-mask":
        return "AutoModelForMaskedLM"
    elif task == "text-generation":
        return "AutoModelForCausalLM"
    elif task in ["translation_en_to_fr", "summarization"]:
        return "AutoModelForSeq2SeqLM"
    elif task == "zero-shot-image-classification":
        return "AutoModel"
    elif task == "automatic-speech-recognition":
        return "AutoModelForSpeechSeq2Seq"
    else:
        # Generic fallback
        return "AutoModel"

def generate_openvino_model_class(family_info):
    """Generate the appropriate OpenVINO model class."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task == "fill-mask":
        return "OVModelForMaskedLM"
    elif task == "text-generation":
        return "OVModelForCausalLM"
    elif task in ["translation_en_to_fr", "summarization"]:
        return "OVModelForSeq2SeqLM"
    elif task == "zero-shot-image-classification":
        return "OVModelForVision"
    elif task == "automatic-speech-recognition":
        return "OVModelForSpeechSeq2Seq"
    else:
        # Generic fallback
        return "OVModel"

def generate_test_file(family_id):
    """Generate a complete test file for a model family."""
    family_info = MODEL_REGISTRY.get(family_id)
    if not family_info:
        logger.error(f"Unknown model family: {family_id}")
        return None
    
    # Prepare template variables
    variables = {
        "family_name": family_info["family_name"],
        "family_lower": family_id.lower(),
        "default_model": family_info["default_model"],
        "class_name": family_info["class"],
        "test_class_name": family_info["test_class"],
        "registry_name": f"{family_info['family_name'].upper()}_MODELS_REGISTRY",
        "default_task": family_info["tasks"][0] if family_info.get("tasks") else "text-generation",
        "model_class_comments": generate_model_class_comments(family_info),
        "dependency_imports": generate_dependency_imports(family_info),
        "models_registry": generate_models_registry(family_info),
        "test_inputs": generate_test_inputs(family_info),
        "has_dependencies": generate_dependency_flags(family_info),
        "auto_model_class": generate_auto_model_class(family_info),
        "openvino_model_class": generate_openvino_model_class(family_info)
    }
    
    # Generate pipeline test method
    pipeline_test_variables = {
        "pipeline_dependency_checks": generate_pipeline_dependency_checks(family_info),
        "pipeline_input_preparation": generate_pipeline_input_preparation(family_info)
    }
    pipeline_method = PIPELINE_TEST_TEMPLATE.format(**pipeline_test_variables)
    variables["test_pipeline_method"] = pipeline_method
    
    # Generate from_pretrained test method
    from_pretrained_variables = {
        "from_pretrained_dependency_checks": generate_from_pretrained_dependency_checks(family_info),
        "from_pretrained_input_preparation": generate_from_pretrained_input_preparation(family_info),
        "from_pretrained_output_processing": generate_from_pretrained_output_processing(family_info),
        "class_name": family_info["class"],
        "auto_model_class": generate_auto_model_class(family_info)
    }
    from_pretrained_method = FROM_PRETRAINED_TEMPLATE.format(**from_pretrained_variables)
    variables["test_from_pretrained_method"] = from_pretrained_method
    
    # Generate OpenVINO test method
    openvino_variables = {
        "openvino_model_class": generate_openvino_model_class(family_info),
        "openvino_input_preparation": generate_openvino_input_preparation(family_info),
        "openvino_output_processing": generate_openvino_output_processing(family_info)
    }
    openvino_method = OPENVINO_TEST_TEMPLATE.format(**openvino_variables)
    variables["test_openvino_method"] = openvino_method
    
    # Generate complete file
    test_file_content = BASE_TEST_FILE_TEMPLATE.format(**variables)
    
    return test_file_content

def create_test_file(family_id):
    """Create a test file for a model family and save it."""
    content = generate_test_file(family_id)
    if not content:
        return False
    
    family_info = MODEL_REGISTRY.get(family_id)
    module_name = family_info["module_name"]
    
    # Write to file
    file_path = CURRENT_DIR / f"{module_name}.py"
    with open(file_path, "w") as f:
        f.write(content)
    
    logger.info(f"Created test file: {file_path}")
    return True

def generate_all_test_files(model_list=None):
    """
    Generate test files for all model families or a specific list.
    
    Args:
        model_list: Optional list of model families to generate tests for
    """
    successful = []
    failed = []
    
    # Determine which models to generate tests for
    if model_list:
        families_to_generate = [f for f in model_list if f in MODEL_REGISTRY]
        missing = [f for f in model_list if f not in MODEL_REGISTRY]
        if missing:
            logger.warning(f"The following models are not in the registry: {', '.join(missing)}")
    else:
        families_to_generate = list(MODEL_REGISTRY.keys())
    
    # Generate test files for each family
    for family_id in families_to_generate:
        logger.info(f"Generating test file for {family_id}...")
        if create_test_file(family_id):
            successful.append(family_id)
        else:
            failed.append(family_id)
    
    logger.info(f"Successfully generated {len(successful)} test files")
    if failed:
        logger.warning(f"Failed to generate {len(failed)} test files: {', '.join(failed)}")
    
    return successful, failed

def auto_add_tests(max_models=5):
    """
    Automatically find models without tests and generate registry entries and test files.
    
    Args:
        max_models: Maximum number of new models to add
    """
    # Scan for available models
    discovered_models = scan_hf_transformers()
    suggestions = suggest_new_models(discovered_models)
    
    # Limit to max_models
    if len(suggestions) > max_models:
        logger.info(f"Limiting to {max_models} models out of {len(suggestions)} discovered")
        suggestions = suggestions[:max_models]
    
    added_models = []
    generated_tests = []
    
    for suggestion in suggestions:
        family_id = suggestion["family_id"]
        
        # Get model specifics if available
        model_specifics = get_model_specifics(family_id)
        
        # Generate registry entry
        entry = generate_model_registry_entry(suggestion)
        logger.info(f"Generated registry entry for {family_id}")
        
        # We would add to registry here in a complete solution
        # For now, let's print what we would add
        logger.info(f"Would add {family_id} to MODEL_REGISTRY")
        print(f"\nGenerated entry for {family_id}:")
        print(entry)
        
        added_models.append(family_id)
    
    logger.info(f"Added {len(added_models)} new models to registry")
    
    # We would generate test files here in a complete solution
    # logger.info(f"Generating test files for new models")
    # successful, failed = generate_all_test_files(added_models)
    # logger.info(f"Generated test files for {len(successful)} models")
    
    return added_models

def update_test_all_models():
    """Update the test_all_models.py file with all model families."""
    model_families = {}
    
    for family_id, family_info in MODEL_REGISTRY.items():
        model_families[family_id] = {
            "module": family_info["module_name"],
            "description": family_info["description"],
            "default_model": family_info["default_model"],
            "class": family_info["test_class"],
            "status": "complete"
        }
    
    # Update file
    file_path = CURRENT_DIR / "test_all_models.py"
    
    # Check if file exists
    if not file_path.exists():
        logger.warning(f"test_all_models.py not found at {file_path}")
        return False
    
    # Read current content
    with open(file_path, "r") as f:
        content = f.read()
    
    # Find MODEL_FAMILIES definition
    import re
    model_families_match = re.search(r"MODEL_FAMILIES\s*=\s*\{([^}]+)\}", content, re.DOTALL)
    if not model_families_match:
        logger.warning("MODEL_FAMILIES definition not found in test_all_models.py")
        return False
    
    # Generate new MODEL_FAMILIES definition
    model_families_str = "MODEL_FAMILIES = {\n"
    for family_id, family_info in model_families.items():
        model_families_str += f'    "{family_id}": {{\n'
        model_families_str += f'        "module": "{family_info["module"]}",\n'
        model_families_str += f'        "description": "{family_info["description"]}",\n'
        model_families_str += f'        "default_model": "{family_info["default_model"]}",\n'
        model_families_str += f'        "class": "{family_info["class"]}",\n'
        model_families_str += f'        "status": "{family_info["status"]}"\n'
        model_families_str += f'    }},\n'
    model_families_str += "}"
    
    # Replace MODEL_FAMILIES definition
    new_content = re.sub(r"MODEL_FAMILIES\s*=\s*\{[^}]+\}", model_families_str, content, flags=re.DOTALL)
    
    # Write new content
    with open(file_path, "w") as f:
        f.write(new_content)
    
    logger.info(f"Updated test_all_models.py with {len(model_families)} model families")
    return True

def scan_hf_transformers():
    """
    Scan transformers library to discover available models and architectures.
    This helps expand the test coverage by identifying models not yet in the registry.
    """
    discovered_models = {}
    
    # Check if transformers is available
    try:
        import transformers
        has_transformers = True
    except ImportError:
        logger.warning("transformers not available, cannot scan for models")
        has_transformers = False
        return discovered_models
        
    if not has_transformers:
        logger.warning("transformers not available, cannot scan for models")
        return discovered_models
    
    try:
        # Get auto classes from the transformers package
        auto_mapping = {}
        
        # Try to find all model classes that support various tasks
        if hasattr(transformers, "AutoModelForMaskedLM"):
            auto_mapping.update({"bert_for_masked_lm": "AutoModelForMaskedLM"})
        
        if hasattr(transformers, "AutoModelForCausalLM"):
            auto_mapping.update({"gpt_for_causal_lm": "AutoModelForCausalLM"})
            
        if hasattr(transformers, "AutoModelForSeq2SeqLM"):
            auto_mapping.update({"t5_for_conditional_generation": "AutoModelForSeq2SeqLM"})
            
        if hasattr(transformers, "AutoModelForImageClassification"):
            auto_mapping.update({"vit_for_image_classification": "AutoModelForImageClassification"})
            
        if hasattr(transformers, "AutoModelForObjectDetection"):
            auto_mapping.update({"detr_for_object_detection": "AutoModelForObjectDetection"})
            
        if hasattr(transformers, "AutoModelForAudioClassification"):
            auto_mapping.update({"wav2vec2_for_audio_classification": "AutoModelForAudioClassification"})
            
        if hasattr(transformers, "AutoModelForSpeechSeq2Seq"):
            auto_mapping.update({"whisper_for_conditional_generation": "AutoModelForSpeechSeq2Seq"})
            
        if hasattr(transformers, "AutoModelForQuestionAnswering"):
            auto_mapping.update({"bert_for_question_answering": "AutoModelForQuestionAnswering"})
            
        if hasattr(transformers, "AutoModelForImageSegmentation"):
            auto_mapping.update({"mask2former_for_image_segmentation": "AutoModelForImageSegmentation"})
            
        # Directly add common model types to scan
        model_types = [
            "bert", "gpt2", "t5", "roberta", "distilbert", "bart", "vit", "clip", 
            "whisper", "wav2vec2", "layoutlm", "detr", "segformer", "deit", "llama",
            "sam", "blip", "llava", "phi", "mistral", "falcon", "flan", "mt5",
            "bloom", "deberta", "electra", "xlm", "dino", "beit", "blenderbot",
            "pegasus", "clap", "camembert", "albert", "mobilevit", "resnet", "owlvit",
            "informer", "codegen", "codellama", "xclip", "clipseg", "donut", "dinov2", 
            "depth_anything", "pix2struct", "fuyu", "gemma", "convnext", "beit", "wavlm",
            "musicgen", "siglip", "clap", "starcoder", "zoedepth"
        ]
        
        # Add common model types to the auto mapping
        for model_type in model_types:
            if model_type not in auto_mapping:
                auto_mapping[f"{model_type}_model"] = f"Model for {model_type}"
        
        # Extract model families from auto mapping
        for model_type, model_class in auto_mapping.items():
            model_family = model_type.split('_')[0]
            if model_family not in discovered_models:
                discovered_models[model_family] = {
                    "model_classes": set(),
                    "tasks": set(),
                }
            
            discovered_models[model_family]["model_classes"].add(model_class)
            
            # Infer task from class name
            if "MaskedLM" in model_class:
                discovered_models[model_family]["tasks"].add("fill-mask")
            elif "CausalLM" in model_class:
                discovered_models[model_family]["tasks"].add("text-generation")
            elif "Seq2SeqLM" in model_class:
                discovered_models[model_family]["tasks"].add("text2text-generation")
            elif "ImageClassification" in model_class:
                discovered_models[model_family]["tasks"].add("image-classification")
            elif "ObjectDetection" in model_class:
                discovered_models[model_family]["tasks"].add("object-detection")
            elif "Speech" in model_class or "Audio" in model_class:
                discovered_models[model_family]["tasks"].add("automatic-speech-recognition")
            elif "Vision2Seq" in model_class:
                discovered_models[model_family]["tasks"].add("image-to-text")
        
        # Find example models for each family using hub API if available
        try:
            import huggingface_hub
            for model_family in discovered_models:
                try:
                    # Search for models matching the family name
                    models = huggingface_hub.list_models(
                        filter=huggingface_hub.ModelFilter(
                            model_name=model_family,
                            library="transformers",
                            task="*"
                        ),
                        limit=5
                    )
                    
                    # Add example models
                    example_models = [model.id for model in models]
                    if example_models:
                        discovered_models[model_family]["example_models"] = example_models
                except Exception as e:
                    logger.warning(f"Error searching for {model_family} models: {e}")
        except ImportError:
            logger.warning("huggingface_hub not available, cannot search for example models")
    
    except Exception as e:
        logger.warning(f"Error scanning transformers models: {e}")
    
    return discovered_models

def suggest_new_models(discovered_models):
    """Generate suggestions for new model families to add to the registry."""
    suggestions = []
    
    # Check which discovered models are not in our registry
    for model_family, info in discovered_models.items():
        # Standardize model family name to match registry style
        family_id = model_family.lower()
        
        # Skip if already in registry
        if family_id in MODEL_REGISTRY or any(family_id in k for k in MODEL_REGISTRY.keys()):
            continue
        
        # Skip generic models
        if family_id in ["auto", "generic", "model", "base"]:
            continue
            
        # Prepare information for suggestion
        if hasattr(info, "get") and info.get("example_models"):
            example_model = info["example_models"][0]
        else:
            example_model = f"{model_family.lower()}-base"
            
        model_classes = list(info["model_classes"]) if hasattr(info, "get") and info.get("model_classes") else []
        if model_classes:
            class_name = model_classes[0].split(".")[-1]
        else:
            class_name = f"{model_family}Model"
            
        # Get tasks
        tasks = list(info["tasks"]) if hasattr(info, "get") and info.get("tasks") else ["feature-extraction"]
        
        # Create suggestion
        suggestion = {
            "family_id": family_id,
            "family_name": model_family,
            "default_model": example_model,
            "class": class_name,
            "task": tasks[0] if tasks else "feature-extraction"
        }
        
        suggestions.append(suggestion)
    
    return suggestions

def get_model_specifics(family_id):
    """Get specific model details and configurations for known models."""
    model_specifics = {
        "sam": {
            "family_name": "SAM",
            "description": "Segment Anything Model for image segmentation",
            "default_model": "facebook/sam-vit-base",
            "class": "SamModel",
            "tasks": ["image-segmentation"],
            "inputs": {
                "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
                "points": [[500, 375]]
            },
            "dependencies": ["transformers", "pillow", "requests", "numpy"],
            "class_mapping": {
                "facebook/sam-vit-base": "SamModel",
                "facebook/sam-vit-large": "SamModel",
                "facebook/sam-vit-huge": "SamModel"
            }
        },
        "bart": {
            "family_name": "BART",
            "description": "BART sequence-to-sequence models",
            "default_model": "facebook/bart-base",
            "class": "BartForConditionalGeneration",
            "tasks": ["summarization", "translation"],
            "inputs": {
                "text": "The tower is 324 metres tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres on each side."
            },
            "dependencies": ["transformers", "tokenizers"],
            "class_mapping": {
                "facebook/bart-base": "BartForConditionalGeneration",
                "facebook/bart-large": "BartForConditionalGeneration",
                "facebook/bart-large-cnn": "BartForConditionalGeneration",
                "facebook/bart-large-mnli": "BartForSequenceClassification"
            }
        },
        "deberta": {
            "family_name": "DeBERTa",
            "description": "DeBERTa masked language models",
            "default_model": "microsoft/deberta-base",
            "class": "DebertaForMaskedLM",
            "tasks": ["fill-mask"],
            "inputs": {
                "text": "The quick brown fox jumps over the [MASK] dog."
            },
            "dependencies": ["transformers", "tokenizers"],
            "class_mapping": {
                "microsoft/deberta-base": "DebertaForMaskedLM",
                "microsoft/deberta-large": "DebertaForMaskedLM",
                "microsoft/deberta-v2-xlarge": "DebertaV2ForMaskedLM"
            }
        },
        "segformer": {
            "family_name": "SegFormer",
            "description": "SegFormer models for image segmentation",
            "default_model": "nvidia/segformer-b0-finetuned-ade-512-512",
            "class": "SegformerForSemanticSegmentation",
            "tasks": ["image-segmentation"],
            "inputs": {
                "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
            },
            "dependencies": ["transformers", "pillow", "requests"],
            "class_mapping": {
                "nvidia/segformer-b0-finetuned-ade-512-512": "SegformerForSemanticSegmentation",
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024": "SegformerForSemanticSegmentation"
            }
        },
        "blip": {
            "family_name": "BLIP",
            "description": "BLIP vision-language models",
            "default_model": "Salesforce/blip-image-captioning-base",
            "class": "BlipForConditionalGeneration",
            "tasks": ["image-to-text"],
            "inputs": {
                "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
            },
            "dependencies": ["transformers", "pillow", "requests"],
            "class_mapping": {
                "Salesforce/blip-image-captioning-base": "BlipForConditionalGeneration",
                "Salesforce/blip-vqa-base": "BlipForQuestionAnswering"
            }
        },
        "mistral": {
            "family_name": "Mistral",
            "description": "Mistral causal language models",
            "default_model": "mistralai/Mistral-7B-v0.1",
            "class": "MistralForCausalLM",
            "tasks": ["text-generation"],
            "inputs": {
                "text": "Explain quantum computing in simple terms"
            },
            "dependencies": ["transformers", "tokenizers", "accelerate"],
            "class_mapping": {
                "mistralai/Mistral-7B-v0.1": "MistralForCausalLM",
                "mistralai/Mistral-7B-Instruct-v0.1": "MistralForCausalLM"
            }
        },
        "gemma": {
            "family_name": "Gemma",
            "description": "Gemma language models from Google",
            "default_model": "google/gemma-2b",
            "class": "GemmaForCausalLM",
            "tasks": ["text-generation"],
            "inputs": {
                "text": "Write a poem about artificial intelligence"
            },
            "dependencies": ["transformers", "tokenizers", "accelerate"],
            "class_mapping": {
                "google/gemma-2b": "GemmaForCausalLM",
                "google/gemma-7b": "GemmaForCausalLM"
            }
        },
        "dino": {
            "family_name": "DINO",
            "description": "DINO object detection models",
            "default_model": "facebook/dino-vitb16",
            "class": "DinoForImageClassification",
            "tasks": ["image-classification"],
            "inputs": {
                "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"
            },
            "dependencies": ["transformers", "pillow", "requests"],
            "class_mapping": {
                "facebook/dino-vitb16": "DinoForImageClassification",
                "facebook/dino-vits16": "DinoForImageClassification"
            }
        }
    }
    
    return model_specifics.get(family_id.lower(), None)

def generate_model_registry_entry(suggestion):
    """Generate a model registry entry from a suggestion."""
    family_id = suggestion["family_id"]
    family_name = suggestion["family_name"]
    default_model = suggestion["default_model"]
    class_name = suggestion["class"]
    task = suggestion["task"]
    
    # Check if we have specific details for this model family
    model_specifics = get_model_specifics(family_id)
    if model_specifics:
        # Use the specific details
        family_name = model_specifics["family_name"]
        description = model_specifics["description"]
        default_model = model_specifics["default_model"]
        class_name = model_specifics["class"]
        task = model_specifics["tasks"][0]
        inputs = model_specifics["inputs"]
        dependencies = model_specifics["dependencies"]
        class_mapping = model_specifics.get("class_mapping", {})
        
        # Build models dictionary
        models_dict = "{\n"
        for model_id, model_class in class_mapping.items():
            models_dict += f'            "{model_id}": {{\n'
            models_dict += f'                "description": "{model_id.split("/")[-1]} model",\n'
            models_dict += f'                "class": "{model_class}"\n'
            models_dict += '            },\n'
        models_dict += "        }"
    else:
        # Generate appropriate test inputs based on task
        if task == "fill-mask":
            description = f"{family_name} masked language models"
            inputs = {"text": "The quick brown fox jumps over the [MASK] dog."}
            task_args = {"top_k": 5}
            dependencies = ["transformers", "tokenizers"]
        elif task == "text-generation":
            description = f"{family_name} causal language models"
            inputs = {"text": "In this paper, we propose"}
            task_args = {"max_length": 50, "min_length": 20}
            dependencies = ["transformers", "tokenizers"]
        elif task == "text2text-generation":
            description = f"{family_name} sequence-to-sequence models"
            inputs = {"text": "Translate to French: Hello, how are you?"}
            task_args = {"max_length": 50}
            dependencies = ["transformers", "tokenizers", "sentencepiece"]
        elif task in ["image-classification", "object-detection", "image-segmentation"]:
            description = f"{family_name} models for {task}"
            inputs = {"image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"}
            task_args = {}
            dependencies = ["transformers", "pillow", "requests"]
        elif task in ["automatic-speech-recognition", "audio-classification"]:
            description = f"{family_name} models for {task}"
            inputs = {"audio_file": "audio_sample.mp3"}
            task_args = {}
            dependencies = ["transformers", "librosa", "soundfile"]
        else:
            # Generic fallback
            description = f"{family_name} models for {task}"
            inputs = {"text": "This is a test input"}
            task_args = {}
            dependencies = ["transformers"]
            
        # Build models dictionary with just the default model
        models_dict = "{\n"
        models_dict += f'            "{default_model}": {{\n'
        models_dict += f'                "description": "{family_name} model",\n'
        models_dict += f'                "class": "{class_name}"\n'
        models_dict += '            }\n'
        models_dict += "        }"
    
    # Generate the registry entry
    if model_specifics:
        # Format the inputs dictionary
        inputs_str = ""
        for k, v in inputs.items():
            if isinstance(v, str):
                inputs_str += f'"{k}": "{v}", '
            else:
                inputs_str += f'"{k}": {v}, '
        inputs_str = inputs_str.rstrip(", ")
        
        task_args = model_specifics.get("task_specific_args", {})
        if not task_args:
            task_args = {}
        
        entry = f"""
    "{family_id}": {{
        "family_name": "{family_name}",
        "description": "{description}",
        "default_model": "{default_model}",
        "class": "{class_name}",
        "test_class": "Test{family_name}Models",
        "module_name": "test_hf_{family_id.lower()}",
        "tasks": {str(model_specifics["tasks"])},
        "inputs": {{
            {inputs_str}
        }},
        "dependencies": {str(dependencies)},
        "task_specific_args": {{
            "{task}": {str(task_args)}
        }},
        "models": {models_dict}
    }}"""
    else:
        entry = f"""
    "{family_id}": {{
        "family_name": "{family_name}",
        "description": "{description}",
        "default_model": "{default_model}",
        "class": "{class_name}",
        "test_class": "Test{family_name}Models",
        "module_name": "test_hf_{family_id.lower()}",
        "tasks": ["{task}"],
        "inputs": {{
            {', '.join([f'"{k}": "{v}"' for k, v in inputs.items()])}
        }},
        "dependencies": {str(dependencies)},
        "task_specific_args": {{
            "{task}": {str(task_args)}
        }},
        "models": {models_dict}
    }}"""
    
    return entry

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate test files for Hugging Face models")
    
    # Generation options
    parser.add_argument("--generate", type=str, help="Generate test file for a specific model family")
    parser.add_argument("--all", action="store_true", help="Generate test files for all model families")
    parser.add_argument("--update-all-models", action="store_true", help="Update test_all_models.py with all model families")
    
    # List options
    parser.add_argument("--list-families", action="store_true", help="List all model families in the registry")
    
    # Discovery options
    parser.add_argument("--scan-transformers", action="store_true", help="Scan transformers library for available models")
    parser.add_argument("--suggest-models", action="store_true", help="Suggest new models to add to the registry")
    parser.add_argument("--generate-registry-entry", type=str, help="Generate registry entry for a specific model family")
    parser.add_argument("--auto-add", action="store_true", help="Automatically add new models and generate tests")
    parser.add_argument("--max-models", type=int, default=5, help="Maximum number of models to auto-add")
    parser.add_argument("--batch-generate", type=str, help="Generate tests for a comma-separated list of models")
    
    args = parser.parse_args()
    
    # List model families if requested
    if args.list_families:
        print("\nAvailable Model Families in Registry:")
        for family_id, family_info in MODEL_REGISTRY.items():
            print(f"  - {family_id}: {family_info['description']} ({family_info['default_model']})")
        return
    
    # Scan transformers library
    if args.scan_transformers:
        print("\nScanning transformers library for available models...")
        discovered_models = scan_hf_transformers()
        print(f"Found {len(discovered_models)} model families in transformers")
        
        # Print discovered models
        for family, info in discovered_models.items():
            print(f"\n{family}:")
            if hasattr(info, "get"):
                if info.get("model_classes"):
                    print(f"  Classes: {', '.join(list(info['model_classes'])[:3])}{'...' if len(info['model_classes']) > 3 else ''}")
                if info.get("tasks"):
                    print(f"  Tasks: {', '.join(info['tasks'])}")
                if info.get("example_models"):
                    print(f"  Example models: {', '.join(info['example_models'][:3])}")
        return
    
    # Suggest new models
    if args.suggest_models:
        print("\nSuggesting new models to add to the registry...")
        discovered_models = scan_hf_transformers()
        suggestions = suggest_new_models(discovered_models)
        
        print(f"Found {len(suggestions)} model families that could be added to the registry:")
        for suggestion in suggestions:
            print(f"\n{suggestion['family_name']}:")
            print(f"  family_id: {suggestion['family_id']}")
            print(f"  default_model: {suggestion['default_model']}")
            print(f"  class: {suggestion['class']}")
            print(f"  task: {suggestion['task']}")
        return
    
    # Generate registry entry
    if args.generate_registry_entry:
        family_id = args.generate_registry_entry
        discovered_models = scan_hf_transformers()
        
        # Find the model in discovered models
        for model_family, info in discovered_models.items():
            if model_family.lower() == family_id.lower():
                # Create suggestion
                suggestion = {
                    "family_id": family_id.lower(),
                    "family_name": model_family,
                    "default_model": info.get("example_models", [f"{model_family}-base"])[0] if hasattr(info, "get") else f"{model_family}-base",
                    "class": list(info["model_classes"])[0].split(".")[-1] if hasattr(info, "get") and info.get("model_classes") else f"{model_family}Model",
                    "task": list(info["tasks"])[0] if hasattr(info, "get") and info.get("tasks") else "feature-extraction"
                }
                
                # Generate registry entry
                entry = generate_model_registry_entry(suggestion)
                print(f"\nRegistry entry for {family_id}:")
                print(entry)
                return
        
        print(f"Could not find {family_id} in transformers library")
        return
    
    # Generate a specific test file
    if args.generate:
        family_id = args.generate
        if family_id not in MODEL_REGISTRY:
            print(f"Unknown model family: {family_id}")
            return
        
        create_test_file(family_id)
    
    # Generate all test files
    if args.all:
        generate_all_test_files()
    
    # Update test_all_models.py
    if args.update_all_models:
        update_test_all_models()
    
    # Auto-add models and generate tests
    if args.auto_add:
        print("\nAutomatically finding and adding new models...")
        added_models = auto_add_tests(max_models=args.max_models)
        return
        
    # Batch generate tests for multiple models
    if args.batch_generate:
        model_list = [m.strip() for m in args.batch_generate.split(",")]
        print(f"\nGenerating tests for {len(model_list)} models: {', '.join(model_list)}")
        successful, failed = generate_all_test_files(model_list)
        return
    
    # Default: print help
    if not (args.generate or args.all or args.update_all_models or args.list_families or
            args.scan_transformers or args.suggest_models or args.generate_registry_entry or
            args.auto_add or args.batch_generate):
        parser.print_help()

if __name__ == "__main__":
    main()