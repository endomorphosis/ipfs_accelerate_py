#!/usr/bin/env python3
"""
Comprehensive HuggingFace Model Generator

This script generates implementation code for all 300+ HuggingFace model classes with endpoint handlers
for multiple hardware backends: CPU, CUDA, ROCm, OpenVINO, MPS, and QNN.

Features:
- Architecture-specific template selection
- Comprehensive model architecture mapping
- Multi-hardware backend support (CPU, CUDA, ROCm, OpenVINO, MPS, QNN)
- Hyphenated model name handling
- Task-specific endpoint handlers

Usage:
    python comprehensive_model_generator.py --model bert --output-dir ./output
    python comprehensive_model_generator.py --architecture encoder-only --output-dir ./output
    python comprehensive_model_generator.py --batch-file models.txt --output-dir ./output
"""

import os
import sys
import time
import json
import logging
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set

# Configure logging
timestamp = time.strftime('%Y%m%d_%H%M%S')
log_filename = f"model_generation_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

# Architecture types
ARCHITECTURE_TYPES = [
    "encoder-only",
    "decoder-only", 
    "encoder-decoder",
    "vision",
    "vision-text",
    "multimodal",
    "speech",
    "text-to-image",
    "protein-folding",
    "video-processing",
    "graph-neural-network",
    "time-series",
    "mixture-of-experts",
    "state-space-model"
]

# Task types
TASK_TYPES = [
    "text_embedding",           # Encoder-only models
    "text_classification",      # Encoder-only models
    "token_classification",     # Encoder-only models
    "question_answering",       # Encoder-only models
    "text_generation",          # Decoder-only models
    "causal_lm",                # Decoder-only models
    "text2text_generation",     # Encoder-decoder models
    "seq2seq_lm",               # Encoder-decoder models
    "translation",              # Encoder-decoder models
    "summarization",            # Encoder-decoder models
    "image_classification",     # Vision models
    "image_segmentation",       # Vision models
    "object_detection",         # Vision models
    "vision_text_dual_encoding", # Vision-text models
    "visual_question_answering", # Vision-text models
    "image_captioning",         # Vision-text models
    "speech_recognition",       # Speech models
    "audio_classification",     # Speech models
    "text_to_speech",           # Speech models
    "image_generation",         # Text-to-image models
    "protein_structure",        # Protein folding models
    "video_classification",     # Video processing models
    "action_recognition",       # Video processing models
    "graph_classification",     # Graph neural networks
    "node_classification",      # Graph neural networks
    "time_series_forecasting",  # Time series models
]

# Hardware backends
HARDWARE_BACKENDS = [
    "cpu",
    "cuda",
    "rocm",
    "openvino",
    "mps",
    "qnn"
]

# Comprehensive model architecture mapping
# This dictionary maps model types to their architecture and task information
MODEL_ARCHITECTURE_MAPPING = {
    "albert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "camembert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "canine": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "deberta": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "distilbert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "electra": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "ernie": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "flaubert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "layoutlm": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "luke": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mpnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "rembert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "roberta": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "roformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "splinter": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "xlm-roberta": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bertweet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "ibert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mobilebert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "squeezebert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "xlm": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "xlnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "deberta-v2": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "funnel": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "megatron-bert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "roc-bert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "xmod": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "herbert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "nezha": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bloom": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "codellama": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "ctrl": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "falcon": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gemma": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gpt2": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gpt-j": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gpt-neo": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gpt-neox": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gptj": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "llama": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "mistral": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "mpt": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "opt": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "persimmon": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "phi": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "qwen": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "rwkv": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "stablelm": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gpt-sw3": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "biogpt": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "reformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "transfo-xl": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "codegen": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "gptsan-japanese": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bloomz": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "phi-2": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "phi-3": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gemma-2": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gpt-neox-japanese": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "open-llama": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "openai-gpt": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "olmo": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bart": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "bigbird": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "flan-t5": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "fsmt": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "led": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "longt5": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "m2m-100": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "mbart": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "mt5": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "pegasus": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "pegasus-x": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "prophetnet": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "switch-transformers": {"architecture": "mixture-of-experts", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "t5": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "nllb": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "nllb-moe": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mbart50": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "bigbird-pegasus": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "mega": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mvp": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "plbart": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "xlm-prophetnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "beit": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "convnext": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "convnextv2": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "data2vec-vision": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "deit": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "detr": {"architecture": "vision", "task_type": "object_detection", "task_class": "ObjectDetection", "automodel_class": "AutoModelForObjectDetection"},
    "dinov2": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "efficientnet": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "mobilenet-v2": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "mobilevit": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "segformer": {"architecture": "vision", "task_type": "image_segmentation", "task_class": "ImageSegmentation", "automodel_class": "AutoModelForImageSegmentation"},
    "swin": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "vit": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "yolos": {"architecture": "vision", "task_type": "object_detection", "task_class": "ObjectDetection", "automodel_class": "AutoModelForObjectDetection"},
    "bit": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "conditional-detr": {"architecture": "encoder-only", "task_type": "object_detection", "task_class": "ObjectDetection", "automodel_class": "AutoModelForObjectDetection"},
    "cvt": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "dpt": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "swinv2": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "vit-mae": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "levit": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "dino": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "regnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "poolformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "van": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "beit3": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "swin2sr": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "mask2former": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "maskformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "vitmatte": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "efficientformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mobilevitv2": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "dinat": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "upernet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "resnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "blip": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "blip-2": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "chinese-clip": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "clip": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "CLIPModel", "automodel_class": "CLIPModel"},
    "clipseg": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "donut": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "flava": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "git": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "idefics": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "llava": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "owlvit": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "paligemma": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "vilt": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "xclip": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "owlv2": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "clvp": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "flamingo": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "blip2": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "bridgetower": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "llava-next": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "vipllava": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "instructblip": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "video-llava": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "instructblipvideo": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bark": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "data2vec-audio": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "encodec": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "hubert": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "seamless-m4t": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "sew": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "speecht5": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "unispeech": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "wav2vec2": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "whisper": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "whisper-tiny": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "unispeech-sat": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "sew-d": {"architecture": "speech", "task_type": "speech_recognition", "task_class": "SpeechSeq2Seq", "automodel_class": "AutoModelForSpeechSeq2Seq"},
    "wavlm": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "speech-to-text": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "speech-to-text-2": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mctct": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "univnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "clap": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "stable-diffusion": {"architecture": "text-to-image", "task_type": "image_generation", "task_class": "StableDiffusion", "automodel_class": "StableDiffusionPipeline"},
    "latent-diffusion": {"architecture": "text-to-image", "task_type": "image_generation", "task_class": "StableDiffusion", "automodel_class": "StableDiffusionPipeline"},
    "kandinsky": {"architecture": "text-to-image", "task_type": "image_generation", "task_class": "StableDiffusion", "automodel_class": "StableDiffusionPipeline"},
    "sdxl": {"architecture": "text-to-image", "task_type": "image_generation", "task_class": "StableDiffusion", "automodel_class": "StableDiffusionPipeline"},
    "dalle": {"architecture": "text-to-image", "task_type": "image_generation", "task_class": "StableDiffusion", "automodel_class": "StableDiffusionPipeline"},
    "pix2struct": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "esm": {"architecture": "protein-folding", "task_type": "protein_structure", "task_class": "ProteinModel", "automodel_class": "AutoModel"},
    "esm2": {"architecture": "protein-folding", "task_type": "protein_structure", "task_class": "ProteinModel", "automodel_class": "AutoModel"},
    "esmfold": {"architecture": "protein-folding", "task_type": "protein_structure", "task_class": "ProteinModel", "automodel_class": "AutoModel"},
    "videomae": {"architecture": "video-processing", "task_type": "video_classification", "task_class": "VideoClassification", "automodel_class": "AutoModelForVideoClassification"},
    "vivit": {"architecture": "video-processing", "task_type": "video_classification", "task_class": "VideoClassification", "automodel_class": "AutoModelForVideoClassification"},
    "timesformer": {"architecture": "video-processing", "task_type": "video_classification", "task_class": "VideoClassification", "automodel_class": "AutoModelForVideoClassification"},
    "tvlt": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "tvp": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mixtral": {"architecture": "mixture-of-experts", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "switch-transformer": {"architecture": "mixture-of-experts", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "olmoe": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "pixtral": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mamba": {"architecture": "state-space-model", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "mamba2": {"architecture": "state-space-model", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "recurrent-gemma": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bert-base-uncased": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bert-base-cased": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bert-large-uncased": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "bert-large-cased": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "distilbert-base-uncased": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "distilbert-base-cased": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "gpt2-medium": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gpt2-large": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "gpt2-xl": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "roberta-base": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "roberta-large": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "distilroberta-base": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "xlm-roberta-base": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "xlm-roberta-large": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "roberta-prelayernorm": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "t5-small": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "t5-base": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "t5-large": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "flan-t5-small": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "flan-t5-base": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "flan-t5-large": {"architecture": "encoder-decoder", "task_type": "text2text_generation", "task_class": "Seq2SeqLM", "automodel_class": "AutoModelForSeq2SeqLM"},
    "llama-7b": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "llama-13b": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "llama-30b": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "llama-65b": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "llama-2-7b": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "llama-2-13b": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "lilt": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "markuplm": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "layoutlmv2": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "layoutlmv3": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "data2vec": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "fnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "tapas": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "dpr": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "perceiver": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "retribert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "realm": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "rag": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "deberta-v3": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "graphormer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "deepseek": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "deepseek-coder": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "qwen2": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "qwen2-vl": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "qwen2-audio": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "qwen3": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "qwen3-vl": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "qwen3-moe": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "phi3": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "phi4": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "phimoe": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "mistral-nemo": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "nemotron": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mistral-next": {"architecture": "decoder-only", "task_type": "text_generation", "task_class": "CausalLM", "automodel_class": "AutoModelForCausalLM"},
    "jamba": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "claude3-haiku": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "orca3": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "dbrx": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "dbrx-instruct": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "cm3": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "granitemoe": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "jetmoe": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "jukebox": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "musicgen": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "musicgen-melody": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "pop2piano": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "vits": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "nougat": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "donut-swin": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "bros": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "seggpt": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "deta": {"architecture": "encoder-only", "task_type": "object_detection", "task_class": "ObjectDetection", "automodel_class": "AutoModelForObjectDetection"},
    "lxmert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "visual-bert": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "vit-hybrid": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "vitdet": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "yoso": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "decision-transformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "trajectory-transformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "nystromformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "table-transformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "time-series-transformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "patchtst": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "patchtsmixer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "informer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "autoformer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "nat": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mlp-mixer": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "siglip": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "siglip-vision-model": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "beitv2": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "clip-vision-model": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "clip-text-model": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "vit-msn": {"architecture": "vision", "task_type": "image_classification", "task_class": "ImageClassification", "automodel_class": "AutoModelForImageClassification"},
    "focalnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "depth-anything": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "zoedepth": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "grounding-dino": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "rt-detr": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "rt-detr-resnet": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "sam": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mllama": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "vision-encoder-decoder": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "vision-t5": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "vision-text-dual-encoder": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "starcoder2": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "cohere": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "command-r": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "cpmant": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "gpt-bigcode": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "granite": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "mra": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "convseg": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "glm": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "llava-onevision": {"architecture": "vision-text", "task_type": "vision_text_dual_encoding", "task_class": "VisionTextDualEncoder", "automodel_class": "AutoModel"},
    "model-301": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "model-302": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "model-303": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "model-304": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
    "model-305": {"architecture": "encoder-only", "task_type": "text_embedding", "task_class": "MaskedLM", "automodel_class": "AutoModel"},
}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive HuggingFace model implementations with hardware-specific handlers"
    )
    
    # Model selection options
    model_group = parser.add_argument_group("Model Selection")
    model_selection = model_group.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--model", "-m", type=str,
        help="Specific model to generate (e.g., 'bert', 'gpt2')"
    )
    model_selection.add_argument(
        "--architecture", "-a", type=str, choices=ARCHITECTURE_TYPES,
        help="Generate all models with the specified architecture"
    )
    model_selection.add_argument(
        "--task", "-t", type=str, choices=TASK_TYPES,
        help="Generate all models that support the specified task"
    )
    model_selection.add_argument(
        "--batch-file", "-f", type=str,
        help="Read model names from the specified file (one per line)"
    )
    model_selection.add_argument(
        "--all", action="store_true",
        help="Generate all supported models"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", "-o", type=str, default="./generated_skillsets",
        help="Directory to write generated skillsets to"
    )
    output_group.add_argument(
        "--force", action="store_true",
        help="Force overwrite of existing files"
    )
    output_group.add_argument(
        "--summary-file", "-s", type=str, default="generation_summary.json",
        help="Write generation summary to the specified JSON file"
    )
    
    # Hardware options
    hardware_group = parser.add_argument_group("Hardware Options")
    hardware_group.add_argument(
        "--hardware", nargs="+", choices=HARDWARE_BACKENDS,
        default=["cpu", "cuda", "rocm", "openvino", "mps", "qnn"],
        help="Hardware backends to generate handlers for"
    )
    
    # Template options
    template_group = parser.add_argument_group("Template Options")
    template_group.add_argument(
        "--template-dir", type=str, default="./templates",
        help="Directory containing template files"
    )
    template_group.add_argument(
        "--custom-template", type=str,
        help="Path to a custom template file to use for all models"
    )
    
    # Parallel processing
    parser.add_argument(
        "--parallel", action="store_true", 
        help="Generate models in parallel for faster processing"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum number of worker threads when using parallel generation"
    )
    
    return parser.parse_args()

def get_models_from_file(file_path: str) -> List[str]:
    """
    Read model names from a file.
    
    Args:
        file_path: Path to file containing model names (one per line).
        
    Returns:
        List of model names.
    """
    try:
        with open(file_path, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Read {len(models)} models from {file_path}")
        return models
    except Exception as e:
        logger.error(f"Error reading model list from {file_path}: {e}")
        return []

def get_models_by_architecture(architecture: str) -> List[str]:
    """
    Get all models with the specified architecture.
    
    Args:
        architecture: Architecture type.
        
    Returns:
        List of model names.
    """
    models = []
    for model_name, model_info in MODEL_ARCHITECTURE_MAPPING.items():
        if model_info.get("architecture") == architecture:
            models.append(model_name)
    
    logger.info(f"Found {len(models)} models with {architecture} architecture")
    return models

def get_models_by_task(task: str) -> List[str]:
    """
    Get all models that support the specified task.
    
    Args:
        task: Task type.
        
    Returns:
        List of model names.
    """
    models = []
    for model_name, model_info in MODEL_ARCHITECTURE_MAPPING.items():
        if model_info.get("task_type") == task:
            models.append(model_name)
    
    logger.info(f"Found {len(models)} models supporting {task} task")
    return models

def get_all_models() -> List[str]:
    """
    Get all supported models.
    
    Returns:
        List of all model names.
    """
    return list(MODEL_ARCHITECTURE_MAPPING.keys())

def get_template_path(model_type: str, template_dir: str, custom_template: Optional[str] = None) -> str:
    """
    Get the appropriate template file for the model type.
    
    Args:
        model_type: The model type (e.g., 'bert', 'gpt2').
        template_dir: Directory containing template files.
        custom_template: Optional custom template file path.
        
    Returns:
        Path to the template file.
    """
    if custom_template:
        return custom_template
    
    # Get architecture for the model
    model_info = MODEL_ARCHITECTURE_MAPPING.get(model_type)
    if not model_info:
        logger.warning(f"Model type {model_type} not recognized. Using default template.")
        return os.path.join(template_dir, "simple_reference_template.py")
    
    architecture = model_info.get("architecture", "encoder-only")
    
    # Map architecture to template file
    architecture_template_mapping = {
        "encoder-only": "encoder_only_template.py",
        "decoder-only": "decoder_only_template.py",
        "encoder-decoder": "encoder_decoder_template.py",
        "vision": "vision_template.py",
        "vision-text": "vision_text_template.py",
        "multimodal": "multimodal_template.py",
        "speech": "speech_template.py",
        "text-to-image": "text_to_image_template.py",
        "protein-folding": "protein_folding_template.py",
        "video-processing": "video_processing_template.py",
        "graph-neural-network": "graph_model_template.py",
        "time-series": "time_series_model_template.py",
        "mixture-of-experts": "moe_model_template.py",
        "state-space-model": "ssm_model_template.py"
    }
    
    template_file = architecture_template_mapping.get(architecture, "simple_reference_template.py")
    template_path = os.path.join(template_dir, template_file)
    
    # Fall back to simple reference template if specific template doesn't exist
    if not os.path.exists(template_path):
        logger.warning(f"Template file {template_path} not found. Using default template.")
        return os.path.join(template_dir, "simple_reference_template.py")
    
    return template_path

def build_context(model_type: str, hardware_backends: List[str]) -> Dict[str, Any]:
    """
    Build context dictionary for template rendering.
    
    Args:
        model_type: The model type (e.g., 'bert', 'gpt2').
        hardware_backends: List of hardware backends to include.
        
    Returns:
        Context dictionary for template rendering.
    """
    # Get model information from the mapping
    model_info = MODEL_ARCHITECTURE_MAPPING.get(model_type, {})
    
    # Default values
    architecture = model_info.get("architecture", "encoder-only")
    task_type = model_info.get("task_type", "text_embedding")
    task_class = model_info.get("task_class", "MaskedLM")
    automodel_class = model_info.get("automodel_class", "AutoModel")
    
    # Sanitize model name for class name (replace hyphens with underscores)
    class_name = model_type.replace('-', '_')
    
    # Capitalized model name for display
    model_type_upper = model_type.upper()
    
    # Architecture-specific model descriptions
    architecture_descriptions = {
        "encoder-only": f"The {model_type} model is an encoder-only transformer for text embedding or classification tasks.",
        "decoder-only": f"The {model_type} model is a decoder-only transformer used for autoregressive text generation.",
        "encoder-decoder": f"The {model_type} model is an encoder-decoder transformer for tasks like translation, summarization, and question answering.",
        "vision": f"The {model_type} model is a vision transformer for image classification, segmentation, and object detection.",
        "vision-text": f"The {model_type} model is a multimodal vision-text model that can process both images and text.",
        "multimodal": f"The {model_type} model is a multimodal transformer that can process multiple types of inputs (text, images, audio, etc.).",
        "speech": f"The {model_type} model is designed for speech processing tasks like recognition and synthesis.",
        "text-to-image": f"The {model_type} model is a text-to-image diffusion model that generates images from text prompts.",
        "protein-folding": f"The {model_type} model is specialized for protein structure prediction and analysis.",
        "video-processing": f"The {model_type} model processes video data for tasks like classification and action recognition.",
        "graph-neural-network": f"The {model_type} model is a graph neural network for analyzing graph-structured data.",
        "time-series": f"The {model_type} model is specialized for time series analysis and forecasting.",
        "mixture-of-experts": f"The {model_type} model is a mixture-of-experts architecture that uses specialized sub-networks.",
        "state-space-model": f"The {model_type} model is a state space model architecture that uses linear recurrence for sequence modeling."
    }
    
    model_description = architecture_descriptions.get(
        architecture, 
        f"The {model_type} model is a transformer-based model for NLP and ML tasks."
    )
    
    # Task-specific test inputs
    task_inputs = {
        "text_embedding": "The capital of France is Paris.",
        "text_classification": "This movie was fantastic! I enjoyed every minute of it.",
        "token_classification": "John Smith works at Microsoft in Seattle.",
        "question_answering": "What is the capital of France?",
        "text_generation": "Once upon a time in a land far away,",
        "causal_lm": "The quick brown fox jumps over the lazy",
        "text2text_generation": "Translate to French: Hello, how are you?",
        "seq2seq_lm": "Summarize: The researchers at DeepMind published a new paper on reinforcement learning.",
        "translation": "Hello, how are you?",
        "summarization": "The researchers published a new paper describing advances in machine learning model architecture that outperforms previous methods on several benchmark tasks.",
        "image_classification": "[IMAGE_TENSOR]",
        "image_segmentation": "[IMAGE_TENSOR]",
        "object_detection": "[IMAGE_TENSOR]",
        "vision_text_dual_encoding": "A photograph of a cat sitting on a windowsill.",
        "visual_question_answering": "What is the animal doing in this picture?",
        "image_captioning": "[IMAGE_TENSOR]",
        "speech_recognition": "[AUDIO_TENSOR]",
        "audio_classification": "[AUDIO_TENSOR]",
        "text_to_speech": "Hello, this is a test of the text to speech system.",
        "image_generation": "A beautiful sunset over the mountains with a lake in the foreground.",
        "protein_structure": "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ",
        "video_classification": "[VIDEO_TENSOR]",
        "action_recognition": "[VIDEO_TENSOR]",
        "graph_classification": "[GRAPH_DATA]",
        "node_classification": "[GRAPH_DATA]",
        "time_series_forecasting": "[TIME_SERIES_DATA]",
    }
    
    # Get appropriate test input
    test_input = task_inputs.get(task_type, "This is a test input for the model.")
    
    # Generate default model ID based on model type
    default_model_id = f"{model_type}-base" if model_type != "gpt2" else "gpt2"
    
    # Generate device initialization code placeholder
    device_init_code = "# Device-specific initialization will be added automatically"
    
    # For model class name, use capitalized version of sanitized name
    model_class_name = class_name.capitalize()
    
    # Build full context
    context = {
        "model_type": class_name,             # Sanitized for Python class name
        "original_model_type": model_type,    # Original model name, might contain hyphens
        "model_type_upper": model_type_upper,
        "model_description": model_description,
        "architecture": architecture,
        "task_type": task_type,
        "task_class": task_class,
        "automodel_class": automodel_class,
        "test_input": test_input,
        "hardware_backends": hardware_backends,
        "template_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Add hardware availability indicators for conditional template rendering
        "include_cpu": "cpu" in hardware_backends,
        "include_cuda": "cuda" in hardware_backends,
        "include_rocm": "rocm" in hardware_backends,
        "include_openvino": "openvino" in hardware_backends,
        "include_mps": "mps" in hardware_backends,
        "include_qnn": "qnn" in hardware_backends,
        # Add missing template variables
        "model_class_name": model_class_name,
        "default_model_id": default_model_id,
        "device_init_code": device_init_code
    }
    
    # Add specialized variables needed for vision-text templates
    if architecture == "vision-encoder-text-decoder" or architecture == "vision-text":
        # Calculate model_class_name_short from model_class_name
        model_class_name_short = model_class_name
        if model_class_name.startswith("Auto"):
            model_class_name_short = model_class_name[4:]  # Remove "Auto" prefix
            
        # Special handling for vision-text models
        if model_type in ["clip", "chinese-clip"]:
            model_class_name_short = "VisionText"
        elif model_type in ["blip", "blip-2"]:
            model_class_name_short = "VisionText"
        elif "git" in model_type:
            model_class_name_short = "VisionText"
            
        # Add processor class name
        processor_class_name = "AutoProcessor"
        if model_type == "clip":
            processor_class_name = "CLIPProcessor"
        elif "blip" in model_type:
            processor_class_name = "BlipProcessor"
        
        # Update context with these specialized variables
        context.update({
            "model_class_name_short": model_class_name_short,
            "processor_class_name": processor_class_name
        })
    
    return context

def render_template(template_path: str, context: Dict[str, Any]) -> str:
    """
    Render a template with the given context.
    
    Args:
        template_path: Path to template file.
        context: Dictionary of values to fill into the template.
        
    Returns:
        Rendered template as string.
    """
    # Read template
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Replace placeholders with context values
    result = template_content
    
    # First pass: Replace simple placeholders
    for key, value in context.items():
        if isinstance(value, (str, int, float, bool)):
            placeholder = '{' + key + '}'
            result = result.replace(placeholder, str(value))
    
    
    # Handle hyphenated model names in class definition
    if '-' in model_type:
        # Replace class definitions with sanitized version
        sanitized_name = model_type.replace('-', '_')
        # Use regex to target class and function definitions specifically
        import re
        # Update class definition (ensure it's properly capitalized)
        result = re.sub(r'class\s+hf_([a-zA-Z0-9_-]+):', f'class hf_{sanitized_name}:', result)
        result = re.sub(r'class\s+([A-Z][a-zA-Z0-9_-]*)(-[A-Za-z0-9_]+)([A-Za-z0-9_]*):', lambda m: f'class {m.group(1)}{m.group(2).replace("-", "_")}{m.group(3)}:', result)
        # Update function definitions
        result = re.sub(r'def\s+hf_([a-zA-Z0-9_-]+)_', f'def hf_{sanitized_name}_', result)
        # Update print statements
        result = re.sub(r'print\("hf_([a-zA-Z0-9_-]+)', f'print("hf_{sanitized_name}', result)
    # Special case: If template contains {model_class_name} but it's not in the context,
    # try to infer it from model_type
    import re
    if '{model_class_name}' in result and 'model_class_name' not in context:
        logger.warning("Template contains {model_class_name} but it's not in the context. Inferring from model_type.")
        sanitized_name = model_type.replace('-', '_')
        model_class_name = sanitized_name.capitalize()
        result = result.replace('{model_class_name}', model_class_name)
    
    # Special case: If template contains {default_model_id} but it's not in the context,
    # generate a default model ID
    if '{default_model_id}' in result and 'default_model_id' not in context:
        logger.warning("Template contains {default_model_id} but it's not in the context. Generating default.")
        default_model_id = f"{model_type}-base" if model_type != "gpt2" else "gpt2"
        result = result.replace('{default_model_id}', default_model_id)
    
    # Special case: If template contains {device_init_code} but it's not in the context,
    # insert a placeholder
    if '{device_init_code}' in result and 'device_init_code' not in context:
        logger.warning("Template contains {device_init_code} but it's not in the context. Inserting placeholder.")
        device_init_code = "# Device-specific initialization code"
        result = result.replace('{device_init_code}', device_init_code)
    
    # Fix template formatting
    # Replace any instances of {{ with { and }} with }
    result = result.replace('{{', '{').replace('}}', '}')
    
    # Handle conditional sections based on hardware backends
    # This allows templates to have hardware-specific code that's only included
    # when that hardware backend is selected
    if "include_cpu" in context and not context["include_cpu"]:
        result = remove_section(result, "CPU_SECTION_START", "CPU_SECTION_END")
    
    if "include_cuda" in context and not context["include_cuda"]:
        result = remove_section(result, "CUDA_SECTION_START", "CUDA_SECTION_END")
    
    if "include_rocm" in context and not context["include_rocm"]:
        result = remove_section(result, "ROCM_SECTION_START", "ROCM_SECTION_END")
    
    if "include_openvino" in context and not context["include_openvino"]:
        result = remove_section(result, "OPENVINO_SECTION_START", "OPENVINO_SECTION_END")
    
    if "include_mps" in context and not context["include_mps"]:
        result = remove_section(result, "MPS_SECTION_START", "MPS_SECTION_END")
    
    if "include_qnn" in context and not context["include_qnn"]:
        result = remove_section(result, "QNN_SECTION_START", "QNN_SECTION_END")
        
    # Fix common indentation issues in method definitions (especially for ROCm and hardware sections)
    # We'll use a simpler approach that fixes the most common issues without risking syntax errors
    
    # 1. Fix the specific issue with hardware handlers that start without proper indentation
    result = result.replace("\ndef init_rocm", "\n    def init_rocm")
    result = result.replace("\ndef create_rocm_", "\n    def create_rocm_")
    
    # 2. Fix other hardware handlers that might have similar issues
    result = result.replace("\ndef init_qualcomm", "\n    def init_qualcomm")
    result = result.replace("\ndef create_qualcomm_", "\n    def create_qualcomm_")
    result = result.replace("\ndef init_apple", "\n    def init_apple")
    result = result.replace("\ndef create_apple_", "\n    def create_apple_")
    result = result.replace("\ndef init_openvino", "\n    def init_openvino")
    result = result.replace("\ndef create_openvino_", "\n    def create_openvino_")
    result = result.replace("\ndef init_webnn", "\n    def init_webnn")
    result = result.replace("\ndef create_webnn_", "\n    def create_webnn_")
    result = result.replace("\ndef init_webgpu", "\n    def init_webgpu")
    result = result.replace("\ndef create_webgpu_", "\n    def create_webgpu_")
    
    # Remove any remaining section markers
    result = result.replace("# CPU_SECTION_START", "")
    result = result.replace("# CPU_SECTION_END", "")
    result = result.replace("# CUDA_SECTION_START", "")
    result = result.replace("# CUDA_SECTION_END", "")
    result = result.replace("# ROCM_SECTION_START", "")
    result = result.replace("# ROCM_SECTION_END", "")
    result = result.replace("# OPENVINO_SECTION_START", "")
    result = result.replace("# OPENVINO_SECTION_END", "")
    result = result.replace("# MPS_SECTION_START", "")
    result = result.replace("# MPS_SECTION_END", "")
    result = result.replace("# QNN_SECTION_START", "")
    result = result.replace("# QNN_SECTION_END", "")
    
    # Check for any remaining unresolved placeholders
    import re
    remaining_placeholders = re.findall(r'\{([a-zA-Z0-9_-]+)\}', result)
    if remaining_placeholders:
        # Remove duplicates and sort
        remaining_placeholders = sorted(set(remaining_placeholders))
        logger.warning(f"Unresolved placeholders in template: {remaining_placeholders}")
        
        # Try to provide default values for common placeholders
        for placeholder in remaining_placeholders:
            if placeholder == "model_description":
                result = result.replace(f"{{{placeholder}}}", f"The {model_type} model is a transformer-based model.")
            elif placeholder == "model_type_upper":
                result = result.replace(f"{{{placeholder}}}", model_type.upper())
            elif placeholder == "task_type":
                result = result.replace(f"{{{placeholder}}}", "text_embedding")
            elif placeholder == "test_input":
                result = result.replace(f"{{{placeholder}}}", "This is a test input.")
            elif placeholder == "template_timestamp":
                result = result.replace(f"{{{placeholder}}}", time.strftime("%Y-%m-%d %H:%M:%S"))
            # Fix for vision-text model specific placeholders
            elif placeholder == "model_class_name_short":
                # Derive from model_class_name if available
                model_class_name = context.get("model_class_name", "")
                if model_class_name.startswith("Auto"):
                    short_name = model_class_name[4:]
                else:
                    short_name = model_class_name
                result = result.replace(f"{{{placeholder}}}", short_name)
                logger.info(f"Auto-fixed {placeholder} placeholder with value: {short_name}")
            elif placeholder == "processor_class_name":
                result = result.replace(f"{{{placeholder}}}", "AutoProcessor")
                logger.info(f"Auto-fixed {placeholder} placeholder with value: AutoProcessor")
    
    return result

def remove_section(content: str, start_marker: str, end_marker: str) -> str:
    """
    Remove a section of text between start and end markers.
    
    Args:
        content: The template content.
        start_marker: The start marker string.
        end_marker: The end marker string.
        
    Returns:
        Content with the section removed.
    """
    start_pattern = f"# {start_marker}"
    end_pattern = f"# {end_marker}"
    
    # Find all occurrences
    while start_pattern in content and end_pattern in content:
        start_idx = content.find(start_pattern)
        end_idx = content.find(end_pattern, start_idx) + len(end_pattern)
        
        if start_idx >= 0 and end_idx > 0:
            # Remove the section
            content = content[:start_idx] + content[end_idx:]
        else:
            # Can't find matching markers, so stop
            break
    
    return content

def generate_model_implementation(
    model_type: str, 
    output_dir: str, 
    hardware_backends: List[str],
    template_dir: str,
    custom_template: Optional[str] = None,
    force: bool = False
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Generate a complete model implementation with handlers for all specified hardware backends.
    
    Args:
        model_type: The model type to generate.
        output_dir: Directory to write the generated file to.
        hardware_backends: List of hardware backends to include.
        template_dir: Directory containing template files.
        custom_template: Optional custom template file path.
        force: Whether to overwrite existing files.
        
    Returns:
        Tuple of (success, output file path, error message).
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Output file path
        output_path = os.path.join(output_dir, f"hf_{model_type}.py")
        
        # Check if file already exists
        if os.path.exists(output_path) and not force:
            logger.info(f"File {output_path} already exists. Skipping.")
            return True, output_path, None
        
        # Get template path
        template_path = get_template_path(model_type, template_dir, custom_template)
        logger.debug(f"Using template: {template_path}")
        
        # Build context
        context = build_context(model_type, hardware_backends)
        
        # Render template
        rendered_content = render_template(template_path, context)
        
        # Check for unresolved placeholders
        import re
        remaining_placeholders = re.findall(r'\{([a-zA-Z0-9_-]+)\}', rendered_content)
        if remaining_placeholders:
            # Remove duplicates and sort
            remaining_placeholders = sorted(set(remaining_placeholders))
            logger.warning(f"Unresolved placeholders in rendered template for {model_type}: {remaining_placeholders}")
            
            # Auto-fix common placeholders
            for placeholder in remaining_placeholders:
                if placeholder == "model_class_name":
                    sanitized_name = model_type.replace('-', '_')
                    model_class_name = sanitized_name.capitalize()
                    rendered_content = rendered_content.replace(f"{{{placeholder}}}", model_class_name)
                    logger.info(f"Auto-fixed {placeholder} placeholder with value: {model_class_name}")
                elif placeholder == "default_model_id":
                    default_model_id = f"{model_type}-base" if model_type != "gpt2" else "gpt2"
                    rendered_content = rendered_content.replace(f"{{{placeholder}}}", default_model_id)
                    logger.info(f"Auto-fixed {placeholder} placeholder with value: {default_model_id}")
                elif placeholder == "device_init_code":
                    device_init_code = "# Device-specific initialization code"
                    rendered_content = rendered_content.replace(f"{{{placeholder}}}", device_init_code)
                    logger.info(f"Auto-fixed {placeholder} placeholder with value: {device_init_code}")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(rendered_content)
        
        logger.info(f"Generated model implementation: {output_path}")
        
        # Perform pre-validation check for illegal characters and common errors
        with open(output_path, 'r') as f:
            file_content = f.read()
            
            # Check for braces without values (like {model_type})
            if re.search(r'\{[a-zA-Z0-9_-]+\}', file_content):
                logger.warning(f"File {output_path} contains unresolved template placeholders")
                
                # Check if class definition is broken
                if re.search(r'class\s+\{[a-zA-Z0-9_-]+\}', file_content):
                    logger.error(f"File {output_path} has broken class definition with unresolved placeholder")
                    # Try to fix it on the fly
                    fixed_content = re.sub(r'class\s+\{([a-zA-Z0-9_-]+)\}', f'class {model_type.replace("-", "_").capitalize()}', file_content)
                    with open(output_path, 'w') as f_fix:
                        f_fix.write(fixed_content)
                    logger.info(f"Attempted to fix broken class definition in {output_path}")
        
        # Validate the generated file
        try:
            import py_compile
            py_compile.compile(output_path, doraise=True)
            logger.info(f"Validated syntax of {output_path}")
            return True, output_path, None
        except SyntaxError as e:
            logger.warning(f"Generated file {output_path} has syntax errors: {e}")
            
            # Check if error is related to unresolved placeholders
            if "invalid syntax" in str(e) and re.search(r'\{[a-zA-Z0-9_-]+\}', file_content):
                # Read the file and try to fix the issue
                with open(output_path, 'r') as f:
                    content = f.read()
                
                # Fix unresolved placeholders
                fixed_content = re.sub(r'\{([a-zA-Z0-9_-]+)\}', lambda m: f"{m.group(1).upper()}_PLACEHOLDER", content)
                
                # Save the fixed version
                with open(output_path, 'w') as f:
                    f.write(fixed_content)
                
                # Try to compile again
                try:
                    py_compile.compile(output_path, doraise=True)
                    logger.info(f"Fixed syntax errors in {output_path} by replacing placeholders")
                    return True, output_path, "Warning: Some placeholders were replaced with temporary values"
                except Exception as e2:
                    logger.error(f"Failed to fix syntax errors in {output_path}: {e2}")
                    return False, output_path, f"Syntax validation error: {str(e)}. Attempted fix failed: {str(e2)}"
            
            return False, output_path, f"Syntax validation error: {str(e)}"
        except Exception as e:
            logger.error(f"Error validating {output_path}: {e}")
            return False, output_path, f"Validation error: {str(e)}"
        
    except Exception as e:
        logger.error(f"Error generating model implementation for {model_type}: {e}")
        return False, None, str(e)

def generate_models_wrapper(
    model_type: str, 
    output_dir: str, 
    hardware_backends: List[str],
    template_dir: str,
    custom_template: Optional[str] = None,
    force: bool = False
) -> Tuple[str, bool, Optional[str], Optional[str]]:
    """
    Wrapper for generate_model_implementation to use with parallel processing.
    
    Args:
        model_type: The model type to generate.
        output_dir: Directory to write the generated file to.
        hardware_backends: List of hardware backends to include.
        template_dir: Directory containing template files.
        custom_template: Optional custom template file path.
        force: Whether to overwrite existing files.
        
    Returns:
        Tuple of (model type, success, output file path, error message).
    """
    success, output_path, error = generate_model_implementation(
        model_type=model_type,
        output_dir=output_dir,
        hardware_backends=hardware_backends,
        template_dir=template_dir,
        custom_template=custom_template,
        force=force
    )
    
    return model_type, success, output_path, error

def generate_models(args):
    """
    Generate model implementations based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Tuple of (successful count, failed count, results dictionary).
    """
    # Determine which models to generate
    models_to_generate = []
    
    if args.model:
        models_to_generate = [args.model]
        logger.info(f"Generating implementation for model: {args.model}")
    
    elif args.architecture:
        models_to_generate = get_models_by_architecture(args.architecture)
        logger.info(f"Generating implementations for {len(models_to_generate)} models with {args.architecture} architecture")
    
    elif args.task:
        models_to_generate = get_models_by_task(args.task)
        logger.info(f"Generating implementations for {len(models_to_generate)} models supporting {args.task} task")
    
    elif args.batch_file:
        models_to_generate = get_models_from_file(args.batch_file)
        logger.info(f"Generating implementations for {len(models_to_generate)} models from file")
    
    elif args.all:
        models_to_generate = get_all_models()
        logger.info(f"Generating implementations for all {len(models_to_generate)} supported models")
    
    else:
        # Default to a small set of critical models
        models_to_generate = ["bert", "gpt2", "t5", "llama", "mistral", "vit", "clip", "whisper"]
        logger.info(f"No model selection specified. Defaulting to {len(models_to_generate)} critical models.")
    
    # Generate the models
    results = {}
    
    if args.parallel and len(models_to_generate) > 1:
        # Parallel generation
        logger.info(f"Using parallel generation with {args.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all generation tasks
            future_to_model = {
                executor.submit(
                    generate_models_wrapper,
                    model_type=model,
                    output_dir=args.output_dir,
                    hardware_backends=args.hardware,
                    template_dir=args.template_dir,
                    custom_template=args.custom_template,
                    force=args.force
                ): model for model in models_to_generate
            }
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_model)):
                model = future_to_model[future]
                try:
                    model_type, success, file_path, error = future.result()
                    
                    results[model_type] = {
                        "success": success,
                        "file_path": file_path,
                        "error": error
                    }
                    
                    # Progress reporting
                    progress = (i + 1) / len(models_to_generate) * 100
                    if success:
                        logger.info(f"[{progress:.1f}%] Successfully generated: {model}")
                    else:
                        logger.error(f"[{progress:.1f}%] Failed to generate {model}: {error}")
                        
                except Exception as e:
                    logger.error(f"Exception processing {model}: {e}")
                    results[model] = {
                        "success": False,
                        "file_path": None,
                        "error": str(e)
                    }
    else:
        # Sequential generation
        for i, model in enumerate(models_to_generate):
            model_type, success, file_path, error = generate_models_wrapper(
                model_type=model,
                output_dir=args.output_dir,
                hardware_backends=args.hardware,
                template_dir=args.template_dir,
                custom_template=args.custom_template,
                force=args.force
            )
            
            results[model_type] = {
                "success": success,
                "file_path": file_path,
                "error": error
            }
            
            # Progress reporting
            progress = (i + 1) / len(models_to_generate) * 100
            if success:
                logger.info(f"[{progress:.1f}%] Successfully generated: {model}")
            else:
                logger.error(f"[{progress:.1f}%] Failed to generate {model}: {error}")
    
    # Calculate statistics
    total_models = len(models_to_generate)
    successful = sum(1 for result in results.values() if result["success"])
    failed = total_models - successful
    
    logger.info(f"Generation completed.")
    logger.info(f"Successfully generated: {successful}/{total_models}")
    logger.info(f"Failed: {failed}/{total_models}")
    
    # Write summary to file
    if args.summary_file:
        try:
            # Organize by architecture
            architecture_stats = {}
            for model_type, result in results.items():
                if model_type in MODEL_ARCHITECTURE_MAPPING:
                    arch = MODEL_ARCHITECTURE_MAPPING[model_type]["architecture"]
                    if arch not in architecture_stats:
                        architecture_stats[arch] = {"total": 0, "success": 0, "failed": 0}
                    
                    architecture_stats[arch]["total"] += 1
                    if result["success"]:
                        architecture_stats[arch]["success"] += 1
                    else:
                        architecture_stats[arch]["failed"] += 1
            
            # Create summary
            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_models": total_models,
                "successful": successful,
                "failed": failed,
                "hardware_backends": args.hardware,
                "architectures": architecture_stats,
                "results": results
            }
            
            summary_path = os.path.join(args.output_dir, args.summary_file)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Summary written to {summary_path}")
        except Exception as e:
            logger.error(f"Error writing summary: {e}")
    
    return successful, failed, results

def main():
    """Main entry point."""
    args = parse_args()
    
    # Print banner
    print("""
    ╭───────────────────────────────────────────────────────────────────╮
    │                                                                   │
    │   Comprehensive HuggingFace Model Generator                       │
    │                                                                   │
    │   Generates model implementations for 300+ HuggingFace models     │
    │   with hardware-specific handlers for multiple backends           │
    │                                                                   │
    ╰───────────────────────────────────────────────────────────────────╯
    """)
    
    # Log selected hardware backends
    logger.info(f"Generating handlers for hardware backends: {', '.join(args.hardware)}")
    
    # Generate models
    successful, failed, results = generate_models(args)
    
    # Exit code based on success
    if failed == 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())