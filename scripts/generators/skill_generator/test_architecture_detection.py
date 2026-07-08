#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for architecture detection with specialized templates.

This script tests the mapping of model names to architecture types,
with a focus on the new specialized templates.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Import the architecture detector
from scripts.generators.architecture_detector import (
    get_architecture_type,
    get_model_metadata,
    get_model_class_name,
    get_default_model_id
)

def test_architecture_detection():
    print("Testing architecture detection with specialized templates...")
    
    # Test models for each architecture type
    test_models = {
        "encoder-only": [
            "bert-base-uncased",
            "roberta-base",
            "electra-base",
            "distilbert-base-uncased"
        ],
        "decoder-only": [
            "gpt2",
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
            "microsoft/phi-2"
        ],
        "encoder-decoder": [
            "t5-small",
            "facebook/bart-base",
            "google/mt5-small",
            "facebook/mbart-large-50"
        ],
        "text-to-image": [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "kandinsky-community/kandinsky-2-1",
            "dalle-mini"
        ],
        "protein-folding": [
            "facebook/esm2_t33_650M_UR50D",
            "facebook/esm1b_t33_650M_UR50S",
            "facebook/esmfold_v1",
            "prot-bert-base"
        ],
        "video-processing": [
            "MCG-NJU/videomae-base-finetuned-kinetics",
            "google/vivit-b-16x2",
            "facebook/timesformer-base-finetuned-k400",
            "video-processing-model"
        ]
    }
    
    # Test and report all models
    for expected_arch, models in test_models.items():
        print(f"\nTesting {expected_arch} models:")
        for model in models:
            detected_arch = get_architecture_type(model)
            model_class, processor_class = get_model_class_name(model, detected_arch)
            default_id = get_default_model_id(model, detected_arch)
            
            # Print comparison
            match = "✓" if detected_arch == expected_arch else "✗"
            print(f"  {match} {model:40} → {detected_arch:20} | {model_class:30} | {default_id}")
            
            if detected_arch != expected_arch:
                print(f"    ERROR: Expected {expected_arch}, got {detected_arch}")
    
    # Testing full metadata
    print("\nTesting full metadata for sample models:")
    sample_models = [
        "runwayml/stable-diffusion-v1-5",  # text-to-image
        "facebook/esm2_t33_650M_UR50D",    # protein-folding
        "MCG-NJU/videomae-base"            # video-processing
    ]
    
    for model in sample_models:
        metadata = get_model_metadata(model)
        print(f"\nMetadata for {model}:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_architecture_detection()