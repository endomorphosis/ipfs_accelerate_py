# Template Integration Documentation Index

## Overview Documents

1. [**README.md**](README.md) - High-level overview of the template integration system
2. [**Template Integration Guide**](TEMPLATE_INTEGRATION_GUIDE.md) - Comprehensive guide to the template integration process
3. [**Architecture Guide**](ARCHITECTURE_GUIDE.md) - Detailed explanation of the system's architecture

## User Guides

1. [**Command Reference**](COMMAND_REFERENCE.md) - Quick reference for common commands
2. [**Troubleshooting Guide**](TROUBLESHOOTING.md) - Solutions for common issues
3. [**Template Extension Guide**](TEMPLATE_EXTENSION_GUIDE.md) - How to extend the system with new models and architectures

## Reports and Summaries

1. [**Template Integration Summary**](template_integration_summary.md) - Summary of the integration results
2. [**Manual Models Analysis**](manual_models_analysis.md) - Analysis of manually created test files

## Core Components

1. `model_template_fixes.py` - Core script for template customization
2. `template_integration_workflow.py` - Orchestrates the integration process
3. `fix_template_issues.py` - Targeted script for fixing indentation issues
4. `apply_changes.py` - Applies generated files to the main codebase

## Templates Directory

Templates are located in `/skills/templates/` and include:

1. `encoder_only_template.py` - Template for encoder-only models (BERT, RoBERTa, etc.)
2. `decoder_only_template.py` - Template for decoder-only models (GPT-2, LLaMA, etc.)
3. `encoder_decoder_template.py` - Template for encoder-decoder models (T5, BART, etc.)
4. `vision_template.py` - Template for vision models (ViT, DeiT, etc.)
5. `vision_text_template.py` - Template for vision-text models (CLIP, BLIP, etc.)
6. `speech_template.py` - Template for speech models (Whisper, Wav2Vec2, etc.)
7. `multimodal_template.py` - Template for multimodal models (LLaVA, FLAVA, etc.)

## Fixed Models

Successfully integrated models include:

1. `layoutlmv2` (vision-encoder-text-decoder)
2. `layoutlmv3` (vision-encoder-text-decoder)
3. `clvp` (speech)
4. `bigbird` (encoder-decoder)
5. `seamless_m4t_v2` (speech)
6. `xlm_prophetnet` (encoder-decoder)

## Quick Links

- **Generate All Models**: `python model_template_fixes.py --generate-all --verify`
- **Fix Problematic Models**: `python fix_template_issues.py`
- **Run Full Workflow**: `python template_integration_workflow.py`
- **Verify Model**: `python model_template_fixes.py --verify-model MODEL`