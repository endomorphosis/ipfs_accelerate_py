#!/usr/bin/env python3
"""
Comprehensive HuggingFace Hub Model Scraper

This script will scrape ALL models from the HuggingFace Hub and populate
the model manager with detailed metadata, performance characteristics,
and hardware compatibility information.

Features:
- Comprehensive scraping of all HuggingFace models
- Intelligent batching and rate limiting
- Progress saving and resume capability
- Hardware compatibility analysis
- Performance estimation
- Bandit algorithm integration for recommendations
- Offline mode with comprehensive mock data
"""

import os
import sys
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
from ipfs_accelerate_py.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def display_progress(processed: int, total: int, current_model: str = ""):
    """Display progress with nice formatting"""
    if total > 0:
        percentage = (processed / total) * 100
        bar_length = 50
        filled_length = int(bar_length * processed / total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'\r🔍 Progress: |{bar}| {percentage:.1f}% ({processed}/{total}) - {current_model[:50]}{"..." if len(current_model) > 50 else ""}', end='', flush=True)
    else:
        print(f'\r🔍 Processing: {processed} models - {current_model[:60]}{"..." if len(current_model) > 60 else ""}', end='', flush=True)

def populate_with_comprehensive_mock_data(model_manager: ModelManager) -> int:
    """Populate model manager with comprehensive mock data representing real HuggingFace models."""
    
    print("🔧 Network unavailable - populating with comprehensive mock HuggingFace model data...")
    
    # Comprehensive mock model data representing different categories from HuggingFace
    mock_models = [
        # Text Generation Models
        ("gpt2", "GPT-2", "text-generation", "transformer", "pytorch", 124, ["gpt", "text-generation", "pytorch"]),
        ("gpt2-medium", "GPT-2 Medium", "text-generation", "transformer", "pytorch", 345, ["gpt", "text-generation", "pytorch"]),
        ("gpt2-large", "GPT-2 Large", "text-generation", "transformer", "pytorch", 774, ["gpt", "text-generation", "pytorch"]),
        ("gpt2-xl", "GPT-2 XL", "text-generation", "transformer", "pytorch", 1558, ["gpt", "text-generation", "pytorch"]),
        ("EleutherAI/gpt-neo-1.3B", "GPT-Neo 1.3B", "text-generation", "transformer", "pytorch", 1300, ["gpt-neo", "eleutherai", "text-generation"]),
        ("EleutherAI/gpt-neo-2.7B", "GPT-Neo 2.7B", "text-generation", "transformer", "pytorch", 2700, ["gpt-neo", "eleutherai", "text-generation"]),
        ("EleutherAI/gpt-j-6B", "GPT-J 6B", "text-generation", "transformer", "pytorch", 6000, ["gpt-j", "eleutherai", "text-generation"]),
        ("microsoft/DialoGPT-medium", "DialoGPT Medium", "text-generation", "transformer", "pytorch", 345, ["dialog", "conversational", "microsoft"]),
        ("microsoft/DialoGPT-large", "DialoGPT Large", "text-generation", "transformer", "pytorch", 774, ["dialog", "conversational", "microsoft"]),
        ("bigscience/bloom-560m", "BLOOM 560M", "text-generation", "transformer", "pytorch", 560, ["bloom", "multilingual", "bigscience"]),
        ("bigscience/bloom-1b1", "BLOOM 1.1B", "text-generation", "transformer", "pytorch", 1100, ["bloom", "multilingual", "bigscience"]),
        ("bigscience/bloom-3b", "BLOOM 3B", "text-generation", "transformer", "pytorch", 3000, ["bloom", "multilingual", "bigscience"]),
        ("facebook/opt-125m", "OPT 125M", "text-generation", "transformer", "pytorch", 125, ["opt", "facebook", "text-generation"]),
        ("facebook/opt-350m", "OPT 350M", "text-generation", "transformer", "pytorch", 350, ["opt", "facebook", "text-generation"]),
        ("facebook/opt-1.3b", "OPT 1.3B", "text-generation", "transformer", "pytorch", 1300, ["opt", "facebook", "text-generation"]),
        
        # BERT Models
        ("bert-base-uncased", "BERT Base Uncased", "fill-mask", "bert", "pytorch", 110, ["bert", "base", "uncased", "fill-mask"]),
        ("bert-base-cased", "BERT Base Cased", "fill-mask", "bert", "pytorch", 110, ["bert", "base", "cased", "fill-mask"]),
        ("bert-large-uncased", "BERT Large Uncased", "fill-mask", "bert", "pytorch", 340, ["bert", "large", "uncased", "fill-mask"]),
        ("bert-large-cased", "BERT Large Cased", "fill-mask", "bert", "pytorch", 340, ["bert", "large", "cased", "fill-mask"]),
        ("google/bert-base-multilingual-cased", "BERT Base Multilingual", "fill-mask", "bert", "pytorch", 110, ["bert", "multilingual", "google"]),
        ("distilbert-base-uncased", "DistilBERT Base Uncased", "fill-mask", "distilbert", "pytorch", 66, ["distilbert", "distilled", "efficient"]),
        ("distilbert-base-cased", "DistilBERT Base Cased", "fill-mask", "distilbert", "pytorch", 66, ["distilbert", "distilled", "efficient"]),
        ("roberta-base", "RoBERTa Base", "fill-mask", "roberta", "pytorch", 125, ["roberta", "base", "facebook"]),
        ("roberta-large", "RoBERTa Large", "fill-mask", "roberta", "pytorch", 355, ["roberta", "large", "facebook"]),
        ("microsoft/deberta-v3-base", "DeBERTa V3 Base", "fill-mask", "deberta", "pytorch", 140, ["deberta", "microsoft", "v3"]),
        
        # Classification Models
        ("cardiffnlp/twitter-roberta-base-sentiment-latest", "Twitter Sentiment", "text-classification", "roberta", "pytorch", 125, ["sentiment", "twitter", "classification"]),
        ("microsoft/DialoGPT-medium", "DialoGPT Medium", "text-classification", "transformer", "pytorch", 345, ["dialog", "classification", "microsoft"]),
        ("unitary/toxic-bert", "Toxic BERT", "text-classification", "bert", "pytorch", 110, ["toxic", "content-moderation", "bert"]),
        ("ProsusAI/finbert", "FinBERT", "text-classification", "bert", "pytorch", 110, ["finance", "bert", "financial"]),
        ("nlptown/bert-base-multilingual-uncased-sentiment", "Multilingual Sentiment", "text-classification", "bert", "pytorch", 110, ["sentiment", "multilingual", "bert"]),
        
        # Question Answering Models
        ("deepset/roberta-base-squad2", "RoBERTa SQuAD2", "question-answering", "roberta", "pytorch", 125, ["qa", "squad", "roberta"]),
        ("distilbert-base-cased-distilled-squad", "DistilBERT SQuAD", "question-answering", "distilbert", "pytorch", 66, ["qa", "squad", "distilbert"]),
        ("bert-large-uncased-whole-word-masking-finetuned-squad", "BERT Large SQuAD", "question-answering", "bert", "pytorch", 340, ["qa", "squad", "bert"]),
        
        # Translation Models
        ("Helsinki-NLP/opus-mt-en-de", "OPUS MT EN-DE", "translation", "marian", "pytorch", 77, ["translation", "opus", "en-de"]),
        ("Helsinki-NLP/opus-mt-en-fr", "OPUS MT EN-FR", "translation", "marian", "pytorch", 77, ["translation", "opus", "en-fr"]),
        ("Helsinki-NLP/opus-mt-en-es", "OPUS MT EN-ES", "translation", "marian", "pytorch", 77, ["translation", "opus", "en-es"]),
        ("facebook/m2m100_418M", "M2M100 418M", "translation", "m2m100", "pytorch", 418, ["translation", "multilingual", "facebook"]),
        ("facebook/m2m100_1.2B", "M2M100 1.2B", "translation", "m2m100", "pytorch", 1200, ["translation", "multilingual", "facebook"]),
        
        # Summarization Models
        ("facebook/bart-large-cnn", "BART Large CNN", "summarization", "bart", "pytorch", 406, ["summarization", "bart", "facebook"]),
        ("google/pegasus-xsum", "PEGASUS XSum", "summarization", "pegasus", "pytorch", 568, ["summarization", "pegasus", "google"]),
        ("microsoft/prophetnet-large-uncased-cnndm", "ProphetNet CNN/DM", "summarization", "prophetnet", "pytorch", 400, ["summarization", "prophetnet", "microsoft"]),
        ("sshleifer/distilbart-cnn-12-6", "DistilBART CNN", "summarization", "bart", "pytorch", 306, ["summarization", "distilbart", "efficient"]),
        
        # Image Models
        ("google/vit-base-patch16-224", "Vision Transformer Base", "image-classification", "vision-transformer", "pytorch", 86, ["vision", "transformer", "google"]),
        ("google/vit-large-patch16-224", "Vision Transformer Large", "image-classification", "vision-transformer", "pytorch", 304, ["vision", "transformer", "google"]),
        ("microsoft/resnet-50", "ResNet-50", "image-classification", "resnet", "pytorch", 25, ["resnet", "image", "microsoft"]),
        ("microsoft/swin-base-patch4-window7-224", "Swin Transformer Base", "image-classification", "swin", "pytorch", 88, ["swin", "transformer", "microsoft"]),
        ("facebook/detr-resnet-50", "DETR ResNet-50", "object-detection", "detr", "pytorch", 41, ["object-detection", "detr", "facebook"]),
        ("hustvl/yolos-tiny", "YOLOS Tiny", "object-detection", "yolos", "pytorch", 6, ["object-detection", "yolo", "efficient"]),
        
        # Audio Models
        ("openai/whisper-tiny", "Whisper Tiny", "automatic-speech-recognition", "whisper", "pytorch", 37, ["whisper", "asr", "openai"]),
        ("openai/whisper-base", "Whisper Base", "automatic-speech-recognition", "whisper", "pytorch", 74, ["whisper", "asr", "openai"]),
        ("openai/whisper-small", "Whisper Small", "automatic-speech-recognition", "whisper", "pytorch", 244, ["whisper", "asr", "openai"]),
        ("openai/whisper-medium", "Whisper Medium", "automatic-speech-recognition", "whisper", "pytorch", 769, ["whisper", "asr", "openai"]),
        ("openai/whisper-large", "Whisper Large", "automatic-speech-recognition", "whisper", "pytorch", 1550, ["whisper", "asr", "openai"]),
        ("facebook/wav2vec2-base-960h", "Wav2Vec2 Base", "automatic-speech-recognition", "wav2vec2", "pytorch", 95, ["wav2vec2", "asr", "facebook"]),
        ("facebook/wav2vec2-large-960h", "Wav2Vec2 Large", "automatic-speech-recognition", "wav2vec2", "pytorch", 317, ["wav2vec2", "asr", "facebook"]),
        
        # Multimodal Models
        ("openai/clip-vit-base-patch32", "CLIP ViT Base", "zero-shot-image-classification", "clip", "pytorch", 151, ["clip", "multimodal", "openai"]),
        ("openai/clip-vit-large-patch14", "CLIP ViT Large", "zero-shot-image-classification", "clip", "pytorch", 427, ["clip", "multimodal", "openai"]),
        ("Salesforce/blip-image-captioning-base", "BLIP Base", "image-to-text", "blip", "pytorch", 223, ["blip", "captioning", "salesforce"]),
        ("Salesforce/blip-image-captioning-large", "BLIP Large", "image-to-text", "blip", "pytorch", 447, ["blip", "captioning", "salesforce"]),
        
        # Code Models
        ("microsoft/CodeBERT-base", "CodeBERT Base", "feature-extraction", "bert", "pytorch", 110, ["code", "bert", "microsoft"]),
        ("Salesforce/codet5-base", "CodeT5 Base", "text2text-generation", "t5", "pytorch", 220, ["code", "t5", "salesforce"]),
        ("Salesforce/codet5-large", "CodeT5 Large", "text2text-generation", "t5", "pytorch", 770, ["code", "t5", "salesforce"]),
        ("microsoft/graphcodebert-base", "GraphCodeBERT", "feature-extraction", "bert", "pytorch", 110, ["code", "graph", "microsoft"]),
        
        # Embeddings Models
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", "sentence-similarity", "sentence-transformers", "pytorch", 23, ["embeddings", "sentence-transformers", "efficient"]),
        ("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2", "sentence-similarity", "sentence-transformers", "pytorch", 110, ["embeddings", "sentence-transformers", "mpnet"]),
        ("sentence-transformers/paraphrase-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "sentence-similarity", "sentence-transformers", "pytorch", 23, ["embeddings", "paraphrase", "efficient"]),
        ("microsoft/mpnet-base", "MPNet Base", "fill-mask", "mpnet", "pytorch", 110, ["mpnet", "microsoft", "embeddings"]),
        
        # Specialized Models
        ("EleutherAI/gpt-neox-20b", "GPT-NeoX 20B", "text-generation", "transformer", "pytorch", 20000, ["gpt-neox", "large", "eleutherai"]),
        ("bigcode/starcoder", "StarCoder", "text-generation", "transformer", "pytorch", 15500, ["code", "starcoder", "bigcode"]),
        ("huggingface/CodeBERTa-small-v1", "CodeBERTa Small", "fill-mask", "roberta", "pytorch", 84, ["code", "roberta", "huggingface"]),
        ("microsoft/DialoGPT-small", "DialoGPT Small", "text-generation", "transformer", "pytorch", 117, ["dialog", "conversational", "microsoft"]),
        ("microsoft/prophetnet-large-uncased", "ProphetNet Large", "text2text-generation", "prophetnet", "pytorch", 400, ["prophetnet", "microsoft", "seq2seq"]),
    ]
    
    added_count = 0
    total_models = len(mock_models)
    
    for i, (model_id, model_name, task, arch, framework, size_mb, tags) in enumerate(mock_models):
        try:
            # Create comprehensive mock model data
            from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceModelInfo, ModelPerformanceData, HardwareCompatibility
            
            # Create detailed model info
            detailed_info = HuggingFaceModelInfo(
                model_id=model_id,
                model_name=model_name,
                description=f"Mock comprehensive model: {model_name}. A {arch} model for {task} tasks.",
                pipeline_tag=task,
                library_name="transformers",
                tags=tags,
                downloads=10000 + (i * 1000),  # Varying download counts
                likes=100 + (i * 10),  # Varying like counts
                created_at="2023-01-01T00:00:00.000Z",
                last_modified="2024-01-01T00:00:00.000Z",
                private=False,
                gated=False,
                config={
                    "model_type": arch,
                    "architectures": [arch.title()],
                    "hidden_size": 768 if size_mb < 1000 else 1024,
                    "num_attention_heads": 12 if size_mb < 1000 else 16,
                    "num_hidden_layers": 12 if size_mb < 1000 else 24,
                },
                model_size_mb=float(size_mb),
                architecture=arch,
                framework=framework
            )
            
            # Create performance data
            performance_data = ModelPerformanceData(
                model_id=model_id,
                inference_time_ms=50.0 + (size_mb / 100),  # Larger models = slower
                throughput_tokens_per_sec=max(10.0, 1000.0 / size_mb),  # Larger models = lower throughput
                memory_usage_mb=size_mb * 1.5,
                gpu_memory_mb=size_mb * 2.0,
                hardware_requirements={
                    "min_cpu_cores": 2 if size_mb < 1000 else 4,
                    "min_ram_gb": max(4, size_mb // 1000 * 2),
                    "recommended_gpu": "RTX 3080" if size_mb > 5000 else "GTX 1660" if size_mb > 1000 else "integrated"
                },
                benchmark_scores={
                    "accuracy": 0.85 + (min(size_mb, 10000) / 20000),  # Larger models generally more accurate
                    "perplexity": max(10.0, 100.0 - (size_mb / 100)),
                    "bleu_score": 0.3 + (min(size_mb, 5000) / 10000)
                }
            )
            
            # Create hardware compatibility
            compatibility_data = HardwareCompatibility(
                model_id=model_id,
                cpu_compatible=True,
                gpu_compatible=True,
                min_ram_gb=max(4.0, size_mb / 1000 * 2),
                min_vram_gb=max(2.0, size_mb / 1000 * 4),
                supported_accelerators=["cpu", "cuda"] + (["mps"] if framework == "pytorch" else []) + (["tpu"] if size_mb < 2000 else [])
            )
            
            # Use the scanner's method to add to model manager
            scanner = HuggingFaceHubScanner(model_manager=model_manager)
            scanner._add_model_to_manager(model_id, detailed_info, performance_data, compatibility_data)
            added_count += 1
            
            # Show progress
            display_progress(added_count, total_models, model_id)
            
        except Exception as e:
            logger.warning(f"Failed to add mock model {model_id}: {e}")
    
    print(f"\n✅ Added {added_count} comprehensive mock models to model manager")
    return added_count

def main():
    """Main function to scrape all HuggingFace models"""
    
    print("🚀 Comprehensive HuggingFace Hub Model Scraper")
    print("=" * 60)
    print("This will scrape ALL models from HuggingFace Hub and populate the model manager.")
    print("Note: This may take several hours for the complete hub.")
    print()
    
    # Initialize components
    try:
        model_manager = ModelManager()
        scanner = HuggingFaceHubScanner(
            model_manager=model_manager,
            cache_dir="./hf_model_cache",
            max_workers=10  # Limit concurrent requests
        )
        
        print("✅ Model Manager and Scanner initialized")
        print(f"📂 Cache directory: ./hf_model_cache")
        print()
        
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return 1
    
    # Ask user for scan parameters
    print("📋 Scan Configuration:")
    try:
        limit_input = input("🔢 Enter maximum models to scan (press Enter for ALL models): ").strip()
        if limit_input:
            limit = int(limit_input)
            print(f"   → Will scan up to {limit:,} models")
        else:
            limit = None
            print("   → Will scan ALL models (may take hours)")
            
        task_filter = input("🏷️  Filter by task type (press Enter for all tasks): ").strip()
        if task_filter:
            print(f"   → Will filter for task: {task_filter}")
        else:
            task_filter = None
            print("   → Will include all task types")
        
        # Ask about mock data mode    
        use_mock = input("🔧 Use comprehensive mock data mode? (y/N): ").strip().lower()
        if use_mock in ['y', 'yes']:
            print("   → Will use comprehensive mock data (70+ models)")
        else:
            print("   → Will attempt live HuggingFace Hub connection")
            
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return 0
    except ValueError:
        print("❌ Invalid number entered, using default limit of 1000")
        limit = 1000
        use_mock = 'n'
    
    print()
    print("🎯 Starting comprehensive scan...")
    print("   ⏰ This may take a while - progress will be saved automatically")
    print("   🛑 Press Ctrl+C to stop gracefully (progress will be saved)")
    print()
    
    # Start the scan
    start_time = time.time()
    try:
        # Check if we should use mock data or try real scraping
        if use_mock in ['y', 'yes']:
            # Use comprehensive mock data
            total_processed = populate_with_comprehensive_mock_data(model_manager)
            results = {
                'total_processed': total_processed,
                'total_cached': total_processed,
                'task_distribution': {},
                'architecture_distribution': {},
                'scan_duration': time.time() - start_time,
                'model_manager_count': len(model_manager.list_models()),
                'errors': 0
            }
        else:
            # Set up progress callback
            def progress_callback(processed: int, total: int, current_model: str):
                display_progress(processed, total, current_model)
            
            # Run the comprehensive scan
            results = scanner.scan_hub(
                limit=limit,
                task_filter=task_filter,
                save_progress=True,
                progress_callback=progress_callback
            )
        
        print()  # New line after progress bar
        print()
        print("🎉 Scan completed successfully!")
        print("=" * 60)
        
        # Display results
        total_processed = results.get('total_processed', 0)
        total_cached = results.get('total_cached', 0)
        task_dist = results.get('task_distribution', {})
        arch_dist = results.get('architecture_distribution', {})
        
        print(f"📊 Scan Results:")
        print(f"   • Total models processed: {total_processed:,}")
        print(f"   • Total models cached: {total_cached:,}")
        print(f"   • Scan duration: {time.time() - start_time:.1f} seconds")
        print()
        
        if task_dist:
            print("🏷️  Top Task Types:")
            sorted_tasks = sorted(task_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            for task, count in sorted_tasks:
                print(f"   • {task}: {count:,} models")
            print()
        
        if arch_dist:
            print("🏗️  Top Architectures:")
            sorted_archs = sorted(arch_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            for arch, count in sorted_archs:
                print(f"   • {arch}: {count:,} models")
            print()
        
        # Verify models are in the model manager
        all_models = model_manager.list_models()
        print(f"✅ Model Manager populated with {len(all_models):,} models")
        
        # Show some example models
        if all_models:
            print()
            print("📋 Sample Models Added:")
            for i, model in enumerate(all_models[:10]):  # Show more samples
                print(f"   {i+1}. {model.model_id} ({model.architecture})")
            if len(all_models) > 10:
                print(f"   ... and {len(all_models) - 10:,} more models")
        
        print()
        print("🎯 All models have been successfully scraped and added to the model manager!")
        print("🔍 You can now search and discover models in the MCP dashboard at:")
        print("   http://127.0.0.1:8900/mcp/models")
        print()
        print("💡 To start the MCP dashboard with model discovery:")
        print("   python -c \"from ipfs_accelerate_py.mcp_dashboard import MCPDashboard; MCPDashboard(port=8900).run()\"")
        
        return 0
        
    except KeyboardInterrupt:
        print()
        print()
        print("🛑 Scan interrupted by user")
        # Still show partial results
        try:
            all_models = model_manager.list_models()
            print(f"📊 Partial results: {len(all_models):,} models added to model manager")
        except:
            pass
        return 0
        
    except Exception as e:
        print()
        print(f"❌ Error during scan: {e}")
        logger.error(f"Scan failed: {e}", exc_info=True)
        
        # Try to show what we managed to get
        try:
            all_models = model_manager.list_models()
            if all_models:
                print(f"📊 Partial results: {len(all_models):,} models were added before the error")
        except:
            pass
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)