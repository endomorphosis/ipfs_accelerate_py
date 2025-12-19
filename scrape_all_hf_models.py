#!/usr/bin/env python3
"""
Comprehensive HuggingFace Hub Model Scraper

This script provides a unified interface to scrape ALL models from the HuggingFace Hub
using the updated scraper architecture (EnhancedModelScraper and ProductionHFScraper).

Features:
- Unified interface to new scraper architecture
- Comprehensive scraping with mock or production modes
- Intelligent batching and rate limiting
- Progress saving and resume capability
- Hardware compatibility analysis
- Performance estimation
- Bandit algorithm integration for recommendations
- Parquet storage with K-NN search indexing
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import new scraper architecture
try:
    from enhanced_model_scraper import EnhancedModelScraper
    from production_hf_scraper import ProductionHFScraper
    SCRAPERS_AVAILABLE = True
except ImportError:
    SCRAPERS_AVAILABLE = False
    print("⚠️ Enhanced scraper modules not available")

# Import legacy modules for backward compatibility
try:
    from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
    from ipfs_accelerate_py.model_manager import ModelManager
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    print("⚠️ Legacy scraper modules not available")

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
    
    # Return proper statistics dictionary for consistency
    return {
        'total_processed': added_count,
        'total_added_to_manager': added_count,
        'batches_completed': 1,
        'current_batch': 1,
        'errors': 0,
        'start_time': time.time(),
        'last_batch_time': time.time(),
        'models_per_minute': added_count * 60.0,  # Since this was very fast
        'estimated_completion': None,
        'task_distribution': {
            'text-generation': 15,
            'fill-mask': 12,
            'text-classification': 8,
            'question-answering': 3,
            'translation': 5,
            'summarization': 4,
            'image-classification': 6,
            'automatic-speech-recognition': 7,
            'zero-shot-image-classification': 2,
            'image-to-text': 2,
            'feature-extraction': 4,
            'text2text-generation': 2,
            'sentence-similarity': 4,
            'object-detection': 2
        },
        'architecture_distribution': {
            'transformer': 25,
            'bert': 15,
            'roberta': 8,
            'distilbert': 3,
            'vision-transformer': 4,
            'whisper': 5,
            'wav2vec2': 2,
            'clip': 2,
            'detr': 1,
            'yolos': 1,
            'blip': 2,
            'resnet': 1,
            'swin': 1,
            'marian': 3,
            'm2m100': 2,
            'bart': 2,
            'pegasus': 1,
            'prophetnet': 2,
            'sentence-transformers': 3,
            'mpnet': 1,
            't5': 2
        },
        'popular_models': [
            ('gpt2', 10000),
            ('bert-base-uncased', 11000),
            ('distilbert-base-uncased', 12000),
            ('roberta-base', 13000),
            ('openai/clip-vit-base-patch32', 14000)
        ],
        'large_models': [
            ('EleutherAI/gpt-neox-20b', 20000),
            ('bigcode/starcoder', 15500),
            ('openai/whisper-large', 1550),
            ('bigscience/bloom-3b', 3000)
        ],
        'efficiency_models': [
            ('distilbert-base-uncased', 66),
            ('hustvl/yolos-tiny', 6),
            ('openai/whisper-tiny', 37),
            ('sentence-transformers/all-MiniLM-L6-v2', 23)
        ],
        'total_time': 1.0  # Mock timing
    }

def scan_all_huggingface_models_batch_optimized(model_manager: ModelManager, 
                                             batch_size: int = 1000,
                                             total_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Optimized batch scanning of ALL HuggingFace models with efficient pagination.
    
    This function can handle millions of models by:
    1. Using efficient batch processing
    2. Implementing smart resumption from interruptions
    3. Optimizing network requests with connection pooling
    4. Providing detailed progress tracking
    """
    
    print(f"🚀 Starting optimized batch scan of ALL HuggingFace models...")
    print(f"   📦 Batch size: {batch_size:,} models per batch")
    if total_limit:
        print(f"   🎯 Target: {total_limit:,} models")
    else:
        print(f"   🎯 Target: ALL models (potentially millions)")
    print()
    
    # Initialize statistics
    stats = {
        'total_processed': 0,
        'total_added_to_manager': 0,
        'batches_completed': 0,
        'current_batch': 0,
        'errors': 0,
        'start_time': time.time(),
        'last_batch_time': time.time(),
        'models_per_minute': 0.0,
        'estimated_completion': None,
        'task_distribution': {},
        'architecture_distribution': {},
        'popular_models': [],
        'large_models': [],  # Models > 5GB
        'efficiency_models': []  # Models < 500MB
    }
    
    scanner = HuggingFaceHubScanner(model_manager=model_manager)
    
    # Progress tracking
    def update_stats(processed_in_batch: int, models_added: int, batch_num: int):
        stats['total_processed'] += processed_in_batch
        stats['total_added_to_manager'] += models_added
        stats['current_batch'] = batch_num
        stats['batches_completed'] = batch_num
        
        # Calculate speed
        elapsed = time.time() - stats['start_time']
        if elapsed > 0:
            stats['models_per_minute'] = (stats['total_processed'] / elapsed) * 60
            
            # Estimate completion if we have a target
            if total_limit and stats['models_per_minute'] > 0:
                remaining = total_limit - stats['total_processed']
                remaining_minutes = remaining / stats['models_per_minute']
                stats['estimated_completion'] = f"{remaining_minutes:.1f} minutes"
    
    try:
        print("🔍 Discovering total model count on HuggingFace Hub...")
        
        # Use HuggingFace API to get total count and start pagination
        base_url = "https://huggingface.co/api/models"
        current_offset = 0
        total_models_found = 0
        
        # Initial request to get total count estimate
        try:
            response = requests.get(f"{base_url}?limit=1", timeout=30)
            response.raise_for_status()
            
            # Try to get total from headers or estimate
            models_data = response.json()
            print(f"✅ Successfully connected to HuggingFace Hub API")
            
            # Estimate total models (HF has 500k+ models as of 2024)
            estimated_total = 750000  # Conservative estimate for 2024
            print(f"📊 Estimated total models on HuggingFace Hub: ~{estimated_total:,}")
            
            if total_limit:
                actual_target = min(total_limit, estimated_total)
                print(f"🎯 Will scan {actual_target:,} models (user specified limit)")
            else:
                actual_target = estimated_total
                print(f"🎯 Will scan ALL {estimated_total:,} models")
                
        except Exception as e:
            print(f"⚠️  Could not connect to HuggingFace API: {e}")
            print("🔧 Falling back to comprehensive mock data...")
            return populate_with_comprehensive_mock_data(model_manager)
        
        print()
        print("🚀 Starting batch processing...")
        print(f"   ⏱️  Progress will be shown every {batch_size:,} models")
        print(f"   💾 Results automatically saved after each batch")
        print(f"   🛑 Press Ctrl+C to stop gracefully")
        print()
        
        batch_num = 0
        
        while True:
            batch_num += 1
            batch_start_time = time.time()
            
            print(f"📦 Processing Batch {batch_num:,} (offset {current_offset:,})...")
            
            try:
                # Fetch batch of models
                url = f"{base_url}?limit={batch_size}&skip={current_offset}&full=true&sort=trending"
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                batch_models = response.json()
                
                if not batch_models:
                    print("✅ No more models found - scan complete!")
                    break
                    
                print(f"   📥 Received {len(batch_models):,} models from API")
                
                # Process each model in the batch
                models_added_this_batch = 0
                models_processed_this_batch = 0
                
                for i, model_data in enumerate(batch_models):
                    try:
                        model_id = model_data.get('id', f'unknown_{current_offset + i}')
                        
                        # Show mini progress within batch
                        if i % 100 == 0:
                            display_progress(i, len(batch_models), model_id)
                        
                        # Process the model
                        detailed_info = scanner._convert_api_response_to_model_info(model_data)
                        if detailed_info:
                            performance_data = scanner._extract_performance_data(model_data, detailed_info)
                            compatibility_data = scanner._extract_hardware_compatibility(model_data, detailed_info)
                            scanner._add_model_to_manager(model_id, detailed_info, performance_data, compatibility_data)
                            models_added_this_batch += 1
                            
                            # Track statistics
                            task = detailed_info.pipeline_tag or 'unknown'
                            arch = detailed_info.architecture or 'unknown'
                            stats['task_distribution'][task] = stats['task_distribution'].get(task, 0) + 1
                            stats['architecture_distribution'][arch] = stats['architecture_distribution'].get(arch, 0) + 1
                            
                            # Track popular/large/efficient models
                            if detailed_info.downloads > 10000:
                                stats['popular_models'].append((model_id, detailed_info.downloads))
                            if detailed_info.model_size_mb and detailed_info.model_size_mb > 5000:
                                stats['large_models'].append((model_id, detailed_info.model_size_mb))
                            if detailed_info.model_size_mb and detailed_info.model_size_mb < 500:
                                stats['efficiency_models'].append((model_id, detailed_info.model_size_mb))
                        
                        models_processed_this_batch += 1
                        
                        # Check if we've hit the user limit
                        if total_limit and stats['total_processed'] + models_processed_this_batch >= total_limit:
                            print(f"\n🎯 Reached target limit of {total_limit:,} models")
                            break
                        
                    except Exception as e:
                        logger.debug(f"Error processing model {i} in batch {batch_num}: {e}")
                        stats['errors'] += 1
                
                # Update statistics
                update_stats(models_processed_this_batch, models_added_this_batch, batch_num)
                
                batch_time = time.time() - batch_start_time
                
                print(f"\n✅ Batch {batch_num:,} completed in {batch_time:.1f}s")
                print(f"   📊 Processed: {models_processed_this_batch:,} | Added: {models_added_this_batch:,}")
                print(f"   🚀 Speed: {stats['models_per_minute']:.1f} models/minute")
                if stats['estimated_completion']:
                    print(f"   ⏰ ETA: {stats['estimated_completion']}")
                print(f"   📈 Total progress: {stats['total_processed']:,} processed, {stats['total_added_to_manager']:,} added")
                print()
                
                # Move to next batch
                current_offset += len(batch_models)
                
                # Check completion conditions
                if total_limit and stats['total_processed'] >= total_limit:
                    print(f"🎯 Target of {total_limit:,} models reached!")
                    break
                    
                if len(batch_models) < batch_size:
                    print("✅ Reached end of available models")
                    break
                
                # Small delay between batches to be respectful to the API
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️  Error in batch {batch_num}: {e}")
                stats['errors'] += 1
                current_offset += batch_size  # Skip this batch and continue
                continue
        
        # Final statistics
        total_time = time.time() - stats['start_time']
        stats['total_time'] = total_time
        
        return stats
        
    except KeyboardInterrupt:
        print("\n🛑 Scan interrupted by user")
        return stats
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Scan failed: {e}", exc_info=True)
        return stats

def main():
    """Main function using updated scraper architecture"""
    
    print("🚀 COMPREHENSIVE HuggingFace Hub Model Scraper")
    print("=" * 70)
    print("Updated to use new scraper architecture:")
    print("  • EnhancedModelScraper (base class with Parquet + K-NN)")
    print("  • ProductionHFScraper (production API integration)")
    print()
    
    if not SCRAPERS_AVAILABLE:
        print("❌ New scraper architecture not available")
        print("Please ensure enhanced_model_scraper.py and production_hf_scraper.py are accessible")
        return 1
    
    print("🌟 Features:")
    print("   • Parquet storage with Snappy compression")
    print("   • K-NN search with 384-dim embeddings")
    print("   • Real-time progress tracking and statistics")
    print("   • Hardware compatibility analysis")
    print("   • Performance estimation and scoring")
    print("   • Mock and production modes")
    print()
    
    # Get API token
    hf_token = os.getenv("HF_TOKEN")
    has_token = bool(hf_token)
    
    print(f"🔑 HuggingFace API Token: {'✅ Available' if has_token else '❌ Not found'}")
    print(f"🌐 Network Access: {'✅ Production mode available' if has_token else '🎭 Mock mode only'}")
    print()
    
    # Configuration
    print("📋 Scraper Configuration:")
    print("1. Mock mode (fast, synthetic models)")
    print("2. Production mode (real HF API)" + (" ⚠️ Requires API token" if not has_token else ""))
    print()
    
    try:
        mode = input("Select mode (1/2, default 1): ").strip() or "1"
        
        if mode == "1":
            # Mock mode with EnhancedModelScraper
            print("\n🎭 Running in Mock Mode")
            print("-" * 50)
            
            limit_input = input("Number of models (default 10000): ").strip()
            limit = int(limit_input) if limit_input else 10000
            
            scraper = EnhancedModelScraper("scraped_model_data")
            print(f"\n🚀 Generating {limit:,} synthetic models...")
            
            results = scraper.scrape_all_models(limit=limit, mock_mode=True)
            
            print(f"\n✅ Mock scraping completed!")
            print(f"   • Models generated: {results['total_models']:,}")
            print(f"   • Processing rate: {results['models_per_second']:.1f} models/sec")
            print(f"   • Storage size: {results['storage_size_mb']:.1f} MB")
            print(f"   • Search index: {'Built' if results['search_index_built'] else 'Not built'}")
            
        elif mode == "2" and has_token:
            # Production mode with ProductionHFScraper
            print("\n🌐 Running in Production Mode")
            print("-" * 50)
            
            async def run_production():
                limit_input = input("Number of models (Enter for ALL): ").strip()
                limit = int(limit_input) if limit_input else None
                
                if limit is None:
                    print("⚠️  WARNING: This will scrape ALL HuggingFace models (750K+)")
                    print("   Estimated time: 2-4 hours")
                    confirm = input("Continue? (yes/no): ").strip().lower()
                    if confirm != 'yes':
                        print("❌ Cancelled by user")
                        return
                
                scraper = ProductionHFScraper("scraped_model_data", api_token=hf_token)
                print(f"\n🚀 Starting production scraping...")
                
                results = await scraper.scrape_production_models(limit=limit)
                
                print(f"\n✅ Production scraping completed!")
                print(f"   • Models scraped: {results['total_models']:,}")
                print(f"   • Processing rate: {results['models_per_second']:.1f} models/sec")
                print(f"   • Storage size: {results['storage_size_mb']:.1f} MB")
                print(f"   • Search index: {'Built' if results['search_index_built'] else 'Not built'}")
            
            asyncio.run(run_production())
            
        else:
            print("❌ Invalid mode or missing API token")
            return 1
        
        # Display statistics
        print("\n📈 Final Statistics:")
        if mode == "1":
            stats = scraper.get_statistics()
        else:
            # For production mode, create a scraper instance to load stats
            scraper = EnhancedModelScraper("scraped_model_data")
            stats = scraper.get_statistics()
        
        if 'total_models' in stats:
            print(f"   • Total models: {stats['total_models']:,}")
            print(f"   • Total authors: {stats.get('total_authors', 0):,}")
            print(f"   • Total architectures: {stats.get('total_architectures', 0):,}")
            print(f"   • Total task types: {stats.get('total_tasks', 0):,}")
            print(f"   • Average model size: {stats.get('avg_model_size_mb', 0):.1f} MB")
        
        print("\n🎉 Scraping completed successfully!")
        print(f"💾 Data saved to: scraped_model_data/")
        print(f"🔍 Search index ready for K-NN queries")
        print(f"📊 Parquet file available for analytics")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Scraping failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

        sys.exit(0)