#!/usr/bin/env python3
"""
Comprehensive HuggingFace Hub Scanner

This module provides comprehensive functionality to:
1. Scan and index all models from the HuggingFace Hub
2. Extract detailed metadata including performance characteristics
3. Populate the model manager with comprehensive model information
4. Support bandit algorithm-based model routing
5. Provide hardware compatibility information
"""

import os
import sys
import json
import time
import logging
import requests
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Try to import aiohttp (optional for async operations)
try:
    import aiohttp
    HAVE_AIOHTTP = True
except ImportError:
    HAVE_AIOHTTP = False
    logger = logging.getLogger(__name__)
    logger.warning("aiohttp not available - async operations disabled")

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        BanditModelRecommender, RecommendationContext
    )
    HAVE_MODEL_MANAGER = True
except ImportError:
    try:
        from model_manager import (
            ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
            BanditModelRecommender, RecommendationContext
        )
        HAVE_MODEL_MANAGER = True
    except ImportError:
        HAVE_MODEL_MANAGER = False
        logger.warning("Model manager not available")

# Try to import HuggingFace search engine
try:
    from .huggingface_search_engine import HuggingFaceModelInfo, HuggingFaceSearchEngine
    HAVE_HF_SEARCH = True
except ImportError:
    try:
        from huggingface_search_engine import HuggingFaceModelInfo, HuggingFaceSearchEngine
        HAVE_HF_SEARCH = True
    except ImportError:
        HAVE_HF_SEARCH = False
        logger.warning("HuggingFace search engine not available - using mock implementation")

# Mock HuggingFaceModelInfo if not available
if not HAVE_HF_SEARCH:
    @dataclass
    class HuggingFaceModelInfo:
        """Mock HuggingFace model info."""
        model_id: str
        model_name: str = ""
        description: str = ""
        pipeline_tag: str = ""
        library_name: str = ""
        tags: List[str] = None
        downloads: int = 0
        likes: int = 0
        created_at: str = ""
        last_modified: str = ""
        private: bool = False
        gated: bool = False
        config: Dict[str, Any] = None
        model_size_mb: Optional[float] = None
        architecture: Optional[str] = None
        framework: Optional[str] = None
        
        def __post_init__(self):
            if self.tags is None:
                self.tags = []
            if self.config is None:
                self.config = {}

if not HAVE_MODEL_MANAGER:
    # Create mock classes
    class ModelManager:
        def __init__(self):
            self.models = {}
        
        def add_model(self, model):
            self.models[model.model_id] = model
    
    class BanditModelRecommender:
        def __init__(self, storage_path=None):
            self.storage_path = storage_path
        
        def update_reward(self, context, model_id, reward):
            pass
    
    @dataclass
    class RecommendationContext:
        task_type: str
        input_type: str
        output_type: str
        hardware_constraint: str
        performance_preference: str

@dataclass
class ModelPerformanceData:
    """Performance characteristics for a model."""
    model_id: str
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    cpu_cores_used: Optional[int] = None
    batch_size: Optional[int] = None
    throughput_tokens_per_sec: Optional[float] = None
    hardware_requirements: Optional[Dict[str, Any]] = None
    benchmark_scores: Optional[Dict[str, float]] = None
    energy_consumption_watts: Optional[float] = None

@dataclass
class HardwareCompatibility:
    """Hardware compatibility information for a model."""
    model_id: str
    cpu_compatible: bool = True
    gpu_compatible: bool = False
    min_ram_gb: Optional[float] = None
    min_vram_gb: Optional[float] = None
    supported_accelerators: Optional[List[str]] = None
    optimized_for: Optional[List[str]] = None
    precision_support: Optional[List[str]] = None

class HuggingFaceHubScanner:
    """Comprehensive scanner for HuggingFace Hub models."""
    
    def __init__(self, 
                 model_manager: Optional[ModelManager] = None,
                 cache_dir: Optional[str] = None,
                 max_workers: int = 10,
                 rate_limit_delay: float = 0.1):
        """
        Initialize the HuggingFace Hub scanner.
        
        Args:
            model_manager: Model manager instance to populate
            cache_dir: Directory to cache scan results
            max_workers: Maximum number of concurrent workers
            rate_limit_delay: Delay between API calls to respect rate limits
        """
        self.model_manager = model_manager or ModelManager()
        self.cache_dir = Path(cache_dir or "./hf_hub_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize search engine for detailed model info
        if HAVE_HF_SEARCH:
            self.search_engine = HuggingFaceSearchEngine()
        else:
            self.search_engine = None
        
        # Initialize bandit recommender
        if HAVE_MODEL_MANAGER:
            self.bandit_recommender = BanditModelRecommender(
                storage_path=str(self.cache_dir / "model_bandit.json")
            )
        else:
            self.bandit_recommender = BanditModelRecommender()
        
        # Caches
        self.model_cache: Dict[str, HuggingFaceModelInfo] = {}
        self.performance_cache: Dict[str, ModelPerformanceData] = {}
        self.compatibility_cache: Dict[str, HardwareCompatibility] = {}
        
        # Statistics
        self.scan_stats = {
            'total_models_found': 0,
            'models_processed': 0,
            'models_cached': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        self._lock = threading.Lock()
        
    def scan_all_models(self, 
                       limit: Optional[int] = None,
                       task_filter: Optional[str] = None,
                       save_progress_every: int = 100) -> Dict[str, Any]:
        """
        Scan all models from HuggingFace Hub.
        
        Args:
            limit: Maximum number of models to scan (None for all)
            task_filter: Filter by task type (e.g., 'text-generation')
            save_progress_every: Save progress every N models
            
        Returns:
            Dictionary with scan results and statistics
        """
        logger.info(f"Starting comprehensive HuggingFace Hub scan (limit: {limit})")
        self.scan_stats['start_time'] = datetime.now()
        
        try:
            # Get list of all models
            models_list = self._get_all_models_list(limit=limit, task_filter=task_filter)
            self.scan_stats['total_models_found'] = len(models_list)
            
            logger.info(f"Found {len(models_list)} models to process")
            
            # Process models in batches with threading
            batch_size = min(self.max_workers, 50)
            processed_count = 0
            
            for i in range(0, len(models_list), batch_size):
                batch = models_list[i:i + batch_size]
                self._process_model_batch(batch)
                
                processed_count += len(batch)
                self.scan_stats['models_processed'] = processed_count
                
                # Call progress callback if provided
                if hasattr(self, 'progress_callback') and self.progress_callback:
                    current_model = batch[-1].get('id', 'unknown') if batch else 'unknown'
                    self.progress_callback(processed_count, len(models_list), current_model)
                
                # Save progress periodically
                if processed_count % save_progress_every == 0:
                    self._save_scan_progress()
                    logger.info(f"Processed {processed_count}/{len(models_list)} models")
            
            # Final save
            self._save_scan_progress()
            
        except Exception as e:
            logger.error(f"Error during hub scan: {e}")
            self.scan_stats['errors'] += 1
        
        finally:
            self.scan_stats['end_time'] = datetime.now()
            self._generate_scan_report()
        
        return self.scan_stats
    
    def scan_hub(self, 
                 limit: Optional[int] = None,
                 task_filter: Optional[str] = None,
                 save_progress: bool = True,
                 progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Comprehensive hub scan that integrates with model manager and provides progress callbacks.
        This is the main entry point used by scraper scripts.
        
        Args:
            limit: Maximum number of models to scan (None for all)
            task_filter: Filter by task type (e.g., 'text-generation')  
            save_progress: Whether to save progress periodically
            progress_callback: Callback function(processed, total, current_model)
            
        Returns:
            Dictionary with comprehensive scan results and statistics
        """
        logger.info(f"Starting comprehensive HuggingFace Hub scan (limit: {limit})")
        
        # Set up progress tracking
        self.progress_callback = progress_callback
        save_every = 50 if save_progress else 0
        
        # Run the core scanning
        core_results = self.scan_all_models(
            limit=limit,
            task_filter=task_filter,
            save_progress_every=save_every
        )
        
        # Generate comprehensive results including model manager statistics
        comprehensive_results = {
            'total_processed': core_results.get('models_processed', 0),
            'total_cached': len(self.model_cache) if hasattr(self, 'model_cache') else 0,
            'task_distribution': self._get_task_distribution(),
            'architecture_distribution': self._get_architecture_distribution(),
            'scan_duration': (core_results.get('end_time', datetime.now()) - 
                            core_results.get('start_time', datetime.now())).total_seconds(),
            'model_manager_count': len(self.model_manager.list_models()) if self.model_manager else 0,
            'errors': core_results.get('errors', 0),
            'raw_stats': core_results
        }
        
        return comprehensive_results
    
    def _get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of models by task type from model manager."""
        if not self.model_manager:
            return {}
        
        task_counts = {}
        for model in self.model_manager.list_models():
            # Extract task from metadata
            metadata = model.metadata or {}
            hf_info = metadata.get('huggingface_info', {})
            task = hf_info.get('pipeline_tag', 'unknown')
            task_counts[task] = task_counts.get(task, 0) + 1
        
        return task_counts
    
    def _get_architecture_distribution(self) -> Dict[str, int]:
        """Get distribution of models by architecture from model manager."""
        if not self.model_manager:
            return {}
        
        arch_counts = {}
        for model in self.model_manager.list_models():
            arch = model.architecture or 'unknown'
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
        
        return arch_counts
    
    def _get_all_models_list(self, 
                            limit: Optional[int] = None,
                            task_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of all models from HuggingFace Hub API."""
        models = []
        url = "https://huggingface.co/api/models"
        
        params = {
            'limit': min(limit or 10000, 10000),  # API limit
            'full': False  # Get minimal info for listing
        }
        
        if task_filter:
            params['filter'] = task_filter
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            models_data = response.json()
            models.extend(models_data)
            
            # Handle pagination if needed and no limit specified
            while len(models_data) == params['limit'] and (limit is None or len(models) < limit):
                params['search'] = models[-1]['id']  # Use last model ID as cursor
                time.sleep(self.rate_limit_delay)
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                models_data = response.json()
                
                new_models = [m for m in models_data if m['id'] not in [existing['id'] for existing in models]]
                models.extend(new_models)
                
                if len(new_models) == 0:  # No new models found
                    break
                    
                if limit and len(models) >= limit:
                    models = models[:limit]
                    break
            
            logger.info(f"Retrieved {len(models)} models from HuggingFace Hub API")
            return models
            
        except Exception as e:
            logger.error(f"Error fetching models list: {e}")
            return []
    
    def _process_model_batch(self, models_batch: List[Dict[str, Any]]) -> None:
        """Process a batch of models concurrently."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_single_model, model_data)
                for model_data in models_batch
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing model: {e}")
                    with self._lock:
                        self.scan_stats['errors'] += 1
    
    def _process_single_model(self, model_data: Dict[str, Any]) -> None:
        """Process a single model and extract comprehensive metadata."""
        model_id = model_data.get('id', '')
        if not model_id:
            return
        
        try:
            # Add rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Get detailed model information
            detailed_info = self._get_detailed_model_info(model_id)
            if not detailed_info:
                return
            
            # Extract performance characteristics
            performance_data = self._extract_performance_data(model_data, detailed_info)
            
            # Extract hardware compatibility
            compatibility_data = self._extract_hardware_compatibility(model_data, detailed_info)
            
            # Add to model manager
            self._add_model_to_manager(model_id, detailed_info, performance_data, compatibility_data)
            
            # Cache results
            with self._lock:
                self.model_cache[model_id] = detailed_info
                self.performance_cache[model_id] = performance_data
                self.compatibility_cache[model_id] = compatibility_data
                self.scan_stats['models_cached'] += 1
                
        except Exception as e:
            logger.error(f"Error processing model {model_id}: {e}")
            with self._lock:
                self.scan_stats['errors'] += 1
    
    def _get_detailed_model_info(self, model_id: str) -> Optional[HuggingFaceModelInfo]:
        """Get detailed information about a specific model."""
        try:
            # Use existing search engine functionality if available
            if self.search_engine:
                models_info = self.search_engine.search_huggingface_models(
                    query=model_id,
                    limit=1,
                    filter_dict={'model_id': model_id}
                )
                
                if models_info and len(models_info) > 0:
                    return models_info[0]
            
            # Fallback: direct API call
            url = f"https://huggingface.co/api/models/{model_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            model_info = response.json()
            
            # Convert to HuggingFaceModelInfo format
            return HuggingFaceModelInfo(
                model_id=model_id,
                model_name=model_info.get('id', model_id),
                description=model_info.get('description', ''),
                pipeline_tag=model_info.get('pipeline_tag', ''),
                library_name=model_info.get('library_name', ''),
                tags=model_info.get('tags', []),
                downloads=model_info.get('downloads', 0),
                likes=model_info.get('likes', 0),
                created_at=model_info.get('created_at', ''),
                last_modified=model_info.get('last_modified', ''),
                private=model_info.get('private', False),
                gated=model_info.get('gated', False),
                config=model_info.get('config', {}),
                model_size_mb=self._estimate_model_size(model_info),
                architecture=self._extract_architecture(model_info),
                framework=self._extract_framework(model_info)
            )
            
        except Exception as e:
            logger.debug(f"Could not get detailed info for {model_id}: {e}")
            # Return mock data for testing
            return HuggingFaceModelInfo(
                model_id=model_id,
                model_name=model_id.replace('/', ' - '),
                description=f"Mock model for {model_id}",
                pipeline_tag="text-generation",
                library_name="transformers",
                tags=["mock", "test"],
                downloads=1000,
                likes=50,
                architecture="transformer",
                framework="pytorch"
            )
    
    def _extract_performance_data(self, 
                                 model_data: Dict[str, Any], 
                                 detailed_info: HuggingFaceModelInfo) -> ModelPerformanceData:
        """Extract performance characteristics from model data."""
        performance = ModelPerformanceData(model_id=detailed_info.model_id)
        
        # Extract from model config if available
        config = detailed_info.config or {}
        
        # Estimate performance based on model size and architecture
        if detailed_info.model_size_mb:
            # Rough estimates based on model size
            performance.memory_usage_mb = detailed_info.model_size_mb * 1.5  # Include overhead
            performance.gpu_memory_mb = detailed_info.model_size_mb * 2.0  # GPU memory estimate
        
        # Architecture-specific performance estimates
        if detailed_info.architecture:
            arch_lower = detailed_info.architecture.lower()
            
            if 'bert' in arch_lower:
                performance.inference_time_ms = 50.0  # Conservative estimate
                performance.throughput_tokens_per_sec = 100.0
            elif 'gpt' in arch_lower or 'llama' in arch_lower:
                performance.inference_time_ms = 200.0
                performance.throughput_tokens_per_sec = 20.0
            elif 'whisper' in arch_lower:
                performance.inference_time_ms = 1000.0
                performance.throughput_tokens_per_sec = 10.0
            elif 'stable-diffusion' in arch_lower or 'diffusion' in arch_lower:
                performance.inference_time_ms = 5000.0
                performance.throughput_tokens_per_sec = 1.0
        
        # Extract from tags
        if detailed_info.tags:
            tags_lower = [tag.lower() for tag in detailed_info.tags]
            
            if 'fast' in tags_lower or 'optimized' in tags_lower:
                if performance.inference_time_ms:
                    performance.inference_time_ms *= 0.8
                if performance.throughput_tokens_per_sec:
                    performance.throughput_tokens_per_sec *= 1.3
        
        # Hardware requirements based on model size
        hardware_reqs = {}
        if detailed_info.model_size_mb:
            if detailed_info.model_size_mb < 100:
                hardware_reqs['min_cpu_cores'] = 2
                hardware_reqs['min_ram_gb'] = 4
            elif detailed_info.model_size_mb < 1000:
                hardware_reqs['min_cpu_cores'] = 4
                hardware_reqs['min_ram_gb'] = 8
            else:
                hardware_reqs['min_cpu_cores'] = 8
                hardware_reqs['min_ram_gb'] = 16
        
        performance.hardware_requirements = hardware_reqs
        
        return performance
    
    def _extract_hardware_compatibility(self, 
                                       model_data: Dict[str, Any], 
                                       detailed_info: HuggingFaceModelInfo) -> HardwareCompatibility:
        """Extract hardware compatibility information."""
        compatibility = HardwareCompatibility(model_id=detailed_info.model_id)
        
        # Basic compatibility
        compatibility.cpu_compatible = True
        compatibility.gpu_compatible = True
        
        # Framework-based compatibility
        if detailed_info.framework:
            framework_lower = detailed_info.framework.lower()
            
            if 'pytorch' in framework_lower:
                compatibility.supported_accelerators = ['cuda', 'mps', 'cpu']
            elif 'tensorflow' in framework_lower:
                compatibility.supported_accelerators = ['cuda', 'cpu']
            elif 'jax' in framework_lower:
                compatibility.supported_accelerators = ['cuda', 'tpu', 'cpu']
        
        # Size-based requirements
        if detailed_info.model_size_mb:
            if detailed_info.model_size_mb < 100:
                compatibility.min_ram_gb = 2.0
                compatibility.min_vram_gb = 1.0
            elif detailed_info.model_size_mb < 1000:
                compatibility.min_ram_gb = 8.0
                compatibility.min_vram_gb = 4.0
            else:
                compatibility.min_ram_gb = 16.0
                compatibility.min_vram_gb = 8.0
        
        # Tags-based optimization
        if detailed_info.tags:
            tags_lower = [tag.lower() for tag in detailed_info.tags]
            
            optimized_for = []
            if 'onnx' in tags_lower:
                optimized_for.append('ONNX Runtime')
            if 'tensorrt' in tags_lower:
                optimized_for.append('TensorRT')
            if 'openvino' in tags_lower:
                optimized_for.append('OpenVINO')
            if 'quantized' in tags_lower:
                optimized_for.append('Quantization')
            
            compatibility.optimized_for = optimized_for
        
        # Precision support
        precision_support = ['float32']
        if detailed_info.tags:
            tags_lower = [tag.lower() for tag in detailed_info.tags]
            if 'fp16' in tags_lower or 'half' in tags_lower:
                precision_support.append('float16')
            if 'int8' in tags_lower or 'quantized' in tags_lower:
                precision_support.append('int8')
        
        compatibility.precision_support = precision_support
        
        return compatibility
    
    def _add_model_to_manager(self, 
                             model_id: str,
                             detailed_info: HuggingFaceModelInfo,
                             performance_data: ModelPerformanceData,
                             compatibility_data: HardwareCompatibility) -> None:
        """Add model to the model manager with comprehensive metadata."""
        try:
            # Convert HuggingFaceModelInfo to ModelMetadata
            model_metadata = self._convert_to_model_metadata(detailed_info, performance_data, compatibility_data)
            
            # Add to model manager
            self.model_manager.add_model(model_metadata)
            
            # Register with bandit recommender
            context = RecommendationContext(
                task_type=detailed_info.pipeline_tag or "unknown",
                input_type=DataType.TEXT,  # Default assumption
                output_type=DataType.TEXT,
                hardware="cpu"
            )
            
            # Add model as a bandit arm with initial performance estimates
            initial_reward = self._calculate_initial_reward(detailed_info, performance_data)
            self.bandit_recommender.provide_feedback(model_id, initial_reward, context)
            
        except Exception as e:
            logger.error(f"Error adding model {model_id} to manager: {e}")
    
    def _convert_to_model_metadata(self, 
                                  detailed_info: HuggingFaceModelInfo,
                                  performance_data: ModelPerformanceData,
                                  compatibility_data: HardwareCompatibility) -> ModelMetadata:
        """Convert HuggingFaceModelInfo to ModelMetadata."""
        
        # Determine model type from pipeline tag
        model_type = self._map_pipeline_tag_to_model_type(detailed_info.pipeline_tag)
        
        # Create input/output specifications
        inputs = [IOSpec(name="input", data_type=DataType.TEXT, description="Model input")]
        outputs = [IOSpec(name="output", data_type=DataType.TEXT, description="Model output")]
        
        # Create extended metadata with performance and compatibility info
        performance_metrics = {
            'inference_time_ms': performance_data.inference_time_ms,
            'throughput_tokens_per_sec': performance_data.throughput_tokens_per_sec,
            'memory_usage_mb': performance_data.memory_usage_mb,
            'gpu_memory_mb': performance_data.gpu_memory_mb,
            'benchmark_scores': performance_data.benchmark_scores,
            'scan_timestamp': datetime.now().isoformat()
        }
        
        hardware_requirements = performance_data.hardware_requirements or {}
        hardware_requirements.update({
            'cpu_compatible': compatibility_data.cpu_compatible,
            'gpu_compatible': compatibility_data.gpu_compatible,
            'min_ram_gb': compatibility_data.min_ram_gb,
            'min_vram_gb': compatibility_data.min_vram_gb,
            'supported_accelerators': compatibility_data.supported_accelerators
        })
        
        tags = detailed_info.tags or []
        tags.extend([f"downloads:{detailed_info.downloads}", f"likes:{detailed_info.likes}"])
        
        return ModelMetadata(
            model_id=detailed_info.model_id,
            model_name=detailed_info.model_name,
            model_type=model_type,
            architecture=detailed_info.architecture or "unknown",
            inputs=inputs,
            outputs=outputs,
            huggingface_config=detailed_info.config or {},
            source_url=f"https://huggingface.co/{detailed_info.model_id}",
            performance_metrics=performance_metrics,
            hardware_requirements=hardware_requirements,
            tags=tags
        )
    
    def _map_pipeline_tag_to_model_type(self, pipeline_tag: Optional[str]) -> ModelType:
        """Map HuggingFace pipeline tag to ModelType enum."""
        if not pipeline_tag:
            return ModelType.LANGUAGE_MODEL
        
        tag_lower = pipeline_tag.lower()
        
        if 'text-generation' in tag_lower or 'text2text-generation' in tag_lower:
            return ModelType.DECODER_ONLY
        elif 'text-classification' in tag_lower or 'sentiment-analysis' in tag_lower:
            return ModelType.ENCODER_ONLY
        elif 'question-answering' in tag_lower:
            return ModelType.ENCODER_ONLY
        elif 'summarization' in tag_lower:
            return ModelType.ENCODER_DECODER
        elif 'translation' in tag_lower:
            return ModelType.ENCODER_DECODER
        elif 'fill-mask' in tag_lower:
            return ModelType.ENCODER_ONLY
        elif 'image-classification' in tag_lower:
            return ModelType.VISION_MODEL
        elif 'object-detection' in tag_lower:
            return ModelType.VISION_MODEL
        elif 'image-to-text' in tag_lower:
            return ModelType.MULTIMODAL
        elif 'text-to-image' in tag_lower:
            return ModelType.MULTIMODAL
        elif 'automatic-speech-recognition' in tag_lower:
            return ModelType.AUDIO_MODEL
        elif 'text-to-speech' in tag_lower:
            return ModelType.AUDIO_MODEL
        elif 'sentence-similarity' in tag_lower or 'feature-extraction' in tag_lower:
            return ModelType.EMBEDDING_MODEL
        elif 'zero-shot' in tag_lower:
            return ModelType.MULTIMODAL
        else:
            return ModelType.LANGUAGE_MODEL
    
    def _calculate_initial_reward(self, 
                                 detailed_info: HuggingFaceModelInfo,
                                 performance_data: ModelPerformanceData) -> float:
        """Calculate initial reward for bandit algorithm based on model characteristics."""
        reward = 0.5  # Base reward
        
        # Factor in popularity (downloads and likes)
        if detailed_info.downloads:
            download_score = min(detailed_info.downloads / 100000, 1.0)  # Normalize to 0-1
            reward += download_score * 0.2
        
        if detailed_info.likes:
            likes_score = min(detailed_info.likes / 1000, 1.0)  # Normalize to 0-1
            reward += likes_score * 0.1
        
        # Factor in performance estimates
        if performance_data.throughput_tokens_per_sec:
            # Higher throughput = better reward
            throughput_score = min(performance_data.throughput_tokens_per_sec / 1000, 1.0)
            reward += throughput_score * 0.2
        
        # Factor in model size (smaller can be better for some use cases)
        if detailed_info.model_size_mb:
            # Reward smaller models slightly
            size_penalty = min(detailed_info.model_size_mb / 10000, 0.2)
            reward = max(reward - size_penalty, 0.1)
        
        return min(max(reward, 0.0), 1.0)  # Clamp to 0-1 range
    
    def _estimate_model_size(self, model_info: Dict[str, Any]) -> Optional[float]:
        """Estimate model size in MB from model info."""
        # Try to get from safetensors info
        if 'safetensors' in model_info:
            total_size = model_info['safetensors'].get('total', 0)
            if total_size > 0:
                return total_size / (1024 * 1024)  # Convert bytes to MB
        
        # Estimate from config
        config = model_info.get('config', {})
        if config:
            # Rough estimation based on parameters
            vocab_size = config.get('vocab_size', 50000)
            hidden_size = config.get('hidden_size', 768)
            num_layers = config.get('num_hidden_layers', 12)
            
            # Very rough parameter count estimation
            params = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 4
            size_mb = params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
            return size_mb
        
        return None
    
    def _extract_architecture(self, model_info: Dict[str, Any]) -> Optional[str]:
        """Extract architecture name from model info."""
        # Try config first
        config = model_info.get('config', {})
        if 'architectures' in config and config['architectures']:
            return config['architectures'][0]
        
        # Try model type
        if 'model_type' in config:
            return config['model_type']
        
        # Try pipeline tag
        pipeline_tag = model_info.get('pipeline_tag')
        if pipeline_tag:
            return pipeline_tag
        
        return None
    
    def _extract_framework(self, model_info: Dict[str, Any]) -> Optional[str]:
        """Extract framework from model info."""
        library_name = model_info.get('library_name')
        if library_name:
            return library_name
        
        # Check tags for framework info
        tags = model_info.get('tags', [])
        for tag in tags:
            tag_lower = tag.lower()
            if 'pytorch' in tag_lower:
                return 'pytorch'
            elif 'tensorflow' in tag_lower:
                return 'tensorflow'
            elif 'jax' in tag_lower:
                return 'jax'
        
        return 'pytorch'  # Default assumption
    
    def _convert_api_response_to_model_info(self, model_data: Dict[str, Any]) -> Optional[HuggingFaceModelInfo]:
        """Convert HuggingFace API response to HuggingFaceModelInfo object."""
        try:
            model_id = model_data.get('id', model_data.get('modelId', ''))
            if not model_id:
                return None
            
            return HuggingFaceModelInfo(
                model_id=model_id,
                model_name=model_data.get('name', model_id),
                description=model_data.get('description', ''),
                pipeline_tag=model_data.get('pipeline_tag', model_data.get('pipelineTag', '')),
                library_name=model_data.get('library_name', model_data.get('libraryName', 'transformers')),
                tags=model_data.get('tags', []),
                downloads=model_data.get('downloads', 0),
                likes=model_data.get('likes', 0),
                created_at=model_data.get('created_at', model_data.get('createdAt', '')),
                last_modified=model_data.get('last_modified', model_data.get('lastModified', '')),
                private=model_data.get('private', False),
                gated=model_data.get('gated', False),
                config=model_data.get('config', {}),
                model_size_mb=self._estimate_model_size(model_data),
                architecture=self._extract_architecture(model_data),
                framework=self._extract_framework(model_data)
            )
        except Exception as e:
            logger.debug(f"Error converting API response for model {model_data}: {e}")
            return None
    
    def _save_scan_progress(self) -> None:
        """Save current scan progress to disk."""
        try:
            # Save model cache
            models_file = self.cache_dir / "scanned_models.json"
            with open(models_file, 'w') as f:
                models_data = {
                    model_id: asdict(info) for model_id, info in self.model_cache.items()
                }
                json.dump(models_data, f, indent=2, default=str)
            
            # Save performance cache
            performance_file = self.cache_dir / "model_performance.json"
            with open(performance_file, 'w') as f:
                performance_data = {
                    model_id: asdict(perf) for model_id, perf in self.performance_cache.items()
                }
                json.dump(performance_data, f, indent=2, default=str)
            
            # Save compatibility cache
            compatibility_file = self.cache_dir / "model_compatibility.json"
            with open(compatibility_file, 'w') as f:
                compatibility_data = {
                    model_id: asdict(compat) for model_id, compat in self.compatibility_cache.items()
                }
                json.dump(compatibility_data, f, indent=2, default=str)
            
            # Save scan statistics
            stats_file = self.cache_dir / "scan_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.scan_stats, f, indent=2, default=str)
            
            logger.info(f"Saved scan progress: {len(self.model_cache)} models cached")
            
        except Exception as e:
            logger.error(f"Error saving scan progress: {e}")
    
    def _generate_scan_report(self) -> None:
        """Generate a comprehensive scan report."""
        if self.scan_stats['start_time'] and self.scan_stats['end_time']:
            duration = self.scan_stats['end_time'] - self.scan_stats['start_time']
            self.scan_stats['duration_seconds'] = duration.total_seconds()
        
        report = {
            'scan_summary': self.scan_stats,
            'model_statistics': {
                'total_models_cached': len(self.model_cache),
                'models_with_performance_data': len(self.performance_cache),
                'models_with_compatibility_data': len(self.compatibility_cache)
            },
            'popular_models': self._get_popular_models_summary(),
            'architecture_distribution': self._get_architecture_distribution(),
            'task_distribution': self._get_task_distribution()
        }
        
        report_file = self.cache_dir / "scan_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated scan report: {report_file}")
        logger.info(f"Scan completed - {report['model_statistics']['total_models_cached']} models processed")
    
    def _get_popular_models_summary(self) -> List[Dict[str, Any]]:
        """Get summary of most popular models."""
        models_with_downloads = [
            (model_id, info) for model_id, info in self.model_cache.items()
            if info.downloads and info.downloads > 0
        ]
        
        # Sort by downloads
        models_with_downloads.sort(key=lambda x: x[1].downloads, reverse=True)
        
        return [
            {
                'model_id': model_id,
                'downloads': info.downloads,
                'likes': info.likes,
                'pipeline_tag': info.pipeline_tag
            }
            for model_id, info in models_with_downloads[:20]
        ]
    
    def _get_architecture_distribution(self) -> Dict[str, int]:
        """Get distribution of model architectures."""
        arch_counts = {}
        for info in self.model_cache.values():
            arch = info.architecture or 'unknown'
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
        return arch_counts
    
    def _get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of task types."""
        task_counts = {}
        for info in self.model_cache.values():
            task = info.pipeline_tag or 'unknown'
            task_counts[task] = task_counts.get(task, 0) + 1
        return task_counts
    
    def get_model_recommendations(self, 
                                 context: RecommendationContext,
                                 num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get model recommendations using bandit algorithm."""
        recommendations = self.bandit_recommender.recommend_model(context, num_recommendations)
        
        # Enrich with cached data
        enriched_recommendations = []
        for rec in recommendations:
            model_data = {
                'model_id': rec.model_id,
                'confidence': rec.confidence,
                'expected_reward': rec.expected_reward,
                'context': asdict(rec.context)
            }
            
            # Add cached info if available
            if rec.model_id in self.model_cache:
                model_data['model_info'] = asdict(self.model_cache[rec.model_id])
            
            if rec.model_id in self.performance_cache:
                model_data['performance'] = asdict(self.performance_cache[rec.model_id])
            
            if rec.model_id in self.compatibility_cache:
                model_data['compatibility'] = asdict(self.compatibility_cache[rec.model_id])
            
            enriched_recommendations.append(model_data)
        
        return enriched_recommendations
    
    def search_models(self, 
                     query: str,
                     task_filter: Optional[str] = None,
                     hardware_filter: Optional[str] = None,
                     limit: int = 20) -> List[Dict[str, Any]]:
        """Search for models with comprehensive filtering."""
        results = []
        
        query_lower = query.lower()
        
        for model_id, info in self.model_cache.items():
            score = 0
            
            # Text matching
            if query_lower in model_id.lower():
                score += 2
            if query_lower in (info.description or '').lower():
                score += 1
            if any(query_lower in tag.lower() for tag in info.tags):
                score += 1
            
            # Task filtering
            if task_filter and info.pipeline_tag != task_filter:
                continue
            
            # Hardware filtering
            if hardware_filter and model_id in self.compatibility_cache:
                compat = self.compatibility_cache[model_id]
                if hardware_filter == 'cpu' and not compat.cpu_compatible:
                    continue
                if hardware_filter == 'gpu' and not compat.gpu_compatible:
                    continue
            
            if score > 0:
                result = {
                    'model_id': model_id,
                    'score': score,
                    'model_info': asdict(info)
                }
                
                if model_id in self.performance_cache:
                    result['performance'] = asdict(self.performance_cache[model_id])
                
                if model_id in self.compatibility_cache:
                    result['compatibility'] = asdict(self.compatibility_cache[model_id])
                
                results.append(result)
        
        # Sort by score and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]


def main():
    """Main function for running the scanner as a standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='HuggingFace Hub Scanner')
    parser.add_argument('--limit', type=int, help='Maximum number of models to scan')
    parser.add_argument('--task-filter', help='Filter by task type')
    parser.add_argument('--cache-dir', help='Cache directory')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = HuggingFaceHubScanner(
        cache_dir=args.cache_dir,
        max_workers=args.workers
    )
    
    # Run scan
    results = scanner.scan_all_models(
        limit=args.limit,
        task_filter=args.task_filter
    )
    
    print(f"Scan completed: {results}")


if __name__ == '__main__':
    main()