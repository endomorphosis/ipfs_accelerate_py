#!/usr/bin/env python3
"""
HuggingFace Model Search Engine and Metadata Scraper

This module provides comprehensive functionality to:
1. Browse and search HuggingFace models
2. Scrape detailed metadata from the HuggingFace Hub
3. Add models to the local model manager with IPFS content addressing
4. Provide advanced search and filtering capabilities
"""

import os
import sys
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from urllib.parse import quote

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from ipfs_accelerate_py.model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        fetch_huggingface_repo_structure
    )
    HAVE_MODEL_MANAGER = True
except ImportError:
    try:
        from model_manager import (
            ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
            fetch_huggingface_repo_structure
        )
        HAVE_MODEL_MANAGER = True
    except ImportError:
        HAVE_MODEL_MANAGER = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HuggingFaceModelInfo:
    """Comprehensive information about a HuggingFace model."""
    model_id: str
    model_name: str
    author: str = ""
    description: str = ""
    tags: List[str] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    language: List[str] = None
    license: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    model_size: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    card_data: Optional[Dict[str, Any]] = None
    model_card: Optional[str] = None
    siblings: List[Dict[str, Any]] = None
    repository_structure: Optional[Dict[str, Any]] = None
    ipfs_cids: Optional[Dict[str, str]] = None
    private: bool = False
    gated: bool = False
    model_size_mb: Optional[float] = None
    architecture: Optional[str] = None
    framework: Optional[str] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.tags is None:
            self.tags = []
        if self.language is None:
            self.language = []
        if self.siblings is None:
            self.siblings = []

class HuggingFaceModelSearchEngine:
    """Advanced search engine for HuggingFace models with local caching."""
    
    def __init__(self, model_manager: Optional[ModelManager] = None, 
                 cache_dir: str = "./hf_model_cache"):
        """Initialize the search engine."""
        self.model_manager = model_manager
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.models_cache_file = self.cache_dir / "models_cache.json"
        self.search_index_file = self.cache_dir / "search_index.json"
        
        # HuggingFace API base URL
        self.hf_api_base = "https://huggingface.co/api"
        
        # Local caches
        self.models_cache: Dict[str, HuggingFaceModelInfo] = {}
        self.search_index: Dict[str, List[str]] = {}
        
        # Load existing caches
        self._load_caches()
        
    def _load_caches(self):
        """Load cached data from disk."""
        try:
            if self.models_cache_file.exists():
                with open(self.models_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.models_cache = {
                        k: HuggingFaceModelInfo(**v) 
                        for k, v in cache_data.items()
                    }
                logger.info(f"Loaded {len(self.models_cache)} models from cache")
            
            if self.search_index_file.exists():
                with open(self.search_index_file, 'r') as f:
                    self.search_index = json.load(f)
                logger.info(f"Loaded search index with {len(self.search_index)} terms")
        except Exception as e:
            logger.warning(f"Error loading caches: {e}")
    
    def _save_caches(self):
        """Save caches to disk."""
        try:
            # Save models cache
            cache_data = {k: asdict(v) for k, v in self.models_cache.items()}
            with open(self.models_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            # Save search index
            with open(self.search_index_file, 'w') as f:
                json.dump(self.search_index, f, indent=2)
            
            logger.info(f"Saved {len(self.models_cache)} models and search index to cache")
        except Exception as e:
            logger.error(f"Error saving caches: {e}")
    
    def search_huggingface_models(self, query: str = "", 
                                 limit: int = 100,
                                 filter_dict: Optional[Dict[str, Any]] = None,
                                 sort: str = "downloads") -> List[HuggingFaceModelInfo]:
        """
        Search HuggingFace models with advanced filtering.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filter_dict: Additional filters (task, library, language, etc.)
            sort: Sort by (downloads, likes, created_at, updated_at)
            
        Returns:
            List of HuggingFaceModelInfo objects
        """
        logger.info(f"Searching HuggingFace for: '{query}' (limit: {limit})")
        
        # Build search URL
        search_url = f"{self.hf_api_base}/models"
        params = {
            "limit": min(limit, 1000),  # API limit
            "sort": sort,
            "direction": -1,  # Descending
        }
        
        if query:
            params["search"] = query
        
        # Add filters
        if filter_dict:
            for key, value in filter_dict.items():
                if value:
                    params[f"filter_{key}"] = value
        
        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            models_data = response.json()
            results = []
            
            for model_data in models_data:
                try:
                    model_info = self._parse_model_data(model_data)
                    results.append(model_info)
                    
                    # Cache the model
                    self.models_cache[model_info.model_id] = model_info
                    
                    # Update search index
                    self._update_search_index(model_info)
                    
                except Exception as e:
                    logger.debug(f"Error parsing model {model_data.get('id', 'unknown')}: {e}")
                    continue
            
            # Save updated caches
            self._save_caches()
            
            logger.info(f"Found {len(results)} models")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching HuggingFace: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return []
    
    def _parse_model_data(self, model_data: Dict[str, Any]) -> HuggingFaceModelInfo:
        """Parse model data from HuggingFace API response."""
        
        # Extract basic information
        model_id = model_data.get("id", "")
        author = model_id.split("/")[0] if "/" in model_id else ""
        model_name = model_data.get("id", "").split("/")[-1]
        
        # Handle tags and metadata
        tags = model_data.get("tags", [])
        pipeline_tag = model_data.get("pipeline_tag")
        library_name = model_data.get("library_name")
        
        # Language handling
        languages = []
        for tag in tags:
            if tag.startswith("language:"):
                languages.append(tag.replace("language:", ""))
        
        # Parse card data if available
        card_data = model_data.get("cardData", {})
        
        return HuggingFaceModelInfo(
            model_id=model_id,
            model_name=model_name,
            author=author,
            description=model_data.get("description", ""),
            tags=tags,
            pipeline_tag=pipeline_tag,
            library_name=library_name,
            language=languages,
            license=card_data.get("license"),
            downloads=model_data.get("downloads", 0),
            likes=model_data.get("likes", 0),
            created_at=model_data.get("createdAt"),
            last_modified=model_data.get("lastModified"),
            model_size=None,  # Would need separate API call
            config=None,  # Would need separate API call
            card_data=card_data,
            siblings=model_data.get("siblings", [])
        )
    
    def _update_search_index(self, model_info: HuggingFaceModelInfo):
        """Update the search index with model information."""
        terms = []
        
        # Add model ID and name terms
        terms.extend(model_info.model_id.lower().split("/"))
        terms.extend(model_info.model_name.lower().split("-"))
        terms.extend(model_info.model_name.lower().split("_"))
        
        # Add author
        if model_info.author:
            terms.append(model_info.author.lower())
        
        # Add tags
        terms.extend([tag.lower() for tag in model_info.tags])
        
        # Add pipeline tag
        if model_info.pipeline_tag:
            terms.append(model_info.pipeline_tag.lower())
        
        # Add library name
        if model_info.library_name:
            terms.append(model_info.library_name.lower())
        
        # Add description words
        if model_info.description:
            desc_words = model_info.description.lower().split()
            terms.extend([word.strip(".,!?()[]{}") for word in desc_words])
        
        # Update index
        for term in set(terms):  # Remove duplicates
            if term not in self.search_index:
                self.search_index[term] = []
            if model_info.model_id not in self.search_index[term]:
                self.search_index[term].append(model_info.model_id)
    
    def get_detailed_model_info(self, model_id: str, 
                               include_repo_structure: bool = True) -> Optional[HuggingFaceModelInfo]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: HuggingFace model ID
            include_repo_structure: Whether to fetch repository structure and IPFS CIDs
            
        Returns:
            Detailed HuggingFaceModelInfo or None if not found
        """
        logger.info(f"Fetching detailed info for model: {model_id}")
        
        try:
            # Get model info from API
            model_url = f"{self.hf_api_base}/models/{quote(model_id)}"
            response = requests.get(model_url, timeout=30)
            response.raise_for_status()
            
            model_data = response.json()
            model_info = self._parse_model_data(model_data)
            
            # Get model configuration if available
            try:
                config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
                config_response = requests.get(config_url, timeout=10)
                if config_response.status_code == 200:
                    model_info.config = config_response.json()
            except:
                pass  # Config not available
            
            # Get repository structure and IPFS CIDs if requested
            if include_repo_structure:
                repo_structure = fetch_huggingface_repo_structure(
                    model_id, include_ipfs_cids=True
                )
                if repo_structure:
                    model_info.repository_structure = repo_structure
                    
                    # Extract IPFS CIDs
                    ipfs_cids = {}
                    for file_path, file_info in repo_structure.get("files", {}).items():
                        if "ipfs_cid" in file_info:
                            ipfs_cids[file_path] = file_info["ipfs_cid"]
                    
                    if ipfs_cids:
                        model_info.ipfs_cids = ipfs_cids
            
            # Cache the detailed model info
            self.models_cache[model_id] = model_info
            self._update_search_index(model_info)
            self._save_caches()
            
            return model_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching model {model_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching model {model_id}: {e}")
            return None
    
    def add_model_to_manager(self, model_info: HuggingFaceModelInfo) -> bool:
        """
        Add a HuggingFace model to the local model manager.
        
        Args:
            model_info: HuggingFaceModelInfo to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.model_manager or not HAVE_MODEL_MANAGER:
            logger.warning("Model manager not available")
            return False
        
        try:
            # Map HF pipeline tags to our model types
            model_type_mapping = {
                "text-generation": ModelType.LANGUAGE_MODEL,
                "text-classification": ModelType.LANGUAGE_MODEL,
                "token-classification": ModelType.LANGUAGE_MODEL,
                "question-answering": ModelType.LANGUAGE_MODEL,
                "fill-mask": ModelType.LANGUAGE_MODEL,
                "summarization": ModelType.ENCODER_DECODER,
                "translation": ModelType.ENCODER_DECODER,
                "text2text-generation": ModelType.ENCODER_DECODER,
                "image-classification": ModelType.VISION_MODEL,
                "object-detection": ModelType.VISION_MODEL,
                "image-segmentation": ModelType.VISION_MODEL,
                "text-to-image": ModelType.MULTIMODAL,
                "image-to-text": ModelType.MULTIMODAL,
                "automatic-speech-recognition": ModelType.AUDIO_MODEL,
                "audio-classification": ModelType.AUDIO_MODEL,
                "text-to-speech": ModelType.AUDIO_MODEL,
                "sentence-similarity": ModelType.EMBEDDING_MODEL,
                "feature-extraction": ModelType.EMBEDDING_MODEL,
            }
            
            model_type = model_type_mapping.get(
                model_info.pipeline_tag, ModelType.LANGUAGE_MODEL
            )
            
            # Create input/output specs based on pipeline tag
            inputs, outputs = self._create_io_specs(model_info.pipeline_tag, model_info.config)
            
            # Create ModelMetadata
            metadata = ModelMetadata(
                model_id=model_info.model_id,
                model_name=model_info.model_name,
                model_type=model_type,
                architecture=model_info.library_name or "unknown",
                inputs=inputs,
                outputs=outputs,
                huggingface_config=model_info.config,
                supported_backends=["transformers"] if model_info.library_name == "transformers" else [],
                tags=model_info.tags,
                source_url=f"https://huggingface.co/{model_info.model_id}",
                license=model_info.license,
                description=model_info.description,
                repository_structure=model_info.repository_structure
            )
            
            # Add to model manager
            self.model_manager.add_model(metadata)
            logger.info(f"Added model {model_info.model_id} to model manager")
            return True
            
        except Exception as e:
            logger.error(f"Error adding model {model_info.model_id} to manager: {e}")
            return False
    
    def _create_io_specs(self, pipeline_tag: Optional[str], 
                        config: Optional[Dict[str, Any]]) -> Tuple[List[IOSpec], List[IOSpec]]:
        """Create input/output specifications based on pipeline tag and config."""
        
        # Default specs
        inputs = [IOSpec(name="input_ids", data_type=DataType.TOKENS, shape=(-1,))]
        outputs = [IOSpec(name="logits", data_type=DataType.LOGITS, shape=(-1, -1))]
        
        if pipeline_tag == "text-generation":
            outputs = [IOSpec(name="generated_text", data_type=DataType.TEXT)]
        
        elif pipeline_tag == "text-classification":
            outputs = [IOSpec(name="scores", data_type=DataType.LOGITS, shape=(-1,))]
        
        elif pipeline_tag == "token-classification":
            outputs = [IOSpec(name="entities", data_type=DataType.LOGITS, shape=(-1, -1))]
        
        elif pipeline_tag == "question-answering":
            inputs = [
                IOSpec(name="question", data_type=DataType.TEXT),
                IOSpec(name="context", data_type=DataType.TEXT)
            ]
            outputs = [IOSpec(name="answer", data_type=DataType.TEXT)]
        
        elif pipeline_tag == "feature-extraction" or pipeline_tag == "sentence-similarity":
            inputs = [IOSpec(name="text", data_type=DataType.TEXT)]
            outputs = [IOSpec(name="embeddings", data_type=DataType.EMBEDDINGS, shape=(-1,))]
        
        elif pipeline_tag in ["image-classification", "object-detection", "image-segmentation"]:
            inputs = [IOSpec(name="image", data_type=DataType.IMAGE)]
            outputs = [IOSpec(name="predictions", data_type=DataType.LOGITS)]
        
        elif pipeline_tag == "text-to-image":
            inputs = [IOSpec(name="prompt", data_type=DataType.TEXT)]
            outputs = [IOSpec(name="image", data_type=DataType.IMAGE)]
        
        elif pipeline_tag == "image-to-text":
            inputs = [IOSpec(name="image", data_type=DataType.IMAGE)]
            outputs = [IOSpec(name="text", data_type=DataType.TEXT)]
        
        elif pipeline_tag in ["automatic-speech-recognition", "audio-classification"]:
            inputs = [IOSpec(name="audio", data_type=DataType.AUDIO)]
            outputs = [IOSpec(name="text" if "speech" in pipeline_tag else "classification", 
                            data_type=DataType.TEXT)]
        
        return inputs, outputs
    
    def local_search(self, query: str, max_results: int = 50) -> List[HuggingFaceModelInfo]:
        """
        Search locally cached models.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching HuggingFaceModelInfo objects
        """
        query_lower = query.lower()
        matching_model_ids = set()
        
        # Search in index
        for term in query_lower.split():
            if term in self.search_index:
                matching_model_ids.update(self.search_index[term])
        
        # Also search directly in model IDs and names
        for model_id, model_info in self.models_cache.items():
            if (query_lower in model_id.lower() or 
                query_lower in model_info.model_name.lower() or
                query_lower in model_info.description.lower()):
                matching_model_ids.add(model_id)
        
        # Get model info objects
        results = []
        for model_id in list(matching_model_ids)[:max_results]:
            if model_id in self.models_cache:
                results.append(self.models_cache[model_id])
        
        # Sort by downloads
        results.sort(key=lambda x: x.downloads, reverse=True)
        
        return results
    
    def get_popular_models(self, limit: int = 20, 
                          task_filter: Optional[str] = None) -> List[HuggingFaceModelInfo]:
        """
        Get popular models, optionally filtered by task.
        
        Args:
            limit: Number of models to return
            task_filter: Optional task/pipeline tag filter
            
        Returns:
            List of popular models
        """
        filter_dict = {}
        if task_filter:
            filter_dict["pipeline_tag"] = task_filter
        
        return self.search_huggingface_models(
            query="", limit=limit, filter_dict=filter_dict, sort="downloads"
        )
    
    def bulk_scrape_models(self, model_ids: List[str], 
                          add_to_manager: bool = True,
                          include_repo_structure: bool = True) -> Dict[str, bool]:
        """
        Scrape multiple models in bulk.
        
        Args:
            model_ids: List of HuggingFace model IDs to scrape
            add_to_manager: Whether to add to model manager
            include_repo_structure: Whether to fetch repository structure
            
        Returns:
            Dictionary mapping model_id to success status
        """
        results = {}
        
        for i, model_id in enumerate(model_ids):
            logger.info(f"Processing model {i+1}/{len(model_ids)}: {model_id}")
            
            try:
                # Get detailed model info
                model_info = self.get_detailed_model_info(
                    model_id, include_repo_structure=include_repo_structure
                )
                
                if model_info:
                    results[model_id] = True
                    
                    # Add to model manager if requested
                    if add_to_manager and self.model_manager:
                        self.add_model_to_manager(model_info)
                else:
                    results[model_id] = False
                
                # Rate limiting - be nice to HF API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {model_id}: {e}")
                results[model_id] = False
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the cached models."""
        if not self.models_cache:
            return {"total_models": 0}
        
        stats = {
            "total_models": len(self.models_cache),
            "total_downloads": sum(m.downloads for m in self.models_cache.values()),
            "total_likes": sum(m.likes for m in self.models_cache.values()),
            "unique_authors": len(set(m.author for m in self.models_cache.values() if m.author)),
            "unique_libraries": len(set(m.library_name for m in self.models_cache.values() if m.library_name)),
            "pipeline_tags": {},
            "languages": {},
            "licenses": {}
        }
        
        # Count pipeline tags
        for model in self.models_cache.values():
            if model.pipeline_tag:
                stats["pipeline_tags"][model.pipeline_tag] = stats["pipeline_tags"].get(model.pipeline_tag, 0) + 1
        
        # Count languages
        for model in self.models_cache.values():
            for lang in model.language:
                stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
        
        # Count licenses
        for model in self.models_cache.values():
            if model.license:
                stats["licenses"][model.license] = stats["licenses"].get(model.license, 0) + 1
        
        return stats

def main():
    """Demo of the HuggingFace model search engine."""
    print("🔍 HuggingFace Model Search Engine Demo")
    print("=" * 50)
    
    # Initialize model manager
    model_manager = None
    if HAVE_MODEL_MANAGER:
        try:
            model_manager = ModelManager("./hf_search_models.db")
            print("✅ Model manager initialized")
        except Exception as e:
            print(f"⚠️ Model manager failed: {e}")
    
    # Initialize search engine
    search_engine = HuggingFaceModelSearchEngine(model_manager)
    
    # Search for popular text generation models
    print("\n🔍 Searching for popular text generation models...")
    models = search_engine.search_huggingface_models(
        query="gpt", 
        limit=10,
        filter_dict={"pipeline_tag": "text-generation"},
        sort="downloads"
    )
    
    print(f"Found {len(models)} models:")
    for i, model in enumerate(models[:5], 1):
        print(f"{i}. {model.model_id} ({model.downloads:,} downloads)")
        print(f"   {model.description[:100]}...")
        
        # Get detailed info for first model
        if i == 1:
            print(f"\n📋 Getting detailed info for {model.model_id}...")
            detailed = search_engine.get_detailed_model_info(model.model_id)
            if detailed and detailed.repository_structure:
                file_count = detailed.repository_structure.get("total_files", 0)
                total_size = detailed.repository_structure.get("total_size", 0)
                print(f"   Repository: {file_count} files, {total_size:,} bytes")
                
                if detailed.ipfs_cids:
                    print(f"   IPFS CIDs: {len(detailed.ipfs_cids)} files with CIDs")
            
            # Add to model manager
            if model_manager:
                success = search_engine.add_model_to_manager(detailed)
                print(f"   Added to manager: {'✅' if success else '❌'}")
    
    # Show statistics
    print(f"\n📊 Cache Statistics:")
    stats = search_engine.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}: {len(value)} unique items")
        else:
            print(f"   {key}: {value:,}")

if __name__ == "__main__":
    main()