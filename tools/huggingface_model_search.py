#!/usr/bin/env python3
"""
HuggingFace Model Search Service

This module provides comprehensive search functionality for HuggingFace models
using both vector embeddings and BM25 keyword search, with advanced filtering
and sorting capabilities.

Features:
- Vector semantic search using sentence transformers
- BM25 keyword-based search
- Advanced filtering by task, library, language, etc.
- Sorting by downloads, likes, date, etc.
- Model metadata caching and indexing
- Real-time search with pagination
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import hashlib

# Import dependencies with fallbacks
try:
    from huggingface_hub import HfApi, ModelCard, ModelCardData
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
    HAVE_HF_HUB = True
except ImportError:
    HAVE_HF_HUB = False
    print("⚠️ HuggingFace Hub not available")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    print("⚠️ SentenceTransformers not available")

try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except ImportError:
    HAVE_BM25 = False
    print("⚠️ BM25 not available")

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False
    print("⚠️ Pandas not available")

# Configure logging
logger = logging.getLogger(__name__)

class HuggingFaceModelSearch:
    """
    Advanced HuggingFace model search with vector and BM25 search capabilities.
    """
    
    def __init__(self, cache_dir: str = "./hf_model_cache", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the HuggingFace model search service.
        
        Args:
            cache_dir: Directory for caching model metadata
            embedding_model: Model to use for generating embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.bm25_index = None
        
        # Cache files
        self.models_cache_file = self.cache_dir / "models_metadata.json"
        self.embeddings_cache_file = self.cache_dir / "model_embeddings.pkl"
        self.bm25_cache_file = self.cache_dir / "bm25_index.pkl"
        
        # Initialize APIs
        self.hf_api = HfApi() if HAVE_HF_HUB else None
        
        # Model metadata storage
        self.models_data: List[Dict] = []
        self.model_embeddings: Optional[np.ndarray] = None
        self.last_update: Optional[datetime] = None
        
        # Search configuration
        self.cache_expiry_hours = 24  # Cache expires after 24 hours
        self.max_models_per_request = 100
        self.default_sort = "downloads"
        
        logger.info(f"Initialized HuggingFace model search with cache dir: {cache_dir}")
    
    async def initialize(self) -> bool:
        """
        Initialize the search service by loading or building indices.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing HuggingFace model search service...")
            
            # Load cached data if available and not expired
            if self._is_cache_valid():
                logger.info("Loading from cache...")
                await self._load_from_cache()
            else:
                logger.info("Cache expired or missing, rebuilding indices...")
                await self._rebuild_indices()
            
            # Initialize embedding model only if we have data and dependencies
            if HAVE_SENTENCE_TRANSFORMERS and self.models_data and not self.embedding_model:
                try:
                    logger.info(f"Loading embedding model: {self.embedding_model_name}")
                    self.embedding_model = SentenceTransformer(self.embedding_model_name)
                except Exception as e:
                    logger.warning(f"Could not load embedding model: {e}")
                    # Continue without embedding model
            
            # If we still don't have data, ensure we have mock data
            if not self.models_data:
                logger.info("No data available, creating mock data...")
                self._create_mock_data()
            
            logger.info(f"Initialized with {len(self.models_data)} models")
            return len(self.models_data) > 0
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace search: {e}")
            # Fallback to mock data
            if not self.models_data:
                self._create_mock_data()
            return len(self.models_data) > 0
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not all([
            self.models_cache_file.exists(),
            self.embeddings_cache_file.exists() if HAVE_SENTENCE_TRANSFORMERS else True,
            self.bm25_cache_file.exists() if HAVE_BM25 else True
        ]):
            return False
        
        # Check if cache is within expiry time
        cache_time = datetime.fromtimestamp(self.models_cache_file.stat().st_mtime)
        return datetime.now() - cache_time < timedelta(hours=self.cache_expiry_hours)
    
    async def _load_from_cache(self):
        """Load model data and indices from cache."""
        try:
            # Load model metadata
            with open(self.models_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                self.models_data = cache_data.get('models', [])
                self.last_update = datetime.fromisoformat(cache_data.get('last_update', datetime.now().isoformat()))
            
            # Load embeddings
            if HAVE_SENTENCE_TRANSFORMERS and self.embeddings_cache_file.exists():
                with open(self.embeddings_cache_file, 'rb') as f:
                    self.model_embeddings = pickle.load(f)
            
            # Load BM25 index
            if HAVE_BM25 and self.bm25_cache_file.exists():
                with open(self.bm25_cache_file, 'rb') as f:
                    self.bm25_index = pickle.load(f)
            
            logger.info(f"Loaded {len(self.models_data)} models from cache")
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            # Fallback to rebuild
            await self._rebuild_indices()
    
    async def _rebuild_indices(self):
        """Rebuild all search indices from HuggingFace Hub."""
        if not HAVE_HF_HUB:
            logger.warning("HuggingFace Hub not available, using mock data")
            self._create_mock_data()
            return
        
        try:
            logger.info("Fetching models from HuggingFace Hub...")
            
            # Fetch popular models with metadata
            models = []
            for sort_method in ["downloads", "likes", "lastModified"]:
                try:
                    model_list = list(self.hf_api.list_models(
                        sort=sort_method,
                        limit=200,  # Get top 200 for each sort method
                        full=True
                    ))
                    models.extend(model_list)
                except Exception as e:
                    logger.warning(f"Failed to fetch models sorted by {sort_method}: {e}")
            
            # Remove duplicates and process models
            seen_ids = set()
            self.models_data = []
            
            for model in models:
                if model.modelId not in seen_ids:
                    seen_ids.add(model.modelId)
                    model_data = await self._process_model(model)
                    if model_data:
                        self.models_data.append(model_data)
            
            logger.info(f"Processed {len(self.models_data)} unique models")
            
            # Build search indices
            await self._build_vector_index()
            await self._build_bm25_index()
            
            # Save to cache
            await self._save_to_cache()
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to rebuild indices: {e}")
            self._create_mock_data()
    
    async def _process_model(self, model) -> Optional[Dict]:
        """Process a single model and extract metadata."""
        try:
            model_data = {
                "id": model.modelId,
                "author": getattr(model, 'author', '') or model.modelId.split('/')[0] if '/' in model.modelId else '',
                "name": model.modelId.split('/')[-1],
                "full_name": model.modelId,
                "downloads": getattr(model, 'downloads', 0) or 0,
                "likes": getattr(model, 'likes', 0) or 0,
                "created_at": getattr(model, 'createdAt', '').isoformat() if hasattr(getattr(model, 'createdAt', ''), 'isoformat') else str(getattr(model, 'createdAt', '')),
                "last_modified": getattr(model, 'lastModified', '').isoformat() if hasattr(getattr(model, 'lastModified', ''), 'isoformat') else str(getattr(model, 'lastModified', '')),
                "tags": getattr(model, 'tags', []) or [],
                "pipeline_tag": getattr(model, 'pipeline_tag', '') or '',
                "library_name": getattr(model, 'library_name', '') or '',
                "description": "",
                "datasets": getattr(model, 'datasets', []) or [],
                "languages": [],
                "metrics": getattr(model, 'metrics', []) or [],
                "card_data": {}
            }
            
            # Try to get additional metadata from model card
            try:
                card = ModelCard.load(model.modelId)
                if card and card.data:
                    model_data["description"] = getattr(card.data, 'description', '') or getattr(card, 'text', '')[:200] if hasattr(card, 'text') else ''
                    model_data["languages"] = getattr(card.data, 'language', []) or getattr(card.data, 'languages', []) or []
                    model_data["card_data"] = card.data.to_dict() if hasattr(card.data, 'to_dict') else {}
            except (RepositoryNotFoundError, RevisionNotFoundError, Exception):
                # Model card not available or accessible
                pass
            
            # Create searchable text for indexing
            searchable_text = f"{model_data['full_name']} {model_data['description']} {' '.join(model_data['tags'])} {model_data['pipeline_tag']} {model_data['library_name']}"
            model_data["searchable_text"] = searchable_text.lower()
            
            return model_data
            
        except Exception as e:
            logger.debug(f"Failed to process model {getattr(model, 'modelId', 'unknown')}: {e}")
            return None
    
    async def _build_vector_index(self):
        """Build vector embeddings for semantic search."""
        if not HAVE_SENTENCE_TRANSFORMERS or not self.models_data:
            return
        
        try:
            logger.info("Building vector embeddings...")
            
            if not self.embedding_model:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Create texts for embedding
            texts = [model['searchable_text'] for model in self.models_data]
            
            # Generate embeddings in batches
            self.model_embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True,
                batch_size=32
            )
            
            logger.info(f"Generated embeddings for {len(texts)} models")
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            self.model_embeddings = None
    
    async def _build_bm25_index(self):
        """Build BM25 index for keyword search."""
        if not HAVE_BM25 or not self.models_data:
            return
        
        try:
            logger.info("Building BM25 index...")
            
            # Tokenize texts for BM25
            tokenized_texts = [
                model['searchable_text'].split() 
                for model in self.models_data
            ]
            
            self.bm25_index = BM25Okapi(tokenized_texts)
            
            logger.info(f"Built BM25 index for {len(tokenized_texts)} models")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    async def _save_to_cache(self):
        """Save model data and indices to cache."""
        try:
            # Save model metadata
            cache_data = {
                "models": self.models_data,
                "last_update": datetime.now().isoformat(),
                "total_models": len(self.models_data)
            }
            
            with open(self.models_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if HAVE_SENTENCE_TRANSFORMERS and self.model_embeddings is not None:
                with open(self.embeddings_cache_file, 'wb') as f:
                    pickle.dump(self.model_embeddings, f)
            
            # Save BM25 index
            if HAVE_BM25 and self.bm25_index is not None:
                with open(self.bm25_cache_file, 'wb') as f:
                    pickle.dump(self.bm25_index, f)
            
            logger.info("Saved indices to cache")
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
    
    def _create_mock_data(self):
        """Create mock data when HuggingFace Hub is not available."""
        logger.info("Creating mock model data...")
        
        self.models_data = [
            {
                "id": "bert-base-uncased",
                "author": "google",
                "name": "bert-base-uncased",
                "full_name": "bert-base-uncased",
                "downloads": 1500000,
                "likes": 2500,
                "created_at": "2020-01-01T00:00:00",
                "last_modified": "2023-01-01T00:00:00",
                "tags": ["transformers", "pytorch", "tf", "bert", "fill-mask"],
                "pipeline_tag": "fill-mask",
                "library_name": "transformers",
                "description": "BERT base model (uncased) for masked language modeling",
                "datasets": ["bookcorpus", "wikipedia"],
                "languages": ["en"],
                "metrics": ["accuracy"],
                "card_data": {},
                "searchable_text": "bert-base-uncased bert base model uncased masked language modeling transformers pytorch tf bert fill-mask"
            },
            {
                "id": "gpt2",
                "author": "openai",
                "name": "gpt2",
                "full_name": "gpt2",
                "downloads": 2000000,
                "likes": 3000,
                "created_at": "2019-02-14T00:00:00",
                "last_modified": "2023-02-01T00:00:00",
                "tags": ["transformers", "pytorch", "tf", "gpt2", "text-generation"],
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "description": "GPT-2 model for text generation",
                "datasets": ["webtext"],
                "languages": ["en"],
                "metrics": ["perplexity"],
                "card_data": {},
                "searchable_text": "gpt2 gpt-2 model text generation transformers pytorch tf gpt2 text-generation"
            },
            {
                "id": "sentence-transformers/all-MiniLM-L6-v2",
                "author": "sentence-transformers",
                "name": "all-MiniLM-L6-v2",
                "full_name": "sentence-transformers/all-MiniLM-L6-v2",
                "downloads": 5000000,
                "likes": 1500,
                "created_at": "2021-08-01T00:00:00",
                "last_modified": "2023-03-01T00:00:00",
                "tags": ["sentence-transformers", "feature-extraction", "sentence-similarity"],
                "pipeline_tag": "sentence-similarity",
                "library_name": "sentence-transformers",
                "description": "Sentence embedding model for semantic similarity",
                "datasets": ["ms_marco", "nli"],
                "languages": ["en"],
                "metrics": ["spearman_cosine"],
                "card_data": {},
                "searchable_text": "sentence-transformers all-minilm-l6-v2 sentence embedding model semantic similarity sentence-transformers feature-extraction sentence-similarity"
            },
            {
                "id": "microsoft/DialoGPT-medium",
                "author": "microsoft",
                "name": "DialoGPT-medium",
                "full_name": "microsoft/DialoGPT-medium",
                "downloads": 750000,
                "likes": 1200,
                "created_at": "2020-05-01T00:00:00",
                "last_modified": "2023-01-15T00:00:00",
                "tags": ["transformers", "pytorch", "gpt2", "conversational"],
                "pipeline_tag": "conversational",
                "library_name": "transformers",
                "description": "DialoGPT conversational response generation model",
                "datasets": ["reddit"],
                "languages": ["en"],
                "metrics": ["bleu"],
                "card_data": {},
                "searchable_text": "microsoft dialogpt medium conversational response generation model transformers pytorch gpt2 conversational"
            },
            {
                "id": "facebook/bart-large-cnn",
                "author": "facebook",
                "name": "bart-large-cnn",
                "full_name": "facebook/bart-large-cnn",
                "downloads": 890000,
                "likes": 980,
                "created_at": "2020-10-01T00:00:00",
                "last_modified": "2023-02-20T00:00:00",
                "tags": ["transformers", "pytorch", "bart", "summarization"],
                "pipeline_tag": "summarization",
                "library_name": "transformers",
                "description": "BART model fine-tuned on CNN/DailyMail for summarization",
                "datasets": ["cnn_dailymail"],
                "languages": ["en"],
                "metrics": ["rouge"],
                "card_data": {},
                "searchable_text": "facebook bart large cnn summarization model transformers pytorch bart summarization"
            }
        ]
        
        # Set last update
        self.last_update = datetime.now()
        
        logger.info(f"Created {len(self.models_data)} mock models")
        
        # Build simple indices for mock data (no external dependencies)
        self._build_simple_indices()
    
    async def search_models(self, 
                          query: str = "",
                          search_type: str = "hybrid",  # "vector", "bm25", "hybrid"
                          filters: Optional[Dict] = None,
                          sort_by: str = "relevance",  # "relevance", "downloads", "likes", "date"
                          sort_order: str = "desc",  # "asc", "desc" 
                          offset: int = 0,
                          limit: int = 20) -> Dict[str, Any]:
        """
        Search for HuggingFace models with advanced filtering and sorting.
        
        Args:
            query: Search query string
            search_type: Type of search ("vector", "bm25", "hybrid")
            filters: Dictionary of filters to apply
            sort_by: Sort field
            sort_order: Sort order
            offset: Results offset for pagination
            limit: Maximum results to return
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            start_time = time.time()
            
            # Initialize if needed
            if not self.models_data:
                await self.initialize()
            
            # Apply filters first
            filtered_models = self._apply_filters(self.models_data, filters or {})
            
            # Perform search if query provided
            if query.strip():
                scored_results = await self._perform_search(query, search_type, filtered_models)
            else:
                # No query, just return filtered results with default scoring
                scored_results = [(i, 1.0, model) for i, model in enumerate(filtered_models)]
            
            # Sort results
            sorted_results = self._sort_results(scored_results, sort_by, sort_order)
            
            # Apply pagination
            total_results = len(sorted_results)
            paginated_results = sorted_results[offset:offset + limit]
            
            # Format results
            formatted_results = []
            for rank, (original_idx, score, model) in enumerate(paginated_results, 1):
                result = {
                    **model,
                    "search_score": round(score, 4),
                    "rank": rank + offset
                }
                # Remove searchable_text from results
                result.pop("searchable_text", None)
                formatted_results.append(result)
            
            search_time = time.time() - start_time
            
            return {
                "results": formatted_results,
                "total": total_results,
                "offset": offset,
                "limit": limit,
                "query": query,
                "search_type": search_type,
                "filters": filters or {},
                "sort_by": sort_by,
                "sort_order": sort_order,
                "search_time_ms": round(search_time * 1000, 2),
                "has_more": offset + limit < total_results
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "results": [],
                "total": 0,
                "offset": offset,
                "limit": limit,
                "error": str(e)
            }
    
    def _apply_filters(self, models: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to model list."""
        filtered = models.copy()
        
        # Filter by task/pipeline
        if "task" in filters and filters["task"]:
            task = filters["task"].lower()
            filtered = [m for m in filtered if task in m.get("pipeline_tag", "").lower()]
        
        # Filter by library
        if "library" in filters and filters["library"]:
            library = filters["library"].lower()
            filtered = [m for m in filtered if library in m.get("library_name", "").lower()]
        
        # Filter by language
        if "language" in filters and filters["language"]:
            language = filters["language"].lower()
            filtered = [m for m in filtered if any(language in lang.lower() for lang in m.get("languages", []))]
        
        # Filter by author
        if "author" in filters and filters["author"]:
            author = filters["author"].lower()
            filtered = [m for m in filtered if author in m.get("author", "").lower()]
        
        # Filter by minimum downloads
        if "min_downloads" in filters and filters["min_downloads"]:
            min_downloads = int(filters["min_downloads"])
            filtered = [m for m in filtered if m.get("downloads", 0) >= min_downloads]
        
        # Filter by minimum likes
        if "min_likes" in filters and filters["min_likes"]:
            min_likes = int(filters["min_likes"])
            filtered = [m for m in filtered if m.get("likes", 0) >= min_likes]
        
        # Filter by tags
        if "tags" in filters and filters["tags"]:
            if isinstance(filters["tags"], str):
                filter_tags = [filters["tags"].lower()]
            else:
                filter_tags = [tag.lower() for tag in filters["tags"]]
            
            filtered = [m for m in filtered if any(
                any(filter_tag in tag.lower() for tag in m.get("tags", []))
                for filter_tag in filter_tags
            )]
        
        return filtered
    
    async def _perform_search(self, query: str, search_type: str, models: List[Dict]) -> List[Tuple[int, float, Dict]]:
        """Perform the actual search and return scored results."""
        query_lower = query.lower()
        
        if search_type == "vector" and self.model_embeddings is not None:
            return await self._vector_search(query_lower, models)
        elif search_type == "bm25" and self.bm25_index is not None:
            return await self._bm25_search(query_lower, models)
        elif search_type == "hybrid":
            return await self._hybrid_search(query_lower, models)
        else:
            # Fallback to simple text matching
            return self._simple_text_search(query_lower, models)
    
    async def _vector_search(self, query: str, models: List[Dict]) -> List[Tuple[int, float, Dict]]:
        """Perform vector similarity search."""
        if not HAVE_SENTENCE_TRANSFORMERS or self.embedding_model is None:
            return self._simple_text_search(query, models)
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Get model indices that are in our filtered set
            model_ids = [model["id"] for model in models]
            all_model_ids = [model["id"] for model in self.models_data]
            
            # Find indices of filtered models in the original embeddings
            indices = [all_model_ids.index(mid) for mid in model_ids if mid in all_model_ids]
            
            if not indices:
                return []
            
            # Get embeddings for filtered models
            model_embeddings = self.model_embeddings[indices]
            
            # Compute similarities
            from numpy.linalg import norm
            similarities = np.dot(model_embeddings, query_embedding.T).flatten()
            similarities = similarities / (norm(model_embeddings, axis=1) * norm(query_embedding))
            
            # Create scored results
            results = []
            for i, similarity in enumerate(similarities):
                if similarity > 0.1:  # Threshold for relevance
                    model_idx = model_ids.index(all_model_ids[indices[i]])
                    results.append((model_idx, float(similarity), models[model_idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._simple_text_search(query, models)
    
    async def _bm25_search(self, query: str, models: List[Dict]) -> List[Tuple[int, float, Dict]]:
        """Perform BM25 keyword search."""
        if not HAVE_BM25 or self.bm25_index is None:
            return self._simple_text_search(query, models)
        
        try:
            # Tokenize query
            query_tokens = query.split()
            
            # Get model indices that are in our filtered set
            model_ids = [model["id"] for model in models]
            all_model_ids = [model["id"] for model in self.models_data]
            
            # Find indices of filtered models
            indices = [all_model_ids.index(mid) for mid in model_ids if mid in all_model_ids]
            
            if not indices:
                return []
            
            # Get BM25 scores for all models
            all_scores = self.bm25_index.get_scores(query_tokens)
            
            # Filter scores for our models
            results = []
            for i in indices:
                score = all_scores[i]
                if score > 0:
                    model_idx = model_ids.index(all_model_ids[i])
                    results.append((model_idx, float(score), models[model_idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return self._simple_text_search(query, models)
    
    async def _hybrid_search(self, query: str, models: List[Dict]) -> List[Tuple[int, float, Dict]]:
        """Perform hybrid search combining vector and BM25."""
        try:
            # Get results from both methods
            vector_results = await self._vector_search(query, models)
            bm25_results = await self._bm25_search(query, models)
            
            # Combine scores with weights
            vector_weight = 0.6
            bm25_weight = 0.4
            
            # Normalize scores to 0-1 range
            if vector_results:
                max_vector = max(result[1] for result in vector_results)
                vector_results = [(idx, score/max_vector, model) for idx, score, model in vector_results]
            
            if bm25_results:
                max_bm25 = max(result[1] for result in bm25_results)
                bm25_results = [(idx, score/max_bm25, model) for idx, score, model in bm25_results]
            
            # Combine results
            combined_scores = {}
            
            for idx, score, model in vector_results:
                combined_scores[idx] = (vector_weight * score, model)
            
            for idx, score, model in bm25_results:
                if idx in combined_scores:
                    existing_score, existing_model = combined_scores[idx]
                    combined_scores[idx] = (existing_score + bm25_weight * score, existing_model)
                else:
                    combined_scores[idx] = (bm25_weight * score, model)
            
            # Convert back to list format
            results = [(idx, score, model) for idx, (score, model) in combined_scores.items()]
            
            return results if results else self._simple_text_search(query, models)
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._simple_text_search(query, models)
    
    def _simple_text_search(self, query: str, models: List[Dict]) -> List[Tuple[int, float, Dict]]:
        """Fallback simple text matching search."""
        results = []
        query_words = query.lower().split()
        
        # Use direct text search (more reliable than index mapping)
        for i, model in enumerate(models):
            score = 0
            searchable = model.get("searchable_text", "").lower()
            
            # Exact phrase match (highest score)
            if query in searchable:
                score += 10
            
            # Word matches
            for word in query_words:
                if word in searchable:
                    score += 1
                    # Boost for matches in title
                    if word in model.get("full_name", "").lower():
                        score += 2
            
            if score > 0:
                # Normalize score
                normalized_score = min(score / (len(query_words) + 10), 1.0)
                results.append((i, normalized_score, model))
        
        return results
    
    def _sort_results(self, results: List[Tuple[int, float, Dict]], 
                     sort_by: str, sort_order: str) -> List[Tuple[int, float, Dict]]:
        """Sort search results."""
        reverse = sort_order.lower() == "desc"
        
        if sort_by == "relevance":
            # Already scored by relevance
            return sorted(results, key=lambda x: x[1], reverse=True)
        elif sort_by == "downloads":
            return sorted(results, key=lambda x: x[2].get("downloads", 0), reverse=reverse)
        elif sort_by == "likes":
            return sorted(results, key=lambda x: x[2].get("likes", 0), reverse=reverse)
        elif sort_by == "date":
            return sorted(results, key=lambda x: x[2].get("last_modified", ""), reverse=reverse)
        elif sort_by == "name":
            return sorted(results, key=lambda x: x[2].get("full_name", ""), reverse=reverse)
        else:
            # Default to relevance
            return sorted(results, key=lambda x: x[1], reverse=True)
    
    async def get_model_details(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a specific model."""
        try:
            # Find model in cache
            model = next((m for m in self.models_data if m["id"] == model_id), None)
            
            if not model:
                # Try to fetch from HuggingFace if not in cache
                if HAVE_HF_HUB and self.hf_api:
                    try:
                        hf_model = self.hf_api.model_info(model_id)
                        model = await self._process_model(hf_model)
                    except Exception as e:
                        logger.error(f"Failed to fetch model {model_id}: {e}")
                        return None
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to get model details for {model_id}: {e}")
            return None
    
    async def get_search_suggestions(self, query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query."""
        try:
            if not query or len(query) < 2:
                return []
            
            query_lower = query.lower()
            suggestions = set()
            
            # Look for matches in model names, authors, and tags
            for model in self.models_data:
                # Model name suggestions
                if query_lower in model.get("full_name", "").lower():
                    suggestions.add(model["full_name"])
                
                # Author suggestions
                if query_lower in model.get("author", "").lower():
                    suggestions.add(model["author"])
                
                # Tag suggestions
                for tag in model.get("tags", []):
                    if query_lower in tag.lower():
                        suggestions.add(tag)
                
                # Pipeline tag suggestions
                pipeline_tag = model.get("pipeline_tag", "")
                if query_lower in pipeline_tag.lower() and pipeline_tag:
                    suggestions.add(pipeline_tag)
            
            return sorted(list(suggestions))[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
    
    def _build_simple_indices(self):
        """Build simple search indices without external dependencies."""
        try:
            # Simple keyword-based index
            self.simple_search_index = {}
            for i, model in enumerate(self.models_data):
                words = model.get('searchable_text', '').lower().split()
                for word in words:
                    if word not in self.simple_search_index:
                        self.simple_search_index[word] = []
                    self.simple_search_index[word].append(i)
            
            logger.info("Built simple search index")
        except Exception as e:
            logger.error(f"Failed to build simple indices: {e}")

    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        try:
            if not self.models_data:
                return {"total_models": 0, "last_update": None}
            
            # Count by categories
            tasks = {}
            libraries = {}
            authors = {}
            
            for model in self.models_data:
                # Count tasks
                task = model.get("pipeline_tag", "unknown")
                tasks[task] = tasks.get(task, 0) + 1
                
                # Count libraries
                library = model.get("library_name", "unknown")
                libraries[library] = libraries.get(library, 0) + 1
                
                # Count authors
                author = model.get("author", "unknown")
                authors[author] = authors.get(author, 0) + 1
            
            return {
                "total_models": len(self.models_data),
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "has_vector_index": self.model_embeddings is not None,
                "has_bm25_index": self.bm25_index is not None,
                "top_tasks": dict(sorted(tasks.items(), key=lambda x: x[1], reverse=True)[:10]),
                "top_libraries": dict(sorted(libraries.items(), key=lambda x: x[1], reverse=True)[:10]),
                "top_authors": dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {"error": str(e)}

# Global instance
hf_search_service = None

async def get_hf_search_service() -> HuggingFaceModelSearch:
    """Get or create the global HuggingFace search service instance."""
    global hf_search_service
    
    if hf_search_service is None:
        hf_search_service = HuggingFaceModelSearch()
        success = await hf_search_service.initialize()
        if not success:
            logger.warning("HuggingFace search service initialization failed")
    
    return hf_search_service

if __name__ == "__main__":
    # Test the service
    async def test_search():
        service = HuggingFaceModelSearch()
        await service.initialize()
        
        # Test search
        results = await service.search_models("text generation", limit=5)
        print(f"Found {results['total']} models:")
        for model in results['results']:
            print(f"  - {model['full_name']} (downloads: {model['downloads']})")
        
        # Test suggestions
        suggestions = await service.get_search_suggestions("bert")
        print(f"\nSuggestions for 'bert': {suggestions}")
        
        # Test stats
        stats = service.get_search_stats()
        print(f"\nSearch stats: {stats}")
    
    asyncio.run(test_search())