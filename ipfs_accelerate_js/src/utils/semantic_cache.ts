/**
 * Converted from Python: semantic_cache.py
 * Conversion date: 2025-03-11 04:08:55
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */


export interface Props {
  cache: Dict;
  ttl: continue;
  lock: current_time;
  max_cache_size: return;
  cache: self;
  cache: key;
  lock: self;
  lock: current_time;
  stats_lock: self;
  stats_lock: self;
  stats_lock: self;
  stats_lock: self;
  stats_lock: stats_copy;
}

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1 as np
import * as $1
import ${$1} from "$1"
import ${$1} from "$1"
import * as $1
import * as $1
import * as $1
import * as $1

# Add parent directory to path
sys.path.insert()))))))))0, os.path.dirname()))))))))os.path.dirname()))))))))os.path.abspath()))))))))__file__))))
sys.path.insert()))))))))0, os.path.dirname()))))))))os.path.dirname()))))))))os.path.dirname()))))))))os.path.abspath()))))))))__file__)))))

logging.basicConfig()))))))))level=logging.INFO)
logger = logging.getLogger()))))))))__name__)

class $1 extends $2 {
  """
  Cache implementation that uses semantic similarity between queries to determine cache hits.
  
}
  Instead of exact string matching, this cache computes embeddings for queries && uses
  cosine similarity to find semantically similar cached entries.
  """
  
  def __init__()))))))))
  self,
  embedding_model: Optional[]],,Any] = null,
  $1: number = 0.85,
  $1: number = 1000,
  $1: number = 3600,  # Time-to-live in seconds
  $1: boolean = true
  ):
    """
    Initialize the semantic cache.
    
    Args:
      embedding_model: Model used to generate embeddings for queries
      similarity_threshold: Minimum cosine similarity score to consider a cache hit
      max_cache_size: Maximum number of entries in the cache
      ttl: Time-to-live for cache entries in seconds
      use_lru: Whether to use LRU eviction policy
      """
      this.embedding_model = embedding_model
      this.similarity_threshold = similarity_threshold
      this.max_cache_size = max_cache_size
      this.ttl = ttl
      this.use_lru = use_lru
    
    # Main cache storage: {}}}cache_key: ()))))))))embedding, response, timestamp, metadata)}
      this.cache: Dict[]],,str, Tuple[]],,torch.Tensor, Any, float, Dict]] = OrderedDict())))))))))
      ,
    # Lock for thread safety
      this.lock = threading.RLock())))))))))
    
      logger.info()))))))))`$1`)
  
  def _generate_embedding()))))))))self, $1: string) -> torch.Tensor:
    """
    Generate an embedding for the given query.
    
    Args:
      query: Input query text
      
    Returns:
      Embedding vector for the query
      """
    if ($1) {
      # Fallback to hash-based representation if no embedding model
      hash_val = int()))))))))hashlib.md5()))))))))query.encode())))))))))).hexdigest()))))))))), 16)
      # Create a pseudo-embedding from the hash
      pseudo_embedding = torch.tensor())))))))):
        $3.map(($2) => $1),:,
        dtype=torch.float32
        )
      # Normalize the pseudo-embedding
      return pseudo_embedding / torch.norm()))))))))pseudo_embedding, p=2)
    
    }
    # Use the actual embedding model
    with torch.no_grad()))))))))):
      embedding = this.embedding_model.encode()))))))))query)
      if ($1) {
        embedding = torch.tensor()))))))))embedding, dtype=torch.float32)
      # Normalize the embedding
      }
      return embedding / torch.norm()))))))))embedding, p=2)
  
  $1($2): $3 {
    """
    Compute cosine similarity between two embeddings.
    
  }
    Args:
      emb1: First embedding
      emb2: Second embedding
      
    Returns:
      Cosine similarity score between the embeddings
      """
      return torch.nn.functional.cosine_similarity()))))))))
      emb1.unsqueeze()))))))))0), emb2.unsqueeze()))))))))0)
      ).item())))))))))
  
      def _find_most_similar()))))))))self, query_embedding: torch.Tensor) -> Tuple[]],,Optional[]],,str], float]:,
      """
      Find the most similar cached entry {:::: to the given query embedding.
    
    Args:
      query_embedding: Query embedding to compare against cache
      
    Returns:
      Tuple of ()))))))))cache_key, similarity_score) for the most similar entry {::::
        """
        most_similar_key = null
        highest_similarity = -1.0
    
    for key, ()))))))))cached_embedding, _, timestamp, _) in this.Object.entries($1)))))))))):
      # Skip expired entries
      if ($1) {
      continue
      }
        
      similarity = this._compute_similarity()))))))))query_embedding, cached_embedding)
      
      if ($1) {
        highest_similarity = similarity
        most_similar_key = key
    
      }
      return most_similar_key, highest_similarity
  
  $1($2): $3 {
    """Remove expired entries from the cache."""
    with this.lock:
      current_time = time.time())))))))))
      keys_to_remove = []],,
      key for key, ()))))))))_, _, timestamp, _) in this.Object.entries($1))))))))))
      if current_time - timestamp > this.ttl
      ]
      :
      for (const $1 of $2) {
        del this.cache[]],,key]
        
      }
        logger.debug()))))))))`$1`)
  
  }
  $1($2): $3 {
    """Remove entries if ($1) {
    if ($1) {
      return
      
    }
    with this.lock:
    }
      # If using LRU, the first item in OrderedDict is the least recently used
      if ($1) {
        this.cache.popitem()))))))))last=false)
        logger.debug()))))))))"Removed least recently used cache entry ${$1} else {
        # Otherwise remove random entry {::::
        }
        if ($1) {
          key = next()))))))))iter()))))))))this.Object.keys($1))))))))))))
          del this.cache[]],,key]
          logger.debug()))))))))"Removed random cache entry {::::")
  
        }
  def get()))))))))self, $1: string, metadata: Optional[]],,Dict] = null) -> Tuple[]],,Optional[]],,Any], float, Optional[]],,Dict]]:
      }
    """
    Get a cached response for a query if a similar one exists.
    :
    Args:
      query: Query to look up in the cache
      metadata: Optional metadata for the query ()))))))))used for filtering)
      
  }
    Returns:
      Tuple of ()))))))))cached_response, similarity_score, cache_metadata)
      """
    # Periodically clean expired entries
      if ($1) {  # Clean roughly every 10 seconds
      this._clean_expired_entries())))))))))
      
    # Generate embedding for the query
      query_embedding = this._generate_embedding()))))))))query)
    
    with this.lock:
      # Find the most similar cached entry {::::
      most_similar_key, similarity = this._find_most_similar()))))))))query_embedding)
      
      if ($1) {
        # Cache hit
        cached_embedding, response, timestamp, cached_metadata = this.cache[]],,most_similar_key]
        
      }
        # Update position in OrderedDict if ($1) {
        if ($1) {
          this.cache.move_to_end()))))))))most_similar_key)
        
        }
          logger.debug()))))))))`$1`)
        return response, similarity, cached_metadata
        }
        
    # Cache miss
        logger.debug()))))))))`$1`)
      return null, similarity, null
  
  $1($2): $3 {
    """
    Add a query-response pair to the cache.
    
  }
    Args:
      query: Query string
      response: Response to cache
      metadata: Optional metadata to store with the cache entry {::::
        """
        this._make_space_if_needed())))))))))
    
        query_embedding = this._generate_embedding()))))))))query)
        current_time = time.time())))))))))
    
    with this.lock:
      # Generate a unique cache key
      cache_key = `$1`
      
      # Store the entry {:::: in the cache
      this.cache[]],,cache_key] = ()))))))))
      query_embedding,
      response,
      current_time,
      metadata || {}}}}
      )
      
      # Move to end if ($1) { to mark as most recently used
      if ($1) {
        this.cache.move_to_end()))))))))cache_key)
        
      }
        logger.debug()))))))))`$1`)
  
  $1($2): $3 {
    """Clear all entries from the cache."""
    with this.lock:
      this.cache.clear())))))))))
      logger.info()))))))))"Cache cleared")
  
  }
  def get_stats()))))))))self) -> Dict[]],,str, Any]:
    """Get statistics about the cache."""
    with this.lock:
      current_time = time.time())))))))))
      active_entries = sum()))))))))
      1 for _, _, timestamp, _ in this.Object.values($1))))))))))
      if current_time - timestamp <= this.ttl
      )
      
      return {}}}:
        "total_entries": len()))))))))this.cache),
        "active_entries": active_entries,
        "expired_entries": len()))))))))this.cache) - active_entries,
        "max_size": this.max_cache_size,
        "similarity_threshold": this.similarity_threshold,
        "ttl": this.ttl,
        }


class $1 extends $2 {
  """
  A wrapper around the Gemini API client that adds semantic caching capabilities.
  """
  
}
  def __init__()))))))))
  self,
  base_client: Any,
  embedding_model: Optional[]],,Any] = null,
  $1: number = 0.85,
  $1: number = 1000,
  $1: number = 3600,
  $1: boolean = true
  ):
    """
    Initialize the semantic cache wrapper for Gemini client.
    
    Args:
      base_client: The base Gemini client to wrap
      embedding_model: Model used to generate embeddings for queries
      similarity_threshold: Minimum similarity threshold for cache hits
      max_cache_size: Maximum cache size
      ttl: Time-to-live for cache entries
      cache_enabled: Whether caching is enabled
      """
      this.base_client = base_client
      this.cache_enabled = cache_enabled
    
    # Initialize semantic cache
      this.cache = SemanticCache()))))))))
      embedding_model=embedding_model,
      similarity_threshold=similarity_threshold,
      max_cache_size=max_cache_size,
      ttl=ttl
      )
    
    # Statistics
      this.stats = {}}}
      "total_requests": 0,
      "cache_hits": 0,
      "cache_misses": 0,
      "avg_similarity": 0.0,
      }
      this.stats_lock = threading.Lock())))))))))
  
      async generate_content()))))))))self,
      prompt: Union[]],,str, List],
      $1: number = 0.7,
      max_tokens: Optional[]],,int] = null,
            **kwargs) -> Any:
              """
              Generate content with semantic caching.
    
    Args:
      prompt: The prompt to send to the model
      temperature: Temperature for generation
      max_tokens: Maximum tokens to generate
      **kwargs: Additional arguments for the base client
      
    Returns:
      Generated content response
      """
    # Update request stats
    with this.stats_lock:
      this.stats[]],,"total_requests"] += 1
    
    # Skip cache for non-deterministic generations
    if ($1) {
      logger.debug()))))))))"Bypassing cache due to non-zero temperature || disabled cache")
      return await this.base_client.generate_content()))))))))prompt, temperature, max_tokens, **kwargs)
    
    }
    # Convert prompt to string for caching
    if ($1) ${$1} else {
      cache_key = prompt
      
    }
    # Include important kwargs in the cache key
      cache_metadata = {}}}
      "temperature": temperature,
      "max_tokens": max_tokens,
      **{}}}k: v for k, v in Object.entries($1)))))))))) if k in []],,'stream', 'n']}
      }
    
    # Try to get response from cache
      cached_response, similarity, _ = this.cache.get()))))))))cache_key, metadata=cache_metadata)
    
    # Update similarity stats:
    with this.stats_lock:
      # Running average of similarity scores
      this.stats[]],,"avg_similarity"] = ()))))))))
      ()))))))))this.stats[]],,"avg_similarity"] * ()))))))))this.stats[]],,"total_requests"] - 1) + similarity) /
      this.stats[]],,"total_requests"]
      )
    
    if ($1) {
      # Cache hit
      with this.stats_lock:
        this.stats[]],,"cache_hits"] += 1
        logger.info()))))))))`$1`)
      return cached_response
    
    }
    # Cache miss - call the base client
    with this.stats_lock:
      this.stats[]],,"cache_misses"] += 1
    
      logger.info()))))))))`$1`)
      response = await this.base_client.generate_content()))))))))prompt, temperature, max_tokens, **kwargs)
    
    # Store in cache if ($1) {
    if ($1) {
      this.cache.put()))))))))cache_key, response, metadata=cache_metadata)
    
    }
      return response
  
    }
      async generate_content_stream()))))))))self,
      prompt: Union[]],,str, List],
      $1: number = 0.7,
      max_tokens: Optional[]],,int] = null,
                **kwargs) -> Any:
                  """
                  Generate streaming content ()))))))))bypasses cache).
    
    Args:
      prompt: The prompt to send to the model
      temperature: Temperature for generation
      max_tokens: Maximum tokens to generate
      **kwargs: Additional arguments for the base client
      
    Returns:
      Streaming response
      """
    # Streaming always bypasses cache
    with this.stats_lock:
      this.stats[]],,"total_requests"] += 1
      this.stats[]],,"cache_misses"] += 1
      
      return await this.base_client.generate_content_stream()))))))))prompt, temperature, max_tokens, **kwargs)
  
  # Pass through all other methods to the base client
  $1($2) {
      return getattr()))))))))this.base_client, name)
  
  }
  def get_cache_stats()))))))))self) -> Dict[]],,str, Any]:
    """Get statistics about the cache usage."""
    with this.stats_lock:
      stats_copy = this.stats.copy())))))))))
    
    # Add cache internal stats
      cache_stats = this.cache.get_stats())))))))))
    return {}}}**stats_copy, **cache_stats}
  
  $1($2): $3 {
    """Clear the cache."""
    this.cache.clear())))))))))
    
  }
  $1($2): $3 ${$1}")


# Example usage:
async $1($2) {
  """
  Example of how to use the semantic cache with the Gemini API.
  """
  # Import your actual Gemini client here
  try ${$1} catch($2: $1) {
    # Mock client for demonstration
    class $1 extends $2 {
      async $1($2) {
        console.log($1)))))))))`$1`)
        await asyncio.sleep()))))))))1)  # Simulate API delay
      return `$1`
      }
        
    }
      async $1($2) {
        console.log($1)))))))))`$1`)
        await asyncio.sleep()))))))))1)  # Simulate API delay
      return iter()))))))))[]],,`$1`])
      }
    
  }
      GeminiClient = MockGeminiClient
  
}
  # Create base client
      base_client = GeminiClient())))))))))
  
  # Create semantic cache wrapper
      cached_client = SemanticCacheGeminiClient()))))))))
      base_client=base_client,
      similarity_threshold=0.85,
      max_cache_size=100,
      ttl=3600
      )
  
  # Example prompts with semantic similarity
      prompts = []],,
      "What is the capital of France?",
      "Could you tell me the capital city of France?",  # Semantically similar
      "What's the capital of France?",  # Semantically similar
      "What is the population of Paris?",  # Different question
    "What is the capital of Italy?",  # Different country {::::
      "What's France's capital city?",  # Very similar to earlier prompts
      "Paris is the capital of which country {::::?",  # Related but different structure
      "Tell me about the capital of France",  # Request for more information
      ]
  
  for (const $1 of $2) {
    console.log($1)))))))))`$1`)
    response = await cached_client.generate_content()))))))))prompt, temperature=0.0)
    console.log($1)))))))))`$1`)
  
  }
  # Print cache stats
    console.log($1)))))))))"\nCache Statistics:")
  for key, value in cached_client.get_cache_stats()))))))))).items()))))))))):
    console.log($1)))))))))`$1`)

if ($1) {
  asyncio.run()))))))))example_usage()))))))))))