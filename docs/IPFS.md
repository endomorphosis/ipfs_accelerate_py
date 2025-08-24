# IPFS Accelerate Python - IPFS Integration Guide

This guide covers the IPFS (InterPlanetary File System) integration features of the IPFS Accelerate Python framework.

## Table of Contents

- [Overview](#overview)
- [IPFS Setup](#ipfs-setup)
- [Basic IPFS Operations](#basic-ipfs-operations)
- [Content-Addressed Model Storage](#content-addressed-model-storage)
- [Distributed Inference](#distributed-inference)
- [Peer-to-Peer Model Distribution](#peer-to-peer-model-distribution)
- [Caching and Optimization](#caching-and-optimization)
- [Advanced IPFS Features](#advanced-ipfs-features)
- [Troubleshooting](#troubleshooting)

## Overview

The IPFS Accelerate Python framework leverages IPFS for:

- **Content-addressed storage**: Models and data are stored with cryptographic hashes
- **Distributed inference**: Offload inference to remote IPFS nodes
- **Efficient caching**: Automatic caching of frequently used models
- **P2P distribution**: Reduce bandwidth through peer-to-peer model sharing
- **Redundancy**: Multiple copies of models across the network

### Key Benefits

1. **Reduced Bandwidth**: Models are cached locally and shared across peers
2. **Scalability**: Distribute inference load across multiple nodes
3. **Reliability**: Redundant storage prevents data loss
4. **Efficiency**: Content deduplication saves storage space
5. **Decentralization**: No single point of failure

## IPFS Setup

### Local IPFS Node Setup

#### Installing IPFS

```bash
# Install IPFS (Linux/macOS)
wget https://dist.ipfs.tech/kubo/v0.21.0/kubo_v0.21.0_linux-amd64.tar.gz
tar -xvzf kubo_v0.21.0_linux-amd64.tar.gz
cd kubo
sudo bash install.sh

# Initialize IPFS
ipfs init

# Start IPFS daemon
ipfs daemon
```

#### IPFS Configuration

```bash
# Configure IPFS for better performance
ipfs config Addresses.Gateway /ip4/127.0.0.1/tcp/8080
ipfs config Addresses.API /ip4/127.0.0.1/tcp/5001

# Enable experimental features
ipfs config --json Experimental.FilestoreEnabled true
ipfs config --json Experimental.UrlstoreEnabled true

# Restart daemon
ipfs shutdown
ipfs daemon
```

### Framework Configuration

```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Basic IPFS configuration
ipfs_config = {
    "ipfs": {
        "gateway": "http://localhost:8080/ipfs/",
        "local_node": "http://localhost:5001",
        "timeout": 30,
        "retry_count": 3,
        "enable_local_gateway": True
    }
}

accelerator = ipfs_accelerate_py(ipfs_config, {})
```

### Remote Gateway Configuration

```python
# Use public IPFS gateways (for testing only)
public_gateway_config = {
    "ipfs": {
        "gateway": "https://ipfs.io/ipfs/",
        "timeout": 60,
        "retry_count": 5,
        "fallback_gateways": [
            "https://gateway.pinata.cloud/ipfs/",
            "https://cloudflare-ipfs.com/ipfs/",
            "https://dweb.link/ipfs/"
        ]
    }
}

# Production configuration with dedicated gateway
production_config = {
    "ipfs": {
        "gateway": "https://your-ipfs-gateway.com/ipfs/",
        "local_node": "http://localhost:5001",
        "api_key": "your-api-key",  # If required
        "timeout": 45,
        "enable_encryption": True,   # Encrypt sensitive data
        "enable_compression": True   # Compress data before storage
    }
}
```

## Basic IPFS Operations

### Storing Data to IPFS

```python
import asyncio
import json

async def store_model_data():
    accelerator = ipfs_accelerate_py({
        "ipfs": {"local_node": "http://localhost:5001"}
    }, {})
    
    # Store inference results
    result_data = {
        "model": "bert-base-uncased",
        "embedding": [0.1, 0.2, -0.3, 0.4],
        "timestamp": "2024-08-24T05:45:00Z"
    }
    
    # Convert to bytes and store
    data_bytes = json.dumps(result_data).encode('utf-8')
    cid = await accelerator.store_to_ipfs(data_bytes)
    
    print(f"Stored data with CID: {cid}")
    return cid

# Example usage
stored_cid = asyncio.run(store_model_data())
```

### Retrieving Data from IPFS

```python
async def retrieve_model_data(cid):
    accelerator = ipfs_accelerate_py({
        "ipfs": {"local_node": "http://localhost:5001"}
    }, {})
    
    # Retrieve data by CID
    data_bytes = await accelerator.query_ipfs(cid)
    
    # Parse JSON data
    result_data = json.loads(data_bytes.decode('utf-8'))
    
    print(f"Retrieved data: {result_data}")
    return result_data

# Example usage
if 'stored_cid' in locals():
    retrieved_data = asyncio.run(retrieve_model_data(stored_cid))
```

### Bulk Operations

```python
async def bulk_ipfs_operations():
    accelerator = ipfs_accelerate_py({
        "ipfs": {"local_node": "http://localhost:5001"}
    }, {})
    
    # Store multiple results
    results = [
        {"model": "bert-base", "score": 0.95},
        {"model": "gpt2", "tokens": 150},
        {"model": "vit-base", "classification": "cat"}
    ]
    
    cids = []
    for result in results:
        data_bytes = json.dumps(result).encode('utf-8')
        cid = await accelerator.store_to_ipfs(data_bytes)
        cids.append(cid)
    
    print(f"Stored {len(cids)} results in IPFS")
    
    # Retrieve all results
    retrieved_results = []
    for cid in cids:
        data_bytes = await accelerator.query_ipfs(cid)
        result = json.loads(data_bytes.decode('utf-8'))
        retrieved_results.append(result)
    
    return retrieved_results

bulk_results = asyncio.run(bulk_ipfs_operations())
```

## Content-Addressed Model Storage

### Model Metadata Storage

```python
async def store_model_metadata():
    """Store model metadata in IPFS for content-addressed access."""
    
    accelerator = ipfs_accelerate_py({
        "ipfs": {"local_node": "http://localhost:5001"}
    }, {})
    
    # Model metadata
    model_metadata = {
        "name": "bert-base-uncased",
        "type": "text_embedding",
        "version": "1.0.0",
        "architecture": "BERT",
        "parameters": 110000000,
        "input_shape": [512],
        "output_shape": [768],
        "supported_hardware": ["cpu", "cuda", "openvino"],
        "precision": ["fp32", "fp16"],
        "created_at": "2024-08-24T05:45:00Z",
        "tags": ["nlp", "embedding", "transformer"]
    }
    
    # Store metadata
    metadata_bytes = json.dumps(model_metadata, indent=2).encode('utf-8')
    metadata_cid = await accelerator.store_to_ipfs(metadata_bytes)
    
    print(f"Model metadata stored with CID: {metadata_cid}")
    
    # Create a model registry entry
    registry_entry = {
        "model_id": "bert-base-uncased",
        "metadata_cid": metadata_cid,
        "versions": {
            "1.0.0": {
                "metadata_cid": metadata_cid,
                "model_cid": "QmExampleModelCID123",  # Would be actual model file CID
                "updated_at": "2024-08-24T05:45:00Z"
            }
        }
    }
    
    registry_bytes = json.dumps(registry_entry, indent=2).encode('utf-8')
    registry_cid = await accelerator.store_to_ipfs(registry_bytes)
    
    print(f"Model registry entry stored with CID: {registry_cid}")
    
    return metadata_cid, registry_cid

metadata_cid, registry_cid = asyncio.run(store_model_metadata())
```

### Model Discovery

```python
async def discover_models():
    """Discover available models through IPFS."""
    
    accelerator = ipfs_accelerate_py({
        "ipfs": {"local_node": "http://localhost:5001"}
    }, {})
    
    # Known model registry CIDs (in practice, these would be published)
    known_registries = [
        registry_cid,  # From previous example
        # "QmAnotherRegistryCID456",
        # "QmYetAnotherRegistryCID789"
    ]
    
    discovered_models = []
    
    for registry_cid in known_registries:
        try:
            # Retrieve registry data
            registry_bytes = await accelerator.query_ipfs(registry_cid)
            registry_data = json.loads(registry_bytes.decode('utf-8'))
            
            # Extract model information
            model_info = {
                "model_id": registry_data["model_id"],
                "metadata_cid": registry_data["metadata_cid"],
                "versions": list(registry_data["versions"].keys()),
                "latest_version": max(registry_data["versions"].keys())
            }
            
            discovered_models.append(model_info)
            
        except Exception as e:
            print(f"Failed to retrieve registry {registry_cid}: {e}")
    
    print(f"Discovered {len(discovered_models)} models")
    return discovered_models

discovered = asyncio.run(discover_models())
```

## Distributed Inference

### IPFS-Accelerated Inference

```python
async def ipfs_distributed_inference():
    """Run inference using IPFS network acceleration."""
    
    accelerator = ipfs_accelerate_py({
        "ipfs": {
            "gateway": "http://localhost:8080/ipfs/",
            "local_node": "http://localhost:5001",
            "enable_distributed_inference": True
        }
    }, {})
    
    # Input data
    input_data = {
        "input_ids": [101, 2054, 2003, 6283, 4083, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1]
    }
    
    # Run distributed inference
    result = await accelerator.accelerate_inference(
        model="bert-base-uncased",
        input_data=input_data,
        use_ipfs=True
    )
    
    print(f"Distributed inference result: {result}")
    return result

distributed_result = asyncio.run(ipfs_distributed_inference())
```

### Provider Discovery and Selection

```python
async def select_optimal_provider():
    """Find and select optimal IPFS providers for inference."""
    
    accelerator = ipfs_accelerate_py({
        "ipfs": {"local_node": "http://localhost:5001"}
    }, {})
    
    model = "bert-base-uncased"
    
    # Find available providers
    providers = await accelerator.find_providers(model)
    print(f"Found {len(providers)} providers for {model}")
    
    # Evaluate provider performance (mock implementation)
    provider_stats = {}
    for provider_id in providers[:3]:  # Test top 3 providers
        try:
            # Connect to provider
            connected = await accelerator.connect_to_provider(provider_id)
            
            if connected:
                # Mock performance test
                import time
                start_time = time.time()
                
                # Simulate provider communication
                await asyncio.sleep(0.1)  # Mock latency
                
                response_time = time.time() - start_time
                
                provider_stats[provider_id] = {
                    "connected": True,
                    "response_time": response_time,
                    "reliability": 0.95,  # Mock reliability score
                    "load": 0.3  # Mock load factor
                }
            else:
                provider_stats[provider_id] = {"connected": False}
                
        except Exception as e:
            provider_stats[provider_id] = {"error": str(e)}
    
    # Select best provider
    best_provider = min(
        [pid for pid, stats in provider_stats.items() if stats.get("connected")],
        key=lambda pid: provider_stats[pid]["response_time"],
        default=None
    )
    
    print(f"Selected provider: {best_provider}")
    print("Provider statistics:")
    for pid, stats in provider_stats.items():
        print(f"  {pid[:20]}...: {stats}")
    
    return best_provider, provider_stats

best_provider, stats = asyncio.run(select_optimal_provider())
```

## Peer-to-Peer Model Distribution

### Model Sharing Network

```python
class IPFSModelNetwork:
    """Network for sharing and distributing models via IPFS."""
    
    def __init__(self, ipfs_config):
        self.accelerator = ipfs_accelerate_py({"ipfs": ipfs_config}, {})
        self.model_cache = {}
        self.peer_registry = {}
    
    async def publish_model(self, model_id, model_data, metadata):
        """Publish a model to the IPFS network."""
        
        # Store model data
        model_bytes = json.dumps(model_data).encode('utf-8')
        model_cid = await self.accelerator.store_to_ipfs(model_bytes)
        
        # Store metadata
        full_metadata = {
            **metadata,
            "model_cid": model_cid,
            "published_at": time.time(),
            "publisher": "local_node"
        }
        metadata_bytes = json.dumps(full_metadata).encode('utf-8')
        metadata_cid = await self.accelerator.store_to_ipfs(metadata_bytes)
        
        # Update local cache
        self.model_cache[model_id] = {
            "model_cid": model_cid,
            "metadata_cid": metadata_cid,
            "local_copy": True
        }
        
        print(f"Published model {model_id}")
        print(f"  Model CID: {model_cid}")
        print(f"  Metadata CID: {metadata_cid}")
        
        return model_cid, metadata_cid
    
    async def discover_models(self, tags=None):
        """Discover models shared by network peers."""
        
        # Mock peer discovery (in practice, use DHT or pubsub)
        mock_peer_models = [
            {
                "model_id": "bert-large-uncased",
                "metadata_cid": "QmMockMetadata123",
                "peer_id": "QmMockPeer456",
                "tags": ["nlp", "bert", "large"]
            },
            {
                "model_id": "gpt2-medium",
                "metadata_cid": "QmMockMetadata789", 
                "peer_id": "QmMockPeer321",
                "tags": ["nlp", "gpt", "generation"]
            }
        ]
        
        # Filter by tags if specified
        if tags:
            mock_peer_models = [
                model for model in mock_peer_models
                if any(tag in model["tags"] for tag in tags)
            ]
        
        print(f"Discovered {len(mock_peer_models)} models")
        return mock_peer_models
    
    async def download_model(self, model_id, metadata_cid):
        """Download a model from the network."""
        
        try:
            # Get metadata first
            metadata_bytes = await self.accelerator.query_ipfs(metadata_cid)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Download model data
            model_cid = metadata["model_cid"]
            model_bytes = await self.accelerator.query_ipfs(model_cid)
            model_data = json.loads(model_bytes.decode('utf-8'))
            
            # Cache locally
            self.model_cache[model_id] = {
                "model_cid": model_cid,
                "metadata_cid": metadata_cid,
                "local_copy": True,
                "metadata": metadata,
                "model_data": model_data
            }
            
            print(f"Downloaded model {model_id}")
            return model_data
            
        except Exception as e:
            print(f"Failed to download model {model_id}: {e}")
            return None

# Example usage
async def model_sharing_example():
    network = IPFSModelNetwork({
        "local_node": "http://localhost:5001"
    })
    
    # Publish a model
    model_data = {
        "architecture": "BERT",
        "weights": "mock_weights_data",
        "config": {"hidden_size": 768, "num_layers": 12}
    }
    
    metadata = {
        "name": "custom-bert-model",
        "version": "1.0",
        "type": "text_embedding",
        "tags": ["nlp", "bert", "custom"]
    }
    
    model_cid, metadata_cid = await network.publish_model(
        "custom-bert-model", model_data, metadata
    )
    
    # Discover available models
    available_models = await network.discover_models(tags=["nlp"])
    
    # Download a model
    if available_models:
        first_model = available_models[0]
        downloaded = await network.download_model(
            first_model["model_id"],
            first_model["metadata_cid"]
        )
    
    return network

network = asyncio.run(model_sharing_example())
```

## Caching and Optimization

### Intelligent Caching

```python
import hashlib
import time
import os

class IPFSCache:
    """Intelligent caching system for IPFS content."""
    
    def __init__(self, cache_dir="./ipfs_cache", max_size_gb=5):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache metadata
        self.cache_index = {}
        self.access_times = {}
        self.load_cache_index()
    
    def _get_cache_path(self, cid):
        """Get local cache file path for a CID."""
        return os.path.join(self.cache_dir, f"{cid}.cache")
    
    def _get_index_path(self):
        """Get cache index file path."""
        return os.path.join(self.cache_dir, "cache_index.json")
    
    def load_cache_index(self):
        """Load cache index from disk."""
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                data = json.load(f)
                self.cache_index = data.get("index", {})
                self.access_times = data.get("access_times", {})
    
    def save_cache_index(self):
        """Save cache index to disk."""
        index_path = self._get_index_path()
        with open(index_path, 'w') as f:
            json.dump({
                "index": self.cache_index,
                "access_times": self.access_times
            }, f, indent=2)
    
    def is_cached(self, cid):
        """Check if content is cached locally."""
        return cid in self.cache_index and os.path.exists(self._get_cache_path(cid))
    
    def cache_content(self, cid, content_bytes):
        """Cache content locally."""
        cache_path = self._get_cache_path(cid)
        
        # Write content to cache
        with open(cache_path, 'wb') as f:
            f.write(content_bytes)
        
        # Update index
        self.cache_index[cid] = {
            "size": len(content_bytes),
            "cached_at": time.time(),
            "path": cache_path
        }
        self.access_times[cid] = time.time()
        
        # Check cache size and evict if necessary
        self._evict_if_necessary()
        
        # Save index
        self.save_cache_index()
    
    def get_cached_content(self, cid):
        """Get content from local cache."""
        if not self.is_cached(cid):
            return None
        
        cache_path = self._get_cache_path(cid)
        with open(cache_path, 'rb') as f:
            content = f.read()
        
        # Update access time
        self.access_times[cid] = time.time()
        self.save_cache_index()
        
        return content
    
    def _get_cache_size(self):
        """Get current cache size in bytes."""
        return sum(info["size"] for info in self.cache_index.values())
    
    def _evict_if_necessary(self):
        """Evict least recently used items if cache is full."""
        while self._get_cache_size() > self.max_size_bytes:
            # Find least recently used item
            lru_cid = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            
            # Remove from cache
            cache_path = self.cache_index[lru_cid]["path"]
            if os.path.exists(cache_path):
                os.remove(cache_path)
            
            del self.cache_index[lru_cid]
            del self.access_times[lru_cid]
            
            print(f"Evicted {lru_cid} from cache")

# Enhanced accelerator with caching
class CachedIPFSAccelerator:
    """IPFS accelerator with intelligent caching."""
    
    def __init__(self, ipfs_config, cache_config=None):
        self.accelerator = ipfs_accelerate_py({"ipfs": ipfs_config}, {})
        
        cache_config = cache_config or {}
        self.cache = IPFSCache(
            cache_dir=cache_config.get("cache_dir", "./ipfs_cache"),
            max_size_gb=cache_config.get("max_size_gb", 5)
        )
    
    async def query_ipfs_cached(self, cid):
        """Query IPFS with caching."""
        
        # Check cache first
        cached_content = self.cache.get_cached_content(cid)
        if cached_content is not None:
            print(f"Cache hit for {cid}")
            return cached_content
        
        print(f"Cache miss for {cid}, fetching from IPFS...")
        
        # Fetch from IPFS
        content = await self.accelerator.query_ipfs(cid)
        
        # Cache the content
        self.cache.cache_content(cid, content)
        
        return content
    
    async def store_to_ipfs_cached(self, data):
        """Store to IPFS and cache locally."""
        
        # Generate expected CID (simplified)
        cid_hash = hashlib.sha256(data).hexdigest()
        mock_cid = f"Qm{cid_hash[:44]}"  # Simplified CID generation
        
        # Check if already cached
        if self.cache.is_cached(mock_cid):
            print(f"Content already cached as {mock_cid}")
            return mock_cid
        
        # Store to IPFS
        actual_cid = await self.accelerator.store_to_ipfs(data)
        
        # Cache locally
        self.cache.cache_content(actual_cid, data)
        
        return actual_cid

# Example usage
async def caching_example():
    cached_accelerator = CachedIPFSAccelerator(
        ipfs_config={"local_node": "http://localhost:5001"},
        cache_config={"max_size_gb": 2}
    )
    
    # Test data
    test_data = json.dumps({
        "model": "bert-base-uncased",
        "results": [0.1, 0.2, 0.3, 0.4, 0.5]
    }).encode('utf-8')
    
    # Store data
    cid = await cached_accelerator.store_to_ipfs_cached(test_data)
    print(f"Stored with CID: {cid}")
    
    # Retrieve data (should hit cache on second call)
    content1 = await cached_accelerator.query_ipfs_cached(cid)
    content2 = await cached_accelerator.query_ipfs_cached(cid)  # Cache hit
    
    print(f"Content matches: {content1 == content2}")
    
    return cached_accelerator

cached_acc = asyncio.run(caching_example())
```

## Advanced IPFS Features

### Content Verification

```python
import hashlib

async def verify_content_integrity(accelerator, cid, expected_hash=None):
    """Verify content integrity using cryptographic hashes."""
    
    try:
        # Retrieve content
        content = await accelerator.query_ipfs(cid)
        
        # Calculate hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Verify against expected hash if provided
        if expected_hash:
            if content_hash == expected_hash:
                print(f"✓ Content integrity verified for {cid}")
                return True, content
            else:
                print(f"✗ Content integrity check failed for {cid}")
                print(f"  Expected: {expected_hash}")
                print(f"  Actual:   {content_hash}")
                return False, None
        else:
            print(f"Content hash for {cid}: {content_hash}")
            return True, content
            
    except Exception as e:
        print(f"Failed to verify content integrity: {e}")
        return False, None

# Example usage
async def integrity_check_example():
    accelerator = ipfs_accelerate_py({
        "ipfs": {"local_node": "http://localhost:5001"}
    }, {})
    
    # Store content and get expected hash
    test_content = b"This is test content for integrity checking"
    expected_hash = hashlib.sha256(test_content).hexdigest()
    
    cid = await accelerator.store_to_ipfs(test_content)
    
    # Verify integrity
    is_valid, content = await verify_content_integrity(
        accelerator, cid, expected_hash
    )
    
    return is_valid

integrity_check = asyncio.run(integrity_check_example())
```

### Pin Management

```python
async def manage_ipfs_pins(accelerator, important_cids):
    """Manage IPFS pin operations for important content."""
    
    # Mock pin operations (would use IPFS API in practice)
    pin_operations = []
    
    for cid in important_cids:
        # Pin important content to prevent garbage collection
        print(f"Pinning {cid}...")
        
        # Mock API call: ipfs pin add <cid>
        pin_result = {
            "cid": cid,
            "pinned": True,
            "type": "recursive",
            "timestamp": time.time()
        }
        pin_operations.append(pin_result)
    
    print(f"Pinned {len(pin_operations)} objects")
    return pin_operations

# Example usage
important_cids = [
    "QmExampleImportantModel123",
    "QmExampleCriticalData456"
]

# pins = asyncio.run(manage_ipfs_pins(accelerator, important_cids))
```

### Pubsub for Real-time Updates

```python
class IPFSPubSub:
    """IPFS pubsub system for real-time model updates."""
    
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.subscriptions = {}
        self.message_handlers = {}
    
    async def publish_model_update(self, topic, model_id, update_data):
        """Publish model update to a topic."""
        
        message = {
            "type": "model_update",
            "model_id": model_id,
            "timestamp": time.time(),
            "data": update_data
        }
        
        # Mock publish (would use IPFS pubsub API)
        print(f"Publishing to {topic}: {message}")
        
        # Store message in IPFS for persistence
        message_bytes = json.dumps(message).encode('utf-8')
        message_cid = await self.accelerator.store_to_ipfs(message_bytes)
        
        return message_cid
    
    async def subscribe_to_updates(self, topic, handler):
        """Subscribe to model updates on a topic."""
        
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        
        self.subscriptions[topic].append(handler)
        self.message_handlers[topic] = handler
        
        print(f"Subscribed to topic: {topic}")
        
        # Mock subscription (would use IPFS pubsub API)
        return True
    
    async def handle_message(self, topic, message):
        """Handle received message."""
        
        if topic in self.message_handlers:
            handler = self.message_handlers[topic]
            await handler(message)

# Example usage
async def pubsub_example():
    accelerator = ipfs_accelerate_py({
        "ipfs": {"local_node": "http://localhost:5001"}
    }, {})
    
    pubsub = IPFSPubSub(accelerator)
    
    # Define message handler
    async def model_update_handler(message):
        print(f"Received model update: {message}")
        
        # Process the update
        if message.get("type") == "model_update":
            model_id = message.get("model_id")
            print(f"Updating local cache for model: {model_id}")
    
    # Subscribe to model updates
    await pubsub.subscribe_to_updates("model_updates", model_update_handler)
    
    # Publish an update
    update_cid = await pubsub.publish_model_update(
        "model_updates",
        "bert-base-uncased",
        {"version": "1.1", "improvements": "Performance optimization"}
    )
    
    return pubsub

# pubsub_system = asyncio.run(pubsub_example())
```

## Troubleshooting

### Common Issues and Solutions

#### Connection Issues

```python
async def diagnose_ipfs_connection():
    """Diagnose IPFS connection issues."""
    
    print("IPFS Connection Diagnostic")
    print("=" * 30)
    
    # Test local node connection
    try:
        import requests
        
        # Check IPFS API
        api_response = requests.get("http://localhost:5001/api/v0/version", timeout=5)
        if api_response.status_code == 200:
            version_info = api_response.json()
            print(f"✓ IPFS API accessible (version: {version_info.get('Version', 'unknown')})")
        else:
            print(f"✗ IPFS API error: {api_response.status_code}")
    
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to IPFS API at localhost:5001")
        print("  Solution: Start IPFS daemon with 'ipfs daemon'")
    except Exception as e:
        print(f"✗ IPFS API connection error: {e}")
    
    # Test gateway
    try:
        gateway_response = requests.get("http://localhost:8080/ipfs/QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn", timeout=10)
        if gateway_response.status_code == 200:
            print("✓ IPFS Gateway accessible")
        else:
            print(f"✗ IPFS Gateway error: {gateway_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to IPFS Gateway at localhost:8080")
    except Exception as e:
        print(f"✗ IPFS Gateway error: {e}")
    
    # Test framework integration
    try:
        accelerator = ipfs_accelerate_py({
            "ipfs": {"local_node": "http://localhost:5001"}
        }, {})
        
        # Try a simple operation
        test_data = b"test connectivity"
        cid = await accelerator.store_to_ipfs(test_data)
        retrieved_data = await accelerator.query_ipfs(cid)
        
        if retrieved_data == test_data:
            print("✓ Framework IPFS integration working")
        else:
            print("✗ Framework IPFS integration failed: data mismatch")
            
    except Exception as e:
        print(f"✗ Framework IPFS integration error: {e}")

# Run diagnostic
# asyncio.run(diagnose_ipfs_connection())
```

#### Performance Issues

```python
async def optimize_ipfs_performance():
    """Optimize IPFS performance settings."""
    
    print("IPFS Performance Optimization")
    print("=" * 35)
    
    # Configuration recommendations
    optimizations = [
        {
            "setting": "Swarm.ConnMgr.HighWater",
            "recommended": 2000,
            "description": "Maximum number of connections"
        },
        {
            "setting": "Swarm.ConnMgr.LowWater", 
            "recommended": 600,
            "description": "Target number of connections"
        },
        {
            "setting": "Datastore.StorageMax",
            "recommended": "50GB",
            "description": "Maximum storage for IPFS"
        },
        {
            "setting": "Gateway.HTTPHeaders.Access-Control-Allow-Origin",
            "recommended": ["*"],
            "description": "CORS settings for gateway"
        }
    ]
    
    print("Recommended IPFS configurations:")
    for opt in optimizations:
        print(f"  {opt['setting']}: {opt['recommended']}")
        print(f"    {opt['description']}")
    
    # Performance testing
    accelerator = ipfs_accelerate_py({
        "ipfs": {
            "local_node": "http://localhost:5001",
            "timeout": 60,
            "retry_count": 3
        }
    }, {})
    
    # Test different data sizes
    test_sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB
    
    print("\nPerformance Test Results:")
    for size in test_sizes:
        test_data = b"x" * size
        
        # Upload test
        start_time = time.time()
        cid = await accelerator.store_to_ipfs(test_data)
        upload_time = time.time() - start_time
        
        # Download test
        start_time = time.time()
        retrieved_data = await accelerator.query_ipfs(cid)
        download_time = time.time() - start_time
        
        print(f"  {size:>7} bytes: Upload {upload_time:.3f}s, Download {download_time:.3f}s")
        
        # Verify data integrity
        if retrieved_data != test_data:
            print(f"    ✗ Data integrity check failed!")
        else:
            print(f"    ✓ Data integrity verified")

# Run performance optimization
# asyncio.run(optimize_ipfs_performance())
```

### Error Recovery Strategies

```python
class RobustIPFSAccelerator:
    """IPFS accelerator with robust error recovery."""
    
    def __init__(self, config):
        self.primary_config = config
        self.fallback_gateways = config.get("ipfs", {}).get("fallback_gateways", [])
        self.max_retries = config.get("ipfs", {}).get("retry_count", 3)
        
        self.primary_accelerator = ipfs_accelerate_py(config, {})
        self.current_gateway_index = 0
    
    async def robust_query_ipfs(self, cid):
        """Query IPFS with automatic fallback and retry."""
        
        last_error = None
        
        # Try primary accelerator first
        for attempt in range(self.max_retries):
            try:
                return await self.primary_accelerator.query_ipfs(cid)
            except Exception as e:
                last_error = e
                print(f"Primary IPFS query attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        
        # Try fallback gateways
        for gateway in self.fallback_gateways:
            try:
                fallback_config = {**self.primary_config}
                fallback_config["ipfs"]["gateway"] = gateway
                
                fallback_accelerator = ipfs_accelerate_py(fallback_config, {})
                return await fallback_accelerator.query_ipfs(cid)
                
            except Exception as e:
                print(f"Fallback gateway {gateway} failed: {e}")
                continue
        
        # If all fallbacks failed
        raise Exception(f"All IPFS query attempts failed. Last error: {last_error}")
    
    async def robust_store_to_ipfs(self, data):
        """Store to IPFS with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                return await self.primary_accelerator.store_to_ipfs(data)
            except Exception as e:
                print(f"IPFS store attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    raise Exception(f"Failed to store to IPFS after {self.max_retries} attempts: {e}")

# Example usage
async def robust_ipfs_example():
    config = {
        "ipfs": {
            "gateway": "http://localhost:8080/ipfs/",
            "local_node": "http://localhost:5001",
            "retry_count": 5,
            "fallback_gateways": [
                "https://ipfs.io/ipfs/",
                "https://gateway.pinata.cloud/ipfs/",
                "https://cloudflare-ipfs.com/ipfs/"
            ]
        }
    }
    
    robust_accelerator = RobustIPFSAccelerator(config)
    
    try:
        # Store data with robustness
        test_data = json.dumps({"test": "robust IPFS operations"}).encode('utf-8')
        cid = await robust_accelerator.robust_store_to_ipfs(test_data)
        print(f"Robustly stored data with CID: {cid}")
        
        # Query data with robustness
        retrieved_data = await robust_accelerator.robust_query_ipfs(cid)
        print(f"Robustly retrieved data: {retrieved_data.decode('utf-8')}")
        
        return True
        
    except Exception as e:
        print(f"Robust IPFS operations failed: {e}")
        return False

# robust_success = asyncio.run(robust_ipfs_example())
```

## Best Practices

### Security Considerations

1. **Content Verification**: Always verify content integrity using hashes
2. **Encryption**: Encrypt sensitive data before storing in IPFS
3. **Access Control**: Use private IPFS networks for sensitive content
4. **Pin Management**: Pin important content to prevent loss

### Performance Optimization

1. **Caching**: Implement intelligent caching for frequently accessed content
2. **Parallel Operations**: Use async operations for better throughput
3. **Content Addressing**: Use IPFS's content addressing for deduplication
4. **Gateway Selection**: Choose optimal gateways based on latency and reliability

### Network Design

1. **Provider Discovery**: Implement efficient provider discovery mechanisms
2. **Load Balancing**: Distribute load across multiple providers
3. **Fault Tolerance**: Implement robust error recovery and fallback strategies
4. **Monitoring**: Monitor network performance and adjust configurations

For more IPFS integration examples, see the [examples directory](../examples/) and related documentation.

## Related Documentation

- [Usage Guide](USAGE.md) - General framework usage
- [API Reference](API.md) - Complete API documentation
- [Hardware Optimization](HARDWARE.md) - Hardware-specific features
- [Examples](../examples/README.md) - Practical examples