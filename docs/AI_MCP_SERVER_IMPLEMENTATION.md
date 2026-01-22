# AI-Powered Model Manager MCP Server with IPFS Integration

## Overview

This implementation provides a complete AI-powered Model Manager MCP (Model Context Protocol) server that integrates intelligent model discovery, multi-armed bandit recommendations, and IPFS content addressing for decentralized model distribution.

## 🚀 Key Features

### 1. Enhanced Model Manager with IPFS Content Addressing
- **IPFS CID Generation**: Automatic generation of IPFS Content IDs for all model files
- **Content Addressing**: Uses `ipfs_multiformats.py` to create CIDs from file hashes
- **Gateway Support**: Provides IPFS gateway URLs for decentralized access
- **Repository Tracking**: Complete file structure tracking from HuggingFace repositories

### 2. AI-Powered Model Recommendations
- **Multi-Armed Bandit Algorithms**: Thompson Sampling, UCB, and Epsilon-Greedy
- **Contextual Learning**: Considers task type, hardware, input/output types
- **Feedback Integration**: Continuous learning from user performance feedback
- **Smart Model Selection**: Automatic model selection when no explicit model specified

### 3. Comprehensive MCP Server Tools
- **Model Discovery**: List, filter, and search models
- **Intelligent Inference**: Common inference patterns with smart model selection
- **IPFS Integration**: Access model files via IPFS
- **Documentation Search**: Semantic search across documentation (when available)

### 4. Inference Capabilities
- **Causal Language Modeling**: GPT-style text generation
- **Masked Language Modeling**: BERT-style masked token prediction
- **Text Classification**: Sentiment analysis, topic classification
- **Embedding Generation**: Text embeddings for similarity and search
- **Image Generation**: Diffusion model support (mock implementation)
- **Question Answering**: Context-based answer extraction

## 📦 Components

### Core Files

1. **`ipfs_accelerate_py/model_manager.py`** (Enhanced)
   - Added IPFS multiformats integration
   - Enhanced `fetch_huggingface_repo_structure()` with CID generation
   - New methods: `get_model_file_ipfs_cid()`, `get_models_with_ipfs_cids()`, `get_ipfs_gateway_urls()`

2. **`ipfs_mcp/ai_model_server.py`** (New)
   - Main MCP server with FastMCP integration
   - Model discovery and recommendation tools
   - IPFS content addressing tools
   - Comprehensive model management

3. **`ipfs_mcp/inference_tools.py`** (New)
   - Inference engine with smart model selection
   - Mock implementations for demonstration
   - Feedback integration for learning

4. **`ai_mcp_demo.py`** (New)
   - Complete demonstration of all features
   - Shows integration between components
   - Example usage patterns

## 🔧 Installation

### Required Dependencies
```bash
pip install multiformats fastmcp
```

### Optional Dependencies (for enhanced features)
```bash
pip install sentence-transformers numpy duckdb requests
```

## 📖 Usage

### Running the MCP Server

```bash
# Basic usage
python ipfs_mcp/ai_model_server.py

# With custom paths
python ipfs_mcp/ai_model_server.py \
  --model-manager-path ./my_models.db \
  --bandit-storage-path ./my_bandit.json \
  --doc-index-path ./my_docs.json

# Network mode (instead of stdio)
python ipfs_mcp/ai_model_server.py \
  --transport sse \
  --host 0.0.0.0 \
  --port 8000
```

### MCP Tools Available

#### Model Discovery Tools
- `list_models`: List available models with filtering
- `get_model_info`: Get detailed model information
- `recommend_model`: Get AI-powered recommendations using bandits
- `provide_model_feedback`: Provide feedback to improve recommendations

#### IPFS Content Addressing Tools
- `get_model_ipfs_cids`: Get IPFS CIDs for all model files
- `get_ipfs_gateway_urls`: Get gateway URLs for IPFS access

#### Inference Tools (with Smart Model Selection)
- `generate_text`: Causal language modeling
- `fill_mask`: Masked language modeling
- `classify_text`: Text classification
- `generate_embeddings`: Text embedding generation
- `generate_image`: Image generation with diffusion
- `answer_question`: Question answering
- `provide_inference_feedback`: Performance feedback for learning

#### Model Management Tools
- `add_huggingface_model`: Add models from HuggingFace Hub
- `search_documentation`: Semantic documentation search

### Programmatic Usage

```python
from ipfs_mcp.ai_model_server import create_ai_model_server
from ipfs_accelerate_py.model_manager import RecommendationContext, DataType

# Create server
server = create_ai_model_server(
    model_manager_path="./models.db",
    bandit_storage_path="./bandit.json",
    doc_index_path="./docs.json"
)

# Get recommendation
context = RecommendationContext(
    task_type="classification",
    hardware="cpu",
    input_type=DataType.TOKENS,
    output_type=DataType.LOGITS
)

recommendation = server.bandit_recommender.recommend_model(context)
print(f"Recommended: {recommendation.model_id}")

# Provide feedback
server.bandit_recommender.provide_feedback(
    recommendation.model_id, 
    0.85,  # Performance score
    context
)
```

## 🧠 AI-Powered Features

### Multi-Armed Bandit Recommendation System

The system uses three bandit algorithms:

1. **Thompson Sampling** (Default)
   - Bayesian approach with Beta distributions
   - Good balance of exploration/exploitation
   - Handles uncertainty well

2. **Upper Confidence Bound (UCB)**
   - Optimistic approach
   - Mathematical guarantees
   - Good for deterministic environments

3. **Epsilon-Greedy**
   - Simple and interpretable
   - Configurable exploration rate
   - Good baseline algorithm

### Smart Model Selection Logic

When no explicit model is provided:

1. **Context Analysis**: Examines task type, hardware, input/output types
2. **Bandit Recommendation**: Uses learning algorithms to suggest best model
3. **Compatibility Check**: Ensures model supports required specifications
4. **Fallback Strategy**: Provides reasonable defaults if no perfect match

### IPFS Content Addressing

The system generates IPFS CIDs using:

1. **File Hash Extraction**: Gets SHA256 hashes from HuggingFace LFS metadata
2. **Multihash Wrapping**: Uses multiformats library for standardization
3. **CID Generation**: Creates CIDv1 identifiers for IPFS network
4. **Gateway URLs**: Provides HTTP URLs for accessing via IPFS gateways

## 🔗 IPFS Integration Details

### Content Addressing Workflow

1. **Repository Fetching**: Downloads file structure from HuggingFace API
2. **Hash Processing**: Extracts Git OIDs and LFS SHA256 hashes
3. **CID Generation**: Creates IPFS CIDs using `ipfs_multiformats.py`
4. **Storage**: Saves CIDs alongside file metadata
5. **Gateway Access**: Provides URLs for IPFS gateway access

### Example IPFS Usage

```python
# Get IPFS CIDs for a model
manager = ModelManager()
cids = manager.get_model_ipfs_cids("bert-base-uncased")

# Access via IPFS gateway
gateway_urls = manager.get_ipfs_gateway_urls("bert-base-uncased")
for file_path, url in gateway_urls.items():
    print(f"{file_path}: {url}")
```

## 📊 Performance and Learning

### Bandit Learning Process

1. **Initial State**: All models start with equal probability
2. **Recommendation**: System selects model based on current knowledge
3. **Execution**: User performs inference with recommended model
4. **Feedback**: User provides performance score (0.0 to 1.0)
5. **Learning**: System updates model performance estimates
6. **Improvement**: Future recommendations become more accurate

### Context-Aware Recommendations

The system considers:
- **Task Type**: classification, generation, embedding, etc.
- **Hardware**: CPU, CUDA, MPS, etc.
- **Data Types**: tokens, images, audio, embeddings
- **Requirements**: Memory constraints, speed requirements, accuracy needs

## 🧪 Testing and Validation

### Running the Demo

```bash
# Complete system demonstration
python ai_mcp_demo.py
```

The demo showcases:
- Model manager with IPFS integration
- Bandit recommendation learning
- Vector documentation search (when available)
- MCP server tool integration
- End-to-end workflows

### Expected Output

The demo will show:
- ✅ Model creation and management
- 🎯 Smart recommendations with learning
- 🔗 IPFS CID generation
- 🌐 MCP tool availability
- 📊 Performance metrics

## 🚀 Production Deployment

### Server Configuration

```python
# Production server setup
server = create_ai_model_server(
    model_manager_path="./production/models.duckdb",  # Use DuckDB for scale
    bandit_storage_path="./production/bandit.json",
    doc_index_path="./production/docs.json"
)

# Run with network transport
await server.run(transport="sse", host="0.0.0.0", port=8000)
```

### Environment Variables

```bash
export MODEL_MANAGER_DB_PATH="./models.duckdb"
export BANDIT_STORAGE_PATH="./bandit_data.json"
export DOC_INDEX_PATH="./doc_index.json"
```

## 🎯 Use Cases

### 1. Intelligent Model Discovery
Users can discover optimal models without expertise:

```python
# Natural language query
recommendation = recommend_model(
    task_type="sentiment_analysis",
    hardware="cpu",
    requirements={"accuracy": "high", "speed": "medium"}
)
```

### 2. Automatic Model Selection
Systems can automatically choose models:

```python
# No model specified - system chooses best one
result = generate_text(
    prompt="Hello world",
    # model_id automatically selected
)
```

### 3. Decentralized Model Distribution
Models can be served via IPFS:

```python
# Get IPFS URLs for offline distribution
ipfs_urls = get_model_ipfs_cids("gpt2")
# Share CIDs for peer-to-peer distribution
```

### 4. Performance Learning
Systems improve through usage:

```python
# Continuous feedback improves recommendations
provide_inference_feedback(
    task_type="classification",
    model_id="bert-base-uncased", 
    performance_score=0.92
)
```

## 🔮 Future Enhancements

### Planned Features
1. **Real Inference Integration**: Connect to actual model execution
2. **Advanced IPFS Features**: Pinning, clustering, content discovery
3. **Model Fine-tuning**: Track and recommend fine-tuned variants
4. **Performance Benchmarks**: Automated benchmarking and comparison
5. **Multi-Modal Models**: Enhanced support for vision, audio, multimodal

### Extensibility
The system is designed for easy extension:
- Add new inference types by extending `InferenceTools`
- Support new model types by updating `ModelType` enum
- Integrate new storage backends via `ModelManager`
- Add new bandit algorithms in `BanditModelRecommender`

## 🎉 Summary

This implementation delivers a complete AI-powered Model Manager MCP server that:

✅ **Intelligently recommends models** using proven bandit algorithms  
✅ **Learns from user feedback** to improve over time  
✅ **Provides IPFS content addressing** for decentralized distribution  
✅ **Offers comprehensive MCP tools** for model discovery and inference  
✅ **Supports smart model selection** when no explicit choice is made  
✅ **Enables semantic documentation search** for better discoverability  
✅ **Includes production-ready storage** with DuckDB and JSON backends  

The system transforms static model registries into intelligent, learning platforms that actively help users discover and utilize optimal models while enabling decentralized distribution through IPFS integration.