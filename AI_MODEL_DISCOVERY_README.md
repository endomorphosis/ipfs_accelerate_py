# AI-Powered Model Discovery and Recommendation System

The IPFS Accelerate Model Manager now includes advanced AI capabilities for intelligent model discovery and recommendation. This system combines vector-based documentation search with multi-armed bandit algorithms to provide personalized, learning-based model recommendations.

## 🚀 Key Features

### 📚 Vector Documentation Search
- **Semantic README Search**: Create vector embeddings of all README.md files in the repository
- **Intelligent Documentation Discovery**: Find relevant documentation using natural language queries
- **Context-Aware Help**: Automatically surface relevant documentation based on user tasks
- **Cross-Reference Analysis**: Identify relationships between different components via documentation similarity

### 🎰 Bandit Algorithm Model Recommendation
- **Multi-Armed Bandit Models**: UCB, Thompson Sampling, and Epsilon-Greedy algorithms
- **Contextual Recommendations**: Consider user requirements, hardware, and task type
- **Continuous Learning**: Adapt recommendations based on user feedback
- **Performance Optimization**: Learn which models work best for specific scenarios

### 🧠 Integrated AI Workflow
- **Automated Model Curation**: Discover and categorize models from multiple sources
- **Performance Prediction**: Predict model performance before deployment
- **Requirement Matching**: Automatically match models to user requirements
- **Feedback-Driven Optimization**: Continuously improve recommendations based on real usage

## 📋 Installation Requirements

### Core Dependencies
```bash
pip install sentence-transformers numpy
```

### Optional Dependencies for Enhanced Features
```bash
pip install faiss-cpu  # For faster vector search
pip install scikit-learn  # For additional ML utilities
```

## 🔧 Quick Start

### Basic Vector Documentation Search

```python
from ipfs_accelerate_py.model_manager import VectorDocumentationIndex

# Initialize and create vector index
doc_index = VectorDocumentationIndex()

# Index all README files in the repository
indexed_count = doc_index.index_all_readmes()
print(f"Indexed {indexed_count} documentation sections")

# Search for relevant documentation
results = doc_index.search("How to optimize CUDA performance?", top_k=3)

for result in results:
    print(f"📄 {result.document.file_path}")
    print(f"📂 Section: {result.document.section}")
    print(f"🎯 Similarity: {result.similarity_score:.3f}")
    print(f"📝 Content: {result.document.content[:100]}...")
```

### Bandit-Based Model Recommendation

```python
from ipfs_accelerate_py.model_manager import (
    BanditModelRecommender, RecommendationContext, DataType
)

# Initialize the recommender
recommender = BanditModelRecommender(algorithm="thompson_sampling")

# Create recommendation context
context = RecommendationContext(
    task_type="sentiment_analysis",
    hardware="cuda",
    input_type=DataType.TOKENS,
    output_type=DataType.LOGITS,
    performance_requirements={"latency": "<100ms"}
)

# Get recommendation
recommendation = recommender.recommend_model(context)

if recommendation:
    print(f"💡 Recommended: {recommendation.model_id}")
    print(f"🎯 Confidence: {recommendation.confidence_score:.3f}")
    print(f"📝 Reasoning: {recommendation.reasoning}")
    
    # Provide feedback (0.0 to 1.0, where 1.0 is best)
    recommender.provide_feedback(
        model_id=recommendation.model_id,
        feedback_score=0.85,
        context=context
    )
```

### Integrated AI Workflow

```python
from ipfs_accelerate_py.model_manager import (
    ModelManager, VectorDocumentationIndex, BanditModelRecommender
)

# Complete AI-powered workflow
def ai_model_discovery_workflow(user_query: str, requirements: dict):
    # 1. Search documentation for relevant information
    doc_index = VectorDocumentationIndex()
    if doc_index.load_index():
        doc_results = doc_index.search(user_query, top_k=3)
        print("📚 Found relevant documentation:")
        for result in doc_results:
            print(f"  📄 {result.document.file_path}")
    
    # 2. Get AI-powered model recommendation
    with ModelManager() as manager:
        recommender = BanditModelRecommender(model_manager=manager)
        
        context = RecommendationContext(
            task_type=requirements.get("task_type"),
            hardware=requirements.get("hardware"),
            input_type=requirements.get("input_type"),
            output_type=requirements.get("output_type")
        )
        
        recommendation = recommender.recommend_model(context)
        
        if recommendation:
            print(f"🎯 AI Recommendation: {recommendation.model_id}")
            return recommendation
    
    return None

# Usage example
requirements = {
    "task_type": "text_classification",
    "hardware": "cuda",
    "input_type": DataType.TOKENS,
    "output_type": DataType.LOGITS
}

recommendation = ai_model_discovery_workflow(
    "CUDA text classification models", 
    requirements
)
```

## 📊 Advanced Features

### Vector Documentation Index

#### Customizing the Embedding Model

```python
# Use a different sentence transformer model
doc_index = VectorDocumentationIndex(
    model_name="all-mpnet-base-v2",  # Higher quality embeddings
    storage_path="custom_doc_index.json"
)
```

#### Advanced Search Options

```python
# Search with custom parameters
results = doc_index.search(
    query="WebGPU acceleration setup",
    top_k=5,                    # Number of results
    min_similarity=0.3          # Minimum similarity threshold
)

# Results include detailed information
for result in results:
    print(f"File: {result.document.file_path}")
    print(f"Section: {result.matched_section}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print(f"Title: {result.document.title}")
```

#### Custom Document Processing

```python
# Index specific directories or file patterns
doc_index.index_all_readmes(root_path="/path/to/docs")

# Manual document addition
from ipfs_accelerate_py.model_manager import DocumentEntry

custom_doc = DocumentEntry(
    file_path="custom_guide.md",
    content="Custom documentation content...",
    title="Custom Guide",
    section="Advanced Usage"
)

# Add embedding and include in index
if doc_index.model:
    custom_doc.embedding = doc_index.model.encode(custom_doc.content).tolist()
    doc_index.documents.append(custom_doc)
```

### Bandit Model Recommender

#### Algorithm Selection

```python
# Different bandit algorithms for different use cases

# UCB (Upper Confidence Bound) - Good balance of exploration/exploitation
recommender_ucb = BanditModelRecommender(algorithm="ucb")

# Thompson Sampling - Bayesian approach, good for varied performance
recommender_ts = BanditModelRecommender(algorithm="thompson_sampling")

# Epsilon-Greedy - Simple and interpretable
recommender_eg = BanditModelRecommender(algorithm="epsilon_greedy")
recommender_eg.epsilon = 0.2  # 20% exploration rate
```

#### Advanced Context Definition

```python
# Rich context for better recommendations
context = RecommendationContext(
    task_type="multimodal_understanding",
    hardware="cuda",
    input_type=DataType.IMAGE,
    output_type=DataType.TEXT,
    performance_requirements={
        "latency": "<500ms",
        "memory": "<8GB",
        "accuracy": ">0.9"
    },
    user_id="user123"  # For personalized recommendations
)
```

#### Feedback and Learning

```python
# Detailed feedback with context
recommender.provide_feedback(
    model_id="recommended_model_id",
    feedback_score=0.8,  # User satisfaction
    context=context
)

# Multiple feedback types
feedback_contexts = [
    {"feedback_score": 0.9, "performance_met": True},
    {"feedback_score": 0.6, "performance_met": False, "issue": "too_slow"},
    {"feedback_score": 0.95, "performance_met": True, "deployment": "production"}
]

for feedback in feedback_contexts:
    recommender.provide_feedback(
        model_id="model_id",
        feedback_score=feedback["feedback_score"],
        context=context
    )
```

#### Performance Analytics

```python
# Get comprehensive performance report
report = recommender.get_performance_report()

print(f"Algorithm: {report['algorithm']}")
print(f"Total trials: {report['total_trials']}")

for context_key, context_data in report['contexts'].items():
    print(f"\nContext: {context_key}")
    print(f"  Best model: {context_data['best_model']}")
    print(f"  Best performance: {context_data['best_average_reward']:.3f}")
    
    # Individual model performance
    for model_id, performance in context_data['arms'].items():
        print(f"    {model_id}: {performance['average_reward']:.3f} "
              f"({performance['num_trials']} trials)")
```

## 🎯 Use Cases

### 1. New User Onboarding

```python
def help_new_user(user_question: str):
    """Help new users find relevant documentation and models."""
    doc_index = VectorDocumentationIndex()
    
    # Search for relevant documentation
    doc_results = doc_index.search(user_question, top_k=3)
    
    print("📚 Relevant Documentation:")
    for result in doc_results:
        print(f"  📄 {result.document.file_path}")
        print(f"  📝 {result.document.content[:150]}...")
    
    # Get beginner-friendly model recommendation
    context = RecommendationContext(
        task_type="beginner_friendly",
        hardware="cpu",  # Assume CPU for beginners
        user_id="new_user"
    )
    
    recommender = BanditModelRecommender()
    recommendation = recommender.recommend_model(context)
    
    if recommendation:
        print(f"\n🎯 Recommended starter model: {recommendation.model_id}")

# Usage
help_new_user("How do I get started with text classification?")
```

### 2. Performance Optimization Guidance

```python
def optimize_for_performance(current_model: str, hardware: str):
    """Find better performing alternatives for current setup."""
    doc_index = VectorDocumentationIndex()
    
    # Search for optimization documentation
    optimization_docs = doc_index.search(
        f"{hardware} optimization performance tuning", 
        top_k=2
    )
    
    print("🚀 Performance Optimization Resources:")
    for result in optimization_docs:
        print(f"  📄 {result.document.file_path}")
    
    # Get performance-optimized recommendation
    context = RecommendationContext(
        task_type="performance_optimization",
        hardware=hardware,
        performance_requirements={
            "priority": "speed",
            "acceptable_accuracy_drop": 0.05
        }
    )
    
    recommender = BanditModelRecommender()
    recommendation = recommender.recommend_model(context)
    
    if recommendation and recommendation.model_id != current_model:
        print(f"💡 Better alternative: {recommendation.model_id}")
        print(f"🎯 Confidence: {recommendation.confidence_score:.3f}")

# Usage
optimize_for_performance("bert-base-uncased", "cuda")
```

### 3. Production Deployment Planning

```python
def plan_production_deployment(requirements: dict):
    """Plan production deployment with AI assistance."""
    
    # Search for deployment documentation
    doc_index = VectorDocumentationIndex()
    deployment_docs = doc_index.search(
        f"production deployment {requirements['hardware']} scalability",
        top_k=2
    )
    
    print("🏭 Production Deployment Resources:")
    for result in deployment_docs:
        print(f"  📄 {result.document.file_path}")
    
    # Get production-ready model recommendation
    context = RecommendationContext(
        task_type="production_deployment",
        hardware=requirements["hardware"],
        performance_requirements=requirements,
        user_id="production_team"
    )
    
    with ModelManager() as manager:
        recommender = BanditModelRecommender(model_manager=manager)
        recommendation = recommender.recommend_model(context)
        
        if recommendation:
            # Get detailed model information
            model_details = manager.get_model(recommendation.model_id)
            
            print(f"\n🎯 Production-Ready Model: {recommendation.model_id}")
            print(f"📊 Confidence: {recommendation.confidence_score:.3f}")
            
            if model_details:
                print(f"🏗️ Architecture: {model_details.architecture}")
                print(f"🔧 Supported backends: {model_details.supported_backends}")
                
                if model_details.hardware_requirements:
                    print(f"💾 Hardware requirements: {model_details.hardware_requirements}")

# Usage
production_requirements = {
    "hardware": "cuda",
    "latency": "<50ms",
    "throughput": ">100 requests/sec",
    "memory": "<4GB",
    "availability": "99.9%"
}

plan_production_deployment(production_requirements)
```

## 🔧 Configuration and Customization

### Storage Configuration

```python
# Configure storage paths for persistence
vector_index = VectorDocumentationIndex(
    storage_path="./ai_data/documentation_index.json"
)

bandit_recommender = BanditModelRecommender(
    storage_path="./ai_data/bandit_learning.json"
)
```

### Advanced Algorithm Parameters

```python
# Customize bandit algorithm parameters
recommender = BanditModelRecommender(algorithm="epsilon_greedy")
recommender.epsilon = 0.15  # 15% exploration rate

# For Thompson Sampling, you can adjust priors
# This is done automatically through the alpha/beta parameters in BanditArm
```

### Integration with Existing Systems

```python
# Integrate with existing model management
from model_manager_integration import ModelManagerIntegration

integration = ModelManagerIntegration()

# Import existing models and enhance with AI features
integration.import_from_existing_metadata()

# Add AI-powered search to existing workflows
def enhanced_model_search(query: str):
    # Traditional search
    traditional_results = integration.search_models(query)
    
    # AI-powered documentation search
    doc_index = VectorDocumentationIndex()
    ai_results = doc_index.search(query)
    
    return {
        "models": traditional_results,
        "documentation": ai_results
    }
```

## 📈 Performance and Scalability

### Optimization Tips

1. **Vector Index Optimization**:
   ```python
   # Use higher quality embeddings for better search
   doc_index = VectorDocumentationIndex(model_name="all-mpnet-base-v2")
   
   # Batch process documents for faster indexing
   doc_index.index_all_readmes()  # Processes all at once
   ```

2. **Bandit Algorithm Tuning**:
   ```python
   # Choose algorithm based on use case
   # UCB: Good general purpose
   # Thompson Sampling: Better for varied performance scenarios
   # Epsilon-Greedy: Simple and predictable
   ```

3. **Memory Management**:
   ```python
   # For large repositories, consider chunking documents
   # Or use external vector databases like Pinecone or Weaviate
   ```

### Monitoring and Analytics

```python
# Monitor AI system performance
def monitor_ai_performance():
    recommender = BanditModelRecommender()
    report = recommender.get_performance_report()
    
    # Track key metrics
    total_trials = report['total_trials']
    num_contexts = len(report['contexts'])
    
    # Calculate average performance across contexts
    avg_performance = 0
    for context_data in report['contexts'].values():
        if context_data['best_average_reward'] > 0:
            avg_performance += context_data['best_average_reward']
    
    avg_performance /= max(num_contexts, 1)
    
    print(f"🔍 AI Performance Metrics:")
    print(f"  Total trials: {total_trials}")
    print(f"  Active contexts: {num_contexts}")
    print(f"  Average performance: {avg_performance:.3f}")
    
    return {
        "total_trials": total_trials,
        "contexts": num_contexts,
        "avg_performance": avg_performance
    }

# Run periodic monitoring
metrics = monitor_ai_performance()
```

## 🚀 Future Enhancements

The AI-powered model discovery system is designed for extensibility. Planned enhancements include:

1. **Advanced Vector Search**: Integration with specialized vector databases
2. **Multi-Modal Embeddings**: Support for code, images, and other content types
3. **Federated Learning**: Collaborative improvement across multiple installations
4. **Reinforcement Learning**: More sophisticated optimization algorithms
5. **Natural Language Interface**: Chat-based model discovery and recommendation

---

## 🎯 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install sentence-transformers numpy
   ```

2. **Initialize Systems**:
   ```python
   from ipfs_accelerate_py.model_manager import VectorDocumentationIndex, BanditModelRecommender
   
   # Create vector index
   doc_index = VectorDocumentationIndex()
   doc_index.index_all_readmes()
   
   # Create bandit recommender
   recommender = BanditModelRecommender()
   ```

3. **Start Using**:
   ```python
   # Search documentation
   results = doc_index.search("your question here")
   
   # Get model recommendations
   recommendation = recommender.recommend_model(your_context)
   ```

The AI-powered system learns from every interaction, continuously improving its recommendations and becoming more helpful over time!