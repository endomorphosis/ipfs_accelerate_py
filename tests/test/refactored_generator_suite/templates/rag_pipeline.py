#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) Pipeline Template for IPFS Accelerate Python.

This module implements a pipeline template for RAG models like RAG-Token, 
RAG-Sequence, and custom RAG implementations. It handles retrieval, 
context integration, and generation.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipelineTemplate


class RAGPipelineTemplate(BasePipelineTemplate):
    """Template for Retrieval-Augmented Generation (RAG) model pipelines."""
    
    def __init__(self):
        """Initialize the RAG pipeline template."""
        super().__init__()
        self.pipeline_type = "rag"
        self.input_type = "text"
        self.output_type = "text"
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = False  # RAG typically processes one query at a time
        self.max_batch_size = 1
    
    def get_import_statements(self) -> str:
        """Get RAG pipeline import statements."""
        return """
# RAG pipeline imports
import os
import json
import numpy as np
from typing import List, Dict, Union, Any, Optional, Tuple
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get RAG preprocessing code for specific task types."""
        if task_type == "generative_qa" or task_type == "question_answering":
            return """
# Preprocess for RAG generative QA
# Parse input
if isinstance(text, dict):
    # Advanced input with parameters
    if "question" in text:
        question = text["question"]
    else:
        question = text.get("text", "")
    
    # Get generation parameters
    max_new_tokens = text.get("max_new_tokens", 128)
    temperature = text.get("temperature", 0.7)
    top_p = text.get("top_p", 0.9)
    num_beams = text.get("num_beams", 4)
    
    # RAG-specific parameters
    num_docs = text.get("num_docs", 5)  # Number of documents to retrieve
    retrieval_context = text.get("context", None)  # Pre-provided context
    doc_sep = text.get("doc_sep", "\\n\\n")  # Separator between documents
    
elif isinstance(text, str):
    # Simple question
    question = text
    max_new_tokens = 128
    temperature = 0.7
    top_p = 0.9
    num_beams = 4
    
    # Default RAG parameters
    num_docs = 5
    retrieval_context = None
    doc_sep = "\\n\\n"
    
else:
    # Default fallback
    question = "What is the capital of France?"
    max_new_tokens = 128
    temperature = 0.7
    top_p = 0.9
    num_beams = 4
    
    # Default RAG parameters
    num_docs = 5
    retrieval_context = None
    doc_sep = "\\n\\n"

# Tokenize the input question
inputs = tokenizer(question, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Prepare generation parameters
generation_config = {
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "num_beams": num_beams,
    "num_return_sequences": 1
}

# Store RAG-specific parameters
rag_config = {
    "num_docs": num_docs,
    "retrieval_context": retrieval_context,
    "doc_sep": doc_sep
}

# Merge with any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in generation_config and param_name not in rag_config:
        generation_config[param_name] = param_value
"""
        elif task_type == "document_retrieval":
            return """
# Preprocess for RAG document retrieval
# Parse input
if isinstance(text, dict):
    # Get query or text
    if "query" in text:
        query = text["query"]
    else:
        query = text.get("text", "")
    
    # RAG-specific parameters
    num_docs = text.get("num_docs", 5)  # Number of documents to retrieve
    
elif isinstance(text, str):
    # Simple query
    query = text
    num_docs = 5
    
else:
    # Default fallback
    query = "What is the capital of France?"
    num_docs = 5

# Tokenize the input query
inputs = tokenizer(query, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Store RAG-specific parameters
rag_config = {
    "num_docs": num_docs
}

# Merge with any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in rag_config:
        rag_config[param_name] = param_value
"""
        else:
            # Default preprocessing for other RAG tasks
            return """
# Default preprocessing for RAG models
# Parse input
if isinstance(text, dict):
    # Get input text or query
    if "text" in text:
        input_text = text["text"]
    elif "question" in text:
        input_text = text["question"]
    elif "query" in text:
        input_text = text["query"]
    else:
        input_text = str(text)
    
    # Get generation parameters
    max_new_tokens = text.get("max_new_tokens", 128)
    temperature = text.get("temperature", 0.7)
    top_p = text.get("top_p", 0.9)
    
    # RAG-specific parameters
    num_docs = text.get("num_docs", 5)  # Number of documents to retrieve
    retrieval_context = text.get("context", None)  # Pre-provided context
    doc_sep = text.get("doc_sep", "\\n\\n")  # Separator between documents
    
elif isinstance(text, str):
    # Simple input
    input_text = text
    max_new_tokens = 128
    temperature = 0.7
    top_p = 0.9
    
    # Default RAG parameters
    num_docs = 5
    retrieval_context = None
    doc_sep = "\\n\\n"
    
else:
    # Default fallback
    input_text = "What is the capital of France?"
    max_new_tokens = 128
    temperature = 0.7
    top_p = 0.9
    
    # Default RAG parameters
    num_docs = 5
    retrieval_context = None
    doc_sep = "\\n\\n"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt")

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Prepare configuration
config = {
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "num_docs": num_docs,
    "retrieval_context": retrieval_context,
    "doc_sep": doc_sep
}

# Merge with any additional parameters from kwargs
for param_name, param_value in kwargs.items():
    if param_name not in config:
        config[param_name] = param_value
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get RAG postprocessing code for specific task types."""
        if task_type == "generative_qa" or task_type == "question_answering":
            return """
# Process outputs from RAG generative QA
with self.torch.no_grad():
    # Extract RAG-specific parameters
    num_docs = rag_config.pop("num_docs", 5)
    retrieval_context = rag_config.pop("retrieval_context", None)
    doc_sep = rag_config.pop("doc_sep", "\\n\\n")
    
    # Custom handling if we have pre-provided context
    if retrieval_context is not None:
        # In a real implementation, we would integrate the context into the model's retrieval
        # For demo purposes, we'll just use it directly
        context_used = True
        
        # Format the context appropriately
        if isinstance(retrieval_context, str):
            context_text = retrieval_context
        elif isinstance(retrieval_context, list):
            context_text = doc_sep.join(retrieval_context)
        else:
            context_text = str(retrieval_context)
    else:
        context_used = False
        context_text = None
    
    # For models with RAG capabilities
    try:
        if hasattr(endpoint, "retrieve_and_generate"):
            # Use the model's unified retrieval and generation method
            outputs = endpoint.retrieve_and_generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                num_docs=num_docs,
                **generation_config
            )
            
            # Extract the generated text
            if hasattr(outputs, "sequences"):
                generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            else:
                generated_texts = ["Model output format not recognized"]
            
            # Extract retrieved documents if available
            retrieved_docs = []
            if hasattr(outputs, "retrieved_docs"):
                for doc in outputs.retrieved_docs:
                    retrieved_docs.append({
                        "text": doc.get("text", ""),
                        "title": doc.get("title", ""),
                        "score": float(doc.get("score", 0.0))
                    })
        else:
            # Regular generation with separate retrieval
            # First, try to retrieve documents if we don't have context
            if not context_used and hasattr(endpoint, "retriever"):
                # Use the model's retriever to get documents
                retrieval_output = endpoint.retriever(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    num_docs=num_docs
                )
                
                # Extract document info
                retrieved_docs = []
                if hasattr(retrieval_output, "retrieved_docs"):
                    for doc in retrieval_output.retrieved_docs:
                        retrieved_docs.append({
                            "text": doc.get("text", ""),
                            "title": doc.get("title", ""),
                            "score": float(doc.get("score", 0.0))
                        })
            else:
                retrieved_docs = []
            
            # Now generate using the model
            output_ids = endpoint.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                **generation_config
            )
            
            # Decode the generated text
            generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    except Exception as e:
        print(f"Error during RAG processing: {e}")
        # Fallback to standard generation if RAG features fail
        output_ids = endpoint.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            **generation_config
        )
        
        # Decode the generated text
        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        retrieved_docs = []
    
    # Create results dictionary
    results = {
        "question": question,
        "answer": generated_texts[0] if generated_texts else "",
        "all_answers": generated_texts,
        "context_used": context_used,
        "retrieved_docs": retrieved_docs
    }
    
    # Add generation parameters used
    results["parameters"] = {
        "max_new_tokens": generation_config.get("max_new_tokens", 128),
        "temperature": generation_config.get("temperature", 0.7),
        "top_p": generation_config.get("top_p", 0.9),
        "num_beams": generation_config.get("num_beams", 4),
        "num_docs": num_docs
    }
"""
        elif task_type == "document_retrieval":
            return """
# Process outputs from RAG document retrieval
with self.torch.no_grad():
    # Extract RAG-specific parameters
    num_docs = rag_config.pop("num_docs", 5)
    
    # For models with retriever capabilities
    try:
        if hasattr(endpoint, "retriever"):
            # Use the model's retriever to get documents
            retrieval_output = endpoint.retriever(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                num_docs=num_docs,
                **rag_config
            )
            
            # Extract document info
            retrieved_docs = []
            if hasattr(retrieval_output, "documents"):
                docs = retrieval_output.documents
            elif hasattr(retrieval_output, "retrieved_docs"):
                docs = retrieval_output.retrieved_docs
            else:
                docs = []
                
            for doc in docs:
                retrieved_docs.append({
                    "text": doc.get("text", ""),
                    "title": doc.get("title", ""),
                    "score": float(doc.get("score", 0.0))
                })
                
        elif hasattr(endpoint, "retrieve_docs"):
            # Some models have a direct retrieve_docs method
            docs = endpoint.retrieve_docs(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                num_docs=num_docs,
                **rag_config
            )
            
            # Extract document info
            retrieved_docs = []
            for doc in docs:
                retrieved_docs.append({
                    "text": doc.get("text", ""),
                    "title": doc.get("title", ""),
                    "score": float(doc.get("score", 0.0))
                })
        else:
            # If the model doesn't have explicit retrieval methods
            # Fall back to a full forward pass and extract retrieval info
            outputs = endpoint(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                output_retrieved=True
            )
            
            # Extract document info from the outputs
            retrieved_docs = []
            if hasattr(outputs, "retrieved_doc_ids") and hasattr(endpoint, "retriever"):
                # Some models return document IDs that we need to convert to texts
                doc_ids = outputs.retrieved_doc_ids
                for i in range(min(len(doc_ids), num_docs)):
                    doc = endpoint.retriever.index.get_doc_dicts(doc_ids[i])
                    retrieved_docs.append({
                        "text": doc.get("text", ""),
                        "title": doc.get("title", ""),
                        "score": float(outputs.doc_scores[0][i].item())
                    })
            elif hasattr(outputs, "doc_scores") and hasattr(endpoint, "retriever"):
                # If we just have scores but no explicit doc IDs
                top_docs = self.torch.topk(outputs.doc_scores[0], k=min(outputs.doc_scores.size(1), num_docs))
                for i, (score, idx) in enumerate(zip(top_docs.values, top_docs.indices)):
                    doc = endpoint.retriever.index.get_doc_dicts(idx.item())
                    retrieved_docs.append({
                        "text": doc.get("text", ""),
                        "title": doc.get("title", ""),
                        "score": float(score.item())
                    })
            else:
                # If we can't extract document info, return an empty list
                retrieved_docs = []
    
    except Exception as e:
        print(f"Error during RAG document retrieval: {e}")
        retrieved_docs = []
    
    # Create results dictionary
    results = {
        "query": inputs["input_ids"],
        "retrieved_docs": retrieved_docs,
        "num_docs_requested": num_docs,
        "num_docs_retrieved": len(retrieved_docs)
    }
"""
        else:
            # Default postprocessing for other RAG tasks
            return """
# Default postprocessing for RAG models
with self.torch.no_grad():
    # Extract important parameters
    max_new_tokens = config.pop("max_new_tokens", 128)
    temperature = config.pop("temperature", 0.7)
    top_p = config.pop("top_p", 0.9)
    num_docs = config.pop("num_docs", 5)
    retrieval_context = config.pop("retrieval_context", None)
    doc_sep = config.pop("doc_sep", "\\n\\n")
    
    # For models with RAG capabilities
    try:
        # Determine if we need to generate text
        need_generation = True  # Default assumption
        
        # Check if we have pre-provided context
        if retrieval_context is not None:
            # Format the context appropriately
            if isinstance(retrieval_context, str):
                context_text = retrieval_context
            elif isinstance(retrieval_context, list):
                context_text = doc_sep.join(retrieval_context)
            else:
                context_text = str(retrieval_context)
                
            context_used = True
            
            # In a real implementation, we would use this context
            # For now, we'll set a flag to indicate it was provided
        else:
            context_text = None
            context_used = False
        
        # Try retrieval if appropriate
        retrieved_docs = []
        if hasattr(endpoint, "retriever") and not context_used:
            # Use the model's retriever to get documents
            retrieval_output = None
            try:
                retrieval_output = endpoint.retriever(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    num_docs=num_docs
                )
            except Exception as retrieval_error:
                print(f"Retrieval error: {retrieval_error}")
                
            # Extract document info if available
            if retrieval_output is not None:
                if hasattr(retrieval_output, "documents"):
                    docs = retrieval_output.documents
                elif hasattr(retrieval_output, "retrieved_docs"):
                    docs = retrieval_output.retrieved_docs
                else:
                    docs = []
                    
                for doc in docs:
                    retrieved_docs.append({
                        "text": doc.get("text", ""),
                        "title": doc.get("title", ""),
                        "score": float(doc.get("score", 0.0))
                    })
        
        # Generate text if needed
        generated_text = None
        if need_generation:
            generation_params = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "num_return_sequences": 1
            }
            
            # Generate using the model
            output_ids = endpoint.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                **generation_params
            )
            
            # Decode the generated text
            generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            generated_text = generated_texts[0] if generated_texts else ""
        
        # Create results dictionary
        results = {
            "input_text": inputs["input_ids"],
            "retrieved_docs": retrieved_docs,
            "context_used": context_used
        }
        
        # Add generated text if available
        if generated_text is not None:
            results["generated_text"] = generated_text
        
        # Add parameters used
        results["parameters"] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_docs": num_docs
        }
        
    except Exception as e:
        print(f"Error during RAG processing: {e}")
        # Create a basic result with error info
        results = {
            "input_text": inputs["input_ids"],
            "error": str(e),
            "retrieved_docs": [],
            "context_used": context_used if 'context_used' in locals() else False
        }
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get RAG result formatting code for specific task types."""
        if task_type == "generative_qa" or task_type == "question_answering":
            return """
# Format results for RAG generative QA
return {
    "success": True,
    "rag_qa": {
        "question": results.get("question", ""),
        "answer": results.get("answer", ""),
        "retrieved_docs": results.get("retrieved_docs", []),
        "context_used": results.get("context_used", False),
        "parameters": results.get("parameters", {})
    },
    "device": device,
    "hardware": hardware_label
}
"""
        elif task_type == "document_retrieval":
            return """
# Format results for RAG document retrieval
return {
    "success": True,
    "rag_retrieval": {
        "num_docs_requested": results.get("num_docs_requested", 0),
        "num_docs_retrieved": results.get("num_docs_retrieved", 0),
        "retrieved_docs": results.get("retrieved_docs", [])
    },
    "device": device,
    "hardware": hardware_label
}
"""
        else:
            # Default result formatting for RAG tasks
            return """
# Default format for RAG model results
response = {
    "success": True,
    "rag_output": {
        "retrieved_docs": results.get("retrieved_docs", []),
        "context_used": results.get("context_used", False),
        "parameters": results.get("parameters", {})
    },
    "device": device,
    "hardware": hardware_label
}

# Add generated text if available
if "generated_text" in results:
    response["rag_output"]["generated_text"] = results["generated_text"]

# Add any error information if present
if "error" in results:
    response["error"] = results["error"]
    
return response
"""
    
    def get_mock_input_code(self) -> str:
        """Get RAG mock input code."""
        return """
# Mock RAG input
mock_input = {
    "question": "What is the capital of France?",
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_p": 0.92,
    "num_docs": 3,  # RAG-specific parameter: number of documents to retrieve
    "context": ["Paris is the capital of France."]  # RAG-specific parameter: pre-provided context
}
"""
    
    def get_mock_output_code(self) -> str:
        """Get RAG mock output code."""
        return """
# Mock RAG output
num_docs = 3  # Number of documents to mock

# Create mock retrieved documents
mock_retrieved_docs = []
for i in range(num_docs):
    mock_retrieved_docs.append({
        "text": f"This is mock document {i+1} about the query topic.",
        "title": f"Mock Document {i+1}",
        "score": 0.9 - (i * 0.1)
    })

# Create appropriate mock output based on task
if "question_answering" in task_type or "generative_qa" in task_type:
    # Mock question answering output
    mock_output = type('MockRAGOutput', (), {})()
    mock_output.sequences = self.torch.randint(0, 50000, (1, 20))  # Mock output tokens
    mock_output.retrieved_docs = mock_retrieved_docs
    
elif "document_retrieval" in task_type:
    # Mock document retrieval output
    mock_output = type('MockRAGRetrievalOutput', (), {})()
    
    # Create mock retriever
    mock_retriever = type('MockRetriever', (), {})()
    def mock_retrieve(*args, **kwargs):
        return type('MockRetrievalOutput', (), {"retrieved_docs": mock_retrieved_docs})()
    mock_retriever.__call__ = mock_retrieve
    
    # Add retriever to mock output
    mock_output.retriever = mock_retriever
    mock_output.doc_scores = self.torch.tensor([[0.9, 0.8, 0.7]])
    
else:
    # Default mock output
    mock_output = type('MockRAGOutput', (), {})()
    mock_output.sequences = self.torch.randint(0, 50000, (1, 20))  # Mock output tokens
    mock_output.retrieved_docs = mock_retrieved_docs
    
    # Create mock retriever
    mock_retriever = type('MockRetriever', (), {})()
    def mock_retrieve(*args, **kwargs):
        return type('MockRetrievalOutput', (), {"retrieved_docs": mock_retrieved_docs})()
    mock_retriever.__call__ = mock_retrieve
    
    # Add retriever to mock output
    mock_output.retriever = mock_retriever

return mock_output
"""
    
    def get_pipeline_utilities(self) -> str:
        """Get RAG utility functions."""
        return """
# RAG pipeline utilities
def evaluate_document_relevance(question, documents, metric="overlap"):
    \"\"\"Evaluate the relevance of retrieved documents to a question.
    
    Args:
        question: The query/question text
        documents: List of retrieved documents
        metric: Relevance metric to use (overlap, bm25, etc.)
        
    Returns:
        Dictionary with relevance evaluation
    \"\"\"
    if not documents:
        return {"average_relevance": 0.0, "documents_evaluated": 0}
    
    # For a simple keyword overlap metric
    if metric == "overlap":
        # Convert to lowercase and tokenize question
        question_terms = question.lower().split()
        
        # Calculate overlap for each document
        relevance_scores = []
        for doc in documents:
            if not isinstance(doc, dict) or "text" not in doc:
                continue
                
            # Get document text and convert to lowercase
            doc_text = doc["text"].lower()
            
            # Count overlapping terms
            overlapping_terms = sum(1 for term in question_terms if term in doc_text)
            
            # Calculate relevance score (normalized by question length)
            if question_terms:
                relevance = overlapping_terms / len(question_terms)
            else:
                relevance = 0.0
                
            relevance_scores.append(relevance)
        
        # Calculate average relevance
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
        else:
            avg_relevance = 0.0
            
        return {
            "average_relevance": avg_relevance,
            "documents_evaluated": len(relevance_scores),
            "individual_scores": relevance_scores
        }
    else:
        # Default simple metric
        return {"average_relevance": 0.5, "documents_evaluated": len(documents)}

def format_context_from_documents(documents, max_length=1024, separator="\\n\\n"):
    \"\"\"Format retrieved documents into a context string for use with models.
    
    Args:
        documents: List of retrieved documents
        max_length: Maximum length of the context string
        separator: Separator to use between documents
        
    Returns:
        Formatted context string
    \"\"\"
    if not documents:
        return ""
        
    context_parts = []
    current_length = 0
    
    for doc in documents:
        if not isinstance(doc, dict) or "text" not in doc:
            continue
            
        doc_text = doc["text"]
        
        # Add title if available
        if "title" in doc and doc["title"]:
            doc_text = f"{doc['title']}:\\n{doc_text}"
            
        # Check if adding this document would exceed max length
        if current_length + len(doc_text) + len(separator) <= max_length:
            context_parts.append(doc_text)
            current_length += len(doc_text) + len(separator)
        else:
            # If we would exceed max length, stop adding documents
            break
            
    # Join all document parts with the separator
    return separator.join(context_parts)
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check RAG pipeline compatibility with architecture type."""
        # RAG pipeline is compatible with RAG-based architectures
        return arch_type in [
            "rag",
            "retrieval-augmented-generation",
            "retrieval-augmented",
            "rag-token",
            "rag-sequence"
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check RAG pipeline compatibility with task type."""
        # RAG pipeline is compatible with these tasks
        return task_type in [
            "text_generation",
            "question_answering",
            "generative_qa",
            "document_retrieval"
        ]