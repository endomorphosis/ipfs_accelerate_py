#!/usr/bin/env python3
"""
Template for Retrieval-Augmented Generation (RAG) models.

This template is designed for models that combine retrieval of documents
with text generation to create context-aware responses. This includes
models like RAG-Token, RAG-Sequence, and custom implementations.
"""

import os
import time
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from transformers import {model_class_name}, {processor_class_name}
    import transformers
except ImportError:
    raise ImportError(
        "The transformers package is required to use this model. "
        "Please install it with `pip install transformers`."
    )

logger = logging.getLogger(__name__)

class {skillset_class_name}:
    """
    Skillset for {model_type_upper} - a retrieval-augmented generation model
    that combines document retrieval with language model generation.
    """
    
    def __init__(self, model_id: str = "{default_model_id}", device: str = "cpu", **kwargs):
        """
        Initialize the {model_type_upper} model.
        
        Args:
            model_id: HuggingFace model ID or path
            device: Device to run the model on ('cpu', 'cuda', 'rocm', 'mps', etc.)
            **kwargs: Additional arguments to pass to the model
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.retriever = None
        self.is_initialized = False
        
        # Track hardware info for reporting
        self.hardware_info = {
            "device": device,
            "device_name": None,
            "memory_available": None,
            "supports_half_precision": False,
            "rag_specific": {
                "retriever_type": None,
                "index_size": None,
                "retriever_batch_size": None,
                "max_retrieved_documents": None
            }
        }
        
        # Optional configuration
        self.low_memory_mode = kwargs.get("low_memory_mode", False)
        self.max_memory = kwargs.get("max_memory", None)
        self.torch_dtype = kwargs.get("torch_dtype", None)
        self.retriever_batch_size = kwargs.get("retriever_batch_size", 16)
        self.max_retrieved_documents = kwargs.get("max_retrieved_documents", 5)
        self.knowledge_base_path = kwargs.get("knowledge_base_path", None)
        
        # Initialize the model if auto_init is True
        auto_init = kwargs.get("auto_init", True)
        if auto_init:
            self.initialize()
    
    def initialize(self):
        """Initialize the model, tokenizer, and retriever."""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing {self.model_id} on {self.device}")
        start_time = time.time()
        
        try:
            # Check available hardware and set appropriate device
            if self.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
                else:
                    device_name = torch.cuda.get_device_name(0)
                    memory_available = torch.cuda.get_device_properties(0).total_memory
                    supports_half = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
                    self.hardware_info.update({
                        "device_name": device_name,
                        "memory_available": memory_available,
                        "supports_half_precision": supports_half
                    })
            
            # Check if MPS is available when device is mps (Apple Silicon)
            elif self.device == "mps":
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    logger.warning("MPS requested but not available, falling back to CPU")
                    self.device = "cpu"
                else:
                    self.hardware_info.update({
                        "device_name": "Apple MPS",
                        "supports_half_precision": True
                    })
            
            # Check if ROCm/HIP is available when device is rocm
            elif self.device == "rocm":
                rocm_available = False
                try:
                    if hasattr(torch, 'hip') and torch.hip.is_available():
                        rocm_available = True
                    elif torch.cuda.is_available():
                        # Could be ROCm using CUDA API
                        device_name = torch.cuda.get_device_name(0)
                        if "AMD" in device_name or "Radeon" in device_name:
                            rocm_available = True
                            self.hardware_info.update({
                                "device_name": device_name,
                                "memory_available": torch.cuda.get_device_properties(0).total_memory,
                                "supports_half_precision": True  # Most AMD GPUs support half precision
                            })
                except:
                    rocm_available = False
                
                if not rocm_available:
                    logger.warning("ROCm requested but not available, falling back to CPU")
                    self.device = "cpu"
            
            # CPU is the fallback
            if self.device == "cpu":
                self.hardware_info.update({
                    "device_name": "CPU",
                    "supports_half_precision": False
                })
            
            # Determine dtype based on hardware
            if self.torch_dtype is None:
                if self.hardware_info["supports_half_precision"] and not self.low_memory_mode:
                    self.torch_dtype = torch.float16
                else:
                    self.torch_dtype = torch.float32
            
            # Load tokenizer
            self.tokenizer = {processor_class_name}.from_pretrained(self.model_id)
            
            # Load model with appropriate configuration
            load_kwargs = {}
            if self.torch_dtype is not None:
                load_kwargs["torch_dtype"] = self.torch_dtype
            
            if self.low_memory_mode:
                load_kwargs["low_cpu_mem_usage"] = True
                if "llama" in self.model_id.lower() or "mistral" in self.model_id.lower():
                    # Use 4-bit quantization for very large models in low memory mode
                    try:
                        load_kwargs["load_in_4bit"] = True
                        load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                    except:
                        logger.warning("4-bit quantization not available, proceeding without it")
            
            if self.max_memory is not None:
                load_kwargs["max_memory"] = self.max_memory
            
            # Specific handling for device placement
            if self.device.startswith(("cuda", "rocm")) and "gpu_map" not in load_kwargs:
                load_kwargs["device_map"] = "auto"
            
            # Load the RAG model
            self.model = {model_class_name}.from_pretrained(self.model_id, **load_kwargs)
            
            # Move to appropriate device if not using device_map
            if "device_map" not in load_kwargs and not self.device.startswith(("cuda", "rocm")):
                self.model.to(self.device)
            
            # Initialize retriever if not already included in the model
            if not hasattr(self.model, "retriever") or self.model.retriever is None:
                self.initialize_retriever()
            else:
                self.retriever = self.model.retriever
                # Update RAG-specific info
                self.hardware_info["rag_specific"].update({
                    "retriever_type": type(self.retriever).__name__ if self.retriever else None,
                    "retriever_batch_size": self.retriever_batch_size,
                    "max_retrieved_documents": self.max_retrieved_documents
                })
            
            # Log initialization time
            elapsed_time = time.time() - start_time
            logger.info(f"Initialized {self.model_id} in {elapsed_time:.2f} seconds")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing {self.model_id}: {str(e)}")
            raise
    
    def initialize_retriever(self):
        """Initialize the document retriever component."""
        try:
            # Check if knowledge base path is provided
            if self.knowledge_base_path and os.path.exists(self.knowledge_base_path):
                logger.info(f"Initializing retriever with knowledge base: {self.knowledge_base_path}")
                
                # Determine the type of knowledge base
                if self.knowledge_base_path.endswith(".faiss"):
                    # FAISS index
                    from transformers import RagRetriever
                    from datasets import load_from_disk
                    
                    # Load the dataset and create the retriever
                    try:
                        dataset = load_from_disk(self.knowledge_base_path)
                        self.retriever = RagRetriever.from_pretrained(
                            self.model_id, 
                            index_name="custom", 
                            passages=dataset
                        )
                    except Exception as e:
                        logger.error(f"Error loading FAISS index: {str(e)}")
                        self._create_mock_retriever()
                
                elif self.knowledge_base_path.endswith((".txt", ".csv", ".json")):
                    # Text files to be processed into an in-memory index
                    try:
                        documents = self._load_documents_from_file(self.knowledge_base_path)
                        self._create_in_memory_retriever(documents)
                    except Exception as e:
                        logger.error(f"Error creating in-memory retriever: {str(e)}")
                        self._create_mock_retriever()
                
                else:
                    logger.warning(f"Unsupported knowledge base format: {self.knowledge_base_path}")
                    self._create_mock_retriever()
            
            else:
                # No custom knowledge base, try to use model's default retriever if available
                if hasattr(self.model, "retriever") and self.model.retriever is not None:
                    self.retriever = self.model.retriever
                    logger.info("Using model's built-in retriever")
                else:
                    # Create a mock retriever as fallback
                    logger.warning("No knowledge base provided and model has no built-in retriever")
                    self._create_mock_retriever()
            
            # Update RAG-specific hardware info
            self.hardware_info["rag_specific"].update({
                "retriever_type": type(self.retriever).__name__ if self.retriever else None,
                "retriever_batch_size": self.retriever_batch_size,
                "max_retrieved_documents": self.max_retrieved_documents
            })
            
        except Exception as e:
            logger.error(f"Error initializing retriever: {str(e)}")
            self._create_mock_retriever()
    
    def _create_mock_retriever(self):
        """Create a mock retriever for graceful degradation."""
        logger.warning("Creating mock retriever")
        
        class MockRetriever:
            def __init__(self, parent):
                self.parent = parent
                self.last_retrieved_docs = []
            
            def __call__(self, question_input_ids, n_docs=5, **kwargs):
                """Mock retrieval function."""
                # Create mock document results
                retrieved_docs = []
                for i in range(n_docs):
                    retrieved_docs.append({
                        "text": f"This is mock document {i+1} with information about the query.",
                        "title": f"Mock Document {i+1}",
                        "score": 0.9 - (i * 0.1)
                    })
                
                # Save for later access
                self.last_retrieved_docs = retrieved_docs
                
                result = {
                    "retrieved_docs": retrieved_docs,
                    "question_input_ids": question_input_ids
                }
                
                # Create a namespace object to mimic HF's retriever output
                from types import SimpleNamespace
                return SimpleNamespace(**result)
        
        self.retriever = MockRetriever(self)
    
    def _load_documents_from_file(self, file_path):
        """Load documents from various file formats."""
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split by double newlines as paragraph boundaries
            documents = [{"text": p.strip(), "title": f"Document {i+1}"} 
                         for i, p in enumerate(content.split("\n\n")) if p.strip()]
            
        elif file_path.endswith(".csv"):
            import csv
            documents = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                
                text_col = 0  # Default to first column
                title_col = 1 if len(headers) > 1 else None
                
                # Try to find text and title columns by name
                if headers:
                    text_candidates = ["text", "content", "document", "passage"]
                    title_candidates = ["title", "headline", "header", "name"]
                    
                    for candidate in text_candidates:
                        if candidate in [h.lower() for h in headers]:
                            text_col = [h.lower() for h in headers].index(candidate)
                            break
                    
                    for candidate in title_candidates:
                        if candidate in [h.lower() for h in headers]:
                            title_col = [h.lower() for h in headers].index(candidate)
                            break
                
                # Read the documents
                for i, row in enumerate(reader):
                    if not row:
                        continue
                    
                    text = row[text_col] if text_col < len(row) else f"Document {i+1}"
                    title = row[title_col] if title_col is not None and title_col < len(row) else f"Document {i+1}"
                    
                    documents.append({"text": text, "title": title})
        
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            documents = []
            
            # Handle various JSON formats
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # Extract text and title from dictionary
                        text = item.get("text", item.get("content", str(item)))
                        title = item.get("title", f"Document {i+1}")
                        documents.append({"text": text, "title": title})
            
            elif isinstance(data, dict):
                # Handle case where data is a dictionary with documents
                if "documents" in data and isinstance(data["documents"], list):
                    for i, item in enumerate(data["documents"]):
                        if isinstance(item, dict):
                            text = item.get("text", item.get("content", str(item)))
                            title = item.get("title", f"Document {i+1}")
                            documents.append({"text": text, "title": title})
                
                # Handle other dictionary formats by flattening
                else:
                    for key, value in data.items():
                        if isinstance(value, str):
                            documents.append({"text": value, "title": key})
        
        return documents
    
    def _create_in_memory_retriever(self, documents):
        """Create an in-memory retriever from documents."""
        try:
            from transformers import RagRetriever
            from datasets import Dataset
            
            # Convert documents to HuggingFace dataset
            dataset = Dataset.from_dict({
                "text": [doc["text"] for doc in documents],
                "title": [doc["title"] for doc in documents]
            })
            
            # Create the retriever
            self.retriever = RagRetriever.from_pretrained(
                self.model_id,
                index_name="custom",
                passages=dataset
            )
            
            logger.info(f"Created in-memory retriever with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating in-memory retriever: {str(e)}")
            self._create_mock_retriever()
    
    def get_parameters(self):
        """Get model and state parameters."""
        if not self.is_initialized:
            self.initialize()
        
        # Extract RAG-specific parameters
        rag_params = {}
        state_params = {}
        
        if hasattr(self.model, "retriever") and self.model.retriever is not None:
            rag_params["has_retriever"] = True
            rag_params["retriever_type"] = type(self.model.retriever).__name__
            
            # Try to get index size
            if hasattr(self.model.retriever, "index"):
                if hasattr(self.model.retriever.index, "document_ids"):
                    rag_params["index_size"] = len(self.model.retriever.index.document_ids)
                elif hasattr(self.model.retriever.index, "passages"):
                    rag_params["index_size"] = len(self.model.retriever.index.passages)
        else:
            rag_params["has_retriever"] = False
        
        # Get general model parameters
        model_params = {}
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "hidden_size"):
                model_params["hidden_size"] = self.model.config.hidden_size
            if hasattr(self.model.config, "vocab_size"):
                model_params["vocab_size"] = self.model.config.vocab_size
            if hasattr(self.model.config, "model_type"):
                model_params["model_type"] = self.model.config.model_type
            
            # Extract QA-specific parameters if available
            if hasattr(self.model.config, "n_docs"):
                rag_params["default_n_docs"] = self.model.config.n_docs
        
        # Update hardware info
        self.hardware_info["rag_specific"].update(rag_params)
        
        return {
            "model": model_params,
            "rag": rag_params,
            "state": state_params,
            "hardware": self.hardware_info
        }
        
    def format_context_from_documents(self, documents, max_length=1024, separator="\n\n"):
        """
        Format retrieved documents into a context string for use with models.
        
        Args:
            documents: List of retrieved documents
            max_length: Maximum length of the context string
            separator: Separator to use between documents
            
        Returns:
            Formatted context string
        """
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
                doc_text = f"{doc['title']}:\n{doc_text}"
                
            # Check if adding this document would exceed max length
            if current_length + len(doc_text) + len(separator) <= max_length:
                context_parts.append(doc_text)
                current_length += len(doc_text) + len(separator)
            else:
                # If we would exceed max length, stop adding documents
                break
                
        # Join all document parts with the separator
        return separator.join(context_parts)
    
    def retrieve_documents(self, query: str, n_docs: int = None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The text query
            n_docs: Number of documents to retrieve (default is self.max_retrieved_documents)
            **kwargs: Additional arguments for the retriever
            
        Returns:
            Dictionary with retrieved documents and scores
        """
        if not self.is_initialized:
            self.initialize()
        
        if n_docs is None:
            n_docs = self.max_retrieved_documents
        
        # First, tokenize the query
        query_inputs = self.tokenizer(query, return_tensors="pt")
        query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
        
        try:
            # Try to use the retriever
            if self.retriever is not None:
                retrieval_kwargs = kwargs.copy()
                if "num_docs" not in retrieval_kwargs:
                    retrieval_kwargs["n_docs"] = n_docs
                
                # Check retriever API
                if hasattr(self.retriever, "__call__"):
                    retrieval_output = self.retriever(
                        query_inputs["input_ids"],
                        attention_mask=query_inputs.get("attention_mask"),
                        **retrieval_kwargs
                    )
                elif hasattr(self.retriever, "retrieve"):
                    retrieval_output = self.retriever.retrieve(
                        query_inputs["input_ids"],
                        attention_mask=query_inputs.get("attention_mask"),
                        **retrieval_kwargs
                    )
                else:
                    raise AttributeError("Retriever doesn't have a valid retrieval method")
                
                # Extract document info from the outputs
                retrieved_docs = []
                
                # Different ways the retriever might expose documents
                if hasattr(retrieval_output, "retrieved_docs"):
                    docs = retrieval_output.retrieved_docs
                elif hasattr(retrieval_output, "documents"):
                    docs = retrieval_output.documents
                elif hasattr(retrieval_output, "doc_ids") and hasattr(self.retriever, "index"):
                    # We have doc IDs but need to get the actual documents
                    docs = []
                    for doc_id in retrieval_output.doc_ids[:n_docs]:
                        doc = self.retriever.index.get_doc_dicts(doc_id)
                        if doc:
                            docs.append(doc)
                else:
                    # Fallback if we can't extract docs
                    docs = []
                    
                # Convert docs to a consistent format
                for doc in docs:
                    if isinstance(doc, dict):
                        # Extract text and metadata
                        doc_entry = {
                            "text": doc.get("text", doc.get("content", "")),
                            "title": doc.get("title", "Untitled Document")
                        }
                        
                        # Add score if available
                        if "score" in doc:
                            doc_entry["score"] = float(doc["score"])
                        
                        retrieved_docs.append(doc_entry)
                    else:
                        # If it's not a dict, convert to one
                        retrieved_docs.append({
                            "text": str(doc),
                            "title": "Untitled Document"
                        })
                
                # If we have scores separately
                if hasattr(retrieval_output, "doc_scores") and len(retrieved_docs) > 0:
                    doc_scores = retrieval_output.doc_scores
                    if hasattr(doc_scores, "cpu"):
                        doc_scores = doc_scores.cpu().numpy()
                    
                    # Assign scores to documents
                    for i, doc in enumerate(retrieved_docs):
                        if i < len(doc_scores):
                            if "score" not in doc:
                                doc["score"] = float(doc_scores[i])
                
                # Store for later use
                self.last_retrieved_docs = retrieved_docs
                
                return {
                    "query": query,
                    "retrieved_docs": retrieved_docs,
                    "n_docs_requested": n_docs,
                    "n_docs_retrieved": len(retrieved_docs)
                }
            
            else:
                # No retriever available
                logger.warning("No retriever available for document retrieval")
                return {
                    "query": query,
                    "retrieved_docs": [],
                    "n_docs_requested": n_docs,
                    "n_docs_retrieved": 0,
                    "error": "No retriever available"
                }
                
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            return {
                "query": query,
                "retrieved_docs": [],
                "n_docs_requested": n_docs,
                "n_docs_retrieved": 0,
                "error": str(e)
            }
    
    def retrieve_and_generate(self, query: str, n_docs: int = None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve documents and generate an answer based on them.
        
        Args:
            query: The question or text query
            n_docs: Number of documents to retrieve (default is self.max_retrieved_documents)
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary with generated answer and retrieved documents
        """
        if not self.is_initialized:
            self.initialize()
        
        # Set default parameters
        n_docs = n_docs or self.max_retrieved_documents
        max_length = kwargs.pop("max_length", 1024)
        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.9)
        context = kwargs.pop("context", None)  # User can provide custom context
        
        # First, retrieve documents if not provided
        if context is None:
            retrieval_result = self.retrieve_documents(query, n_docs=n_docs)
            retrieved_docs = retrieval_result.get("retrieved_docs", [])
            
            # Create context from retrieved documents
            if retrieved_docs:
                context_parts = []
                for doc in retrieved_docs:
                    doc_text = doc.get("text", "")
                    if "title" in doc and doc["title"]:
                        doc_text = f"{doc['title']}:\n{doc_text}"
                    context_parts.append(doc_text)
                
                context = "\n\n".join(context_parts)
            else:
                context = ""
        
        try:
            # Check if model has specific RAG generation capabilities
            if hasattr(self.model, "generate_with_retrieved_docs"):
                # Use dedicated RAG generation method
                query_inputs = self.tokenizer(query, return_tensors="pt")
                query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
                
                generation_kwargs = {
                    "max_length": max_length,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_docs": n_docs,
                    **kwargs
                }
                
                outputs = self.model.generate_with_retrieved_docs(
                    query_inputs["input_ids"],
                    attention_mask=query_inputs.get("attention_mask"),
                    **generation_kwargs
                )
                
                # Process outputs
                generated_text = self.tokenizer.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )[0]
                
                # Extract retrieved docs if available
                if hasattr(outputs, "retrieved_docs"):
                    retrieved_docs = outputs.retrieved_docs
                else:
                    retrieved_docs = self.last_retrieved_docs if hasattr(self, "last_retrieved_docs") else []
            
            elif hasattr(self.model, "generate"):
                # Standard model with generate method - use retrieved context
                # Format input with context and query
                if context:
                    input_text = f"Context: {context}\n\nQuestion: {query}"
                else:
                    input_text = query
                
                inputs = self.tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                generation_kwargs = {
                    "max_length": max_length,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    **kwargs
                }
                
                # Generate text
                output_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **generation_kwargs
                )
                
                # Decode generated text
                generated_text = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0]
                
                # Use previously retrieved docs
                retrieved_docs = self.last_retrieved_docs if hasattr(self, "last_retrieved_docs") else []
            
            else:
                # Fallback for models without generate method
                logger.warning("Model doesn't support generation, returning context only")
                generated_text = "Model doesn't support text generation"
                retrieved_docs = self.last_retrieved_docs if hasattr(self, "last_retrieved_docs") else []
            
            # Create response
            return {
                "query": query,
                "answer": generated_text,
                "retrieved_docs": retrieved_docs,
                "context_used": bool(context),
                "parameters": {
                    "n_docs": n_docs,
                    "max_length": max_length,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
        except Exception as e:
            logger.error(f"Error during retrieval and generation: {str(e)}")
            return {
                "query": query,
                "answer": f"Error: {str(e)}",
                "retrieved_docs": [],
                "error": str(e)
            }
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Use the RAG model for chat functionality with context retrieval.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generated response and retrieved documents
        """
        if not self.is_initialized:
            self.initialize()
        
        # Set default parameters
        n_docs = kwargs.pop("n_docs", self.max_retrieved_documents)
        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.9)
        
        # Extract the latest user message for retrieval
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        latest_user_message = user_messages[-1] if user_messages else ""
        
        # Retrieve context based on the latest user message
        retrieval_result = self.retrieve_documents(latest_user_message, n_docs=n_docs)
        retrieved_docs = retrieval_result.get("retrieved_docs", [])
        
        try:
            # Format messages for the model
            chat_text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    chat_text += f"System: {content}\n\n"
                elif role == "user":
                    chat_text += f"User: {content}\n\n"
                elif role == "assistant":
                    chat_text += f"Assistant: {content}\n\n"
            
            # Add context from retrieved documents
            context_text = ""
            if retrieved_docs:
                context_parts = []
                for doc in retrieved_docs:
                    doc_text = doc.get("text", "")
                    if "title" in doc and doc["title"]:
                        doc_text = f"{doc['title']}:\n{doc_text}"
                    context_parts.append(doc_text)
                
                context_text = "\n\n".join(context_parts)
            
            # Combine context and chat history
            if context_text:
                input_text = f"Context:\n{context_text}\n\n{chat_text}Assistant:"
            else:
                input_text = f"{chat_text}Assistant:"
            
            # Tokenize and prepare for generation
            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                **kwargs
            }
            
            # Generate response
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **generation_kwargs
            )
            
            # Decode generated text
            generated_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            
            # Extract just the assistant's response (not the prompt)
            assistant_response = generated_text
            if input_text in assistant_response:
                assistant_response = assistant_response[len(input_text):].strip()
            
            # Create response
            return {
                "response": assistant_response,
                "retrieved_docs": retrieved_docs,
                "messages": messages + [{"role": "assistant", "content": assistant_response}],
                "parameters": {
                    "n_docs": n_docs,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
        except Exception as e:
            logger.error(f"Error during RAG chat: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "retrieved_docs": retrieved_docs,
                "messages": messages,
                "error": str(e)
            }
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Get embeddings for text using the RAG model's encoder.
        
        Args:
            text: Single string or list of strings to embed
            **kwargs: Additional embedding parameters
            
        Returns:
            Dictionary with embeddings
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process input to ensure it's a list
        if isinstance(text, str):
            batch_text = [text]
        else:
            batch_text = text
        
        try:
            # Check if the model has a question or document encoder
            if hasattr(self.model, "question_encoder"):
                encoder = self.model.question_encoder
            elif hasattr(self.model, "query_encoder"):
                encoder = self.model.query_encoder
            elif hasattr(self.model, "encoder"):
                encoder = self.model.encoder
            else:
                raise AttributeError("Model doesn't have a suitable encoder component")
            
            # Tokenize the input text
            inputs = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=kwargs.get("max_length", 512)
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = encoder(**inputs)
                
                # Extract embeddings
                if hasattr(outputs, "pooler_output"):
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    # Mean pooling of the last hidden state
                    last_hidden = outputs.last_hidden_state
                    attention_mask = inputs.get("attention_mask", None)
                    
                    if attention_mask is not None:
                        # Mean pooling with attention mask
                        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                        sum_embeddings = torch.sum(last_hidden * mask, 1)
                        sum_mask = torch.sum(mask, 1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        embeddings = sum_embeddings / sum_mask
                    else:
                        # Simple mean pooling
                        embeddings = torch.mean(last_hidden, dim=1)
                else:
                    raise ValueError("Couldn't extract embeddings from model outputs")
            
            # Convert to numpy array
            embeddings_np = embeddings.cpu().numpy()
            
            # Normalize embeddings if requested
            if kwargs.get("normalize", False):
                from sklearn.preprocessing import normalize
                embeddings_np = normalize(embeddings_np)
            
            # Format the result
            if len(batch_text) == 1:
                # Single input case
                result = {
                    "text": text,
                    "embedding": embeddings_np[0].tolist(),
                    "dimensions": embeddings_np.shape[-1]
                }
            else:
                # Batch input case
                result = {
                    "texts": batch_text,
                    "embeddings": [emb.tolist() for emb in embeddings_np],
                    "dimensions": embeddings_np.shape[-1]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return {
                "text": text,
                "error": str(e)
            }
    
    def analyze_retrieval_quality(self, query: str, n_docs: int = None, **kwargs) -> Dict[str, Any]:
        """
        Analyze the quality of document retrieval for a query.
        
        Args:
            query: The text query
            n_docs: Number of documents to retrieve
            **kwargs: Additional retrieval parameters
            
        Returns:
            Dictionary with retrieval quality metrics
        """
        if not self.is_initialized:
            self.initialize()
        
        # Retrieve documents
        retrieval_result = self.retrieve_documents(query, n_docs=n_docs, **kwargs)
        retrieved_docs = retrieval_result.get("retrieved_docs", [])
        
        try:
            # Calculate basic metrics
            metrics = {
                "num_docs_retrieved": len(retrieved_docs),
                "query": query
            }
            
            # Calculate score statistics if available
            if retrieved_docs and "score" in retrieved_docs[0]:
                scores = [doc.get("score", 0) for doc in retrieved_docs]
                metrics["score_min"] = min(scores)
                metrics["score_max"] = max(scores)
                metrics["score_mean"] = sum(scores) / len(scores)
                metrics["score_median"] = sorted(scores)[len(scores) // 2]
            
            # Calculate text length statistics
            if retrieved_docs:
                text_lengths = [len(doc.get("text", "")) for doc in retrieved_docs]
                metrics["text_length_min"] = min(text_lengths)
                metrics["text_length_max"] = max(text_lengths)
                metrics["text_length_mean"] = sum(text_lengths) / len(text_lengths)
                metrics["text_length_total"] = sum(text_lengths)
            
            # Calculate query term overlap
            if retrieved_docs:
                query_terms = set(query.lower().split())
                
                if query_terms:
                    overlap_ratios = []
                    
                    for doc in retrieved_docs:
                        doc_text = doc.get("text", "").lower()
                        doc_terms = set(doc_text.split())
                        
                        if doc_terms:
                            overlap = query_terms.intersection(doc_terms)
                            overlap_ratio = len(overlap) / len(query_terms)
                            overlap_ratios.append(overlap_ratio)
                    
                    if overlap_ratios:
                        metrics["term_overlap_mean"] = sum(overlap_ratios) / len(overlap_ratios)
                        metrics["term_overlap_max"] = max(overlap_ratios)
            
            # Include document brief
            doc_briefs = []
            for i, doc in enumerate(retrieved_docs[:min(5, len(retrieved_docs))]):
                text = doc.get("text", "")
                title = doc.get("title", f"Document {i+1}")
                score = doc.get("score", "N/A")
                
                # Truncate text for the brief
                text_sample = text[:200] + "..." if len(text) > 200 else text
                
                doc_briefs.append({
                    "title": title,
                    "score": score,
                    "text_sample": text_sample,
                    "text_length": len(text)
                })
            
            metrics["document_briefs"] = doc_briefs
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing retrieval quality: {str(e)}")
            return {
                "query": query,
                "num_docs_retrieved": len(retrieved_docs),
                "error": str(e)
            }
    
    def __call__(self, text: Union[str, Dict, List], task: str = "retrieve_and_generate", **kwargs) -> Dict[str, Any]:
        """
        Process text with the RAG model.
        
        Args:
            text: Input text or query (can be a string, dict with parameters, or list of messages)
            task: Task to perform ("retrieve_and_generate", "retrieve", "chat", "embed")
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with task results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process different task types
        if task == "retrieve":
            # Extract query from text
            if isinstance(text, dict) and "query" in text:
                query = text["query"]
            elif isinstance(text, str):
                query = text
            else:
                query = str(text)
            
            # Extract parameters
            n_docs = text.get("n_docs", None) if isinstance(text, dict) else kwargs.get("n_docs", None)
            
            # Perform retrieval
            return self.retrieve_documents(query, n_docs=n_docs, **kwargs)
        
        elif task == "retrieve_and_generate":
            # Extract query from text
            if isinstance(text, dict) and "query" in text:
                query = text["query"]
            elif isinstance(text, dict) and "question" in text:
                query = text["question"]
            elif isinstance(text, str):
                query = text
            else:
                query = str(text)
            
            # Extract parameters
            n_docs = text.get("n_docs", None) if isinstance(text, dict) else kwargs.get("n_docs", None)
            
            # Perform retrieval and generation
            return self.retrieve_and_generate(query, n_docs=n_docs, **kwargs)
        
        elif task == "chat":
            # Process messages format
            if isinstance(text, list):
                messages = text
            elif isinstance(text, dict) and "messages" in text:
                messages = text["messages"]
            else:
                # Convert single text to messages format
                messages = [{"role": "user", "content": str(text)}]
            
            # Perform chat
            return self.chat(messages, **kwargs)
        
        elif task == "embed":
            # Process text for embedding
            if isinstance(text, dict) and "text" in text:
                content = text["text"]
            else:
                content = text
            
            # Get embeddings
            return self.embed(content, **kwargs)
        
        else:
            # Default to retrieve and generate
            if isinstance(text, str):
                query = text
            elif isinstance(text, dict) and "query" in text:
                query = text["query"]
            elif isinstance(text, dict) and "question" in text:
                query = text["question"]
            else:
                query = str(text)
            
            return self.retrieve_and_generate(query, **kwargs)

    def __test__(self, **kwargs):
        """
        Run a self-test to verify the model is working correctly.
        
        Returns:
            Dictionary with test results
        """
        if not self.is_initialized:
            try:
                self.initialize()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Initialization failed: {str(e)}",
                    "hardware": self.hardware_info
                }
        
        results = {
            "hardware": self.hardware_info,
            "tests": {}
        }
        
        # Test 1: Retrieve documents
        try:
            test_query = "What is machine learning?"
            retrieval_result = self.retrieve_documents(test_query, n_docs=2)
            
            results["tests"]["document_retrieval"] = {
                "success": len(retrieval_result.get("retrieved_docs", [])) > 0,
                "n_docs_retrieved": len(retrieval_result.get("retrieved_docs", [])),
                "query": test_query
            }
        except Exception as e:
            results["tests"]["document_retrieval"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Retrieve and generate
        try:
            test_query = "What are neural networks?"
            generation_result = self.retrieve_and_generate(test_query, n_docs=2, max_new_tokens=50)
            
            results["tests"]["retrieval_generation"] = {
                "success": "answer" in generation_result and generation_result["answer"],
                "answer_length": len(generation_result.get("answer", "")),
                "n_docs_retrieved": len(generation_result.get("retrieved_docs", [])),
                "query": test_query
            }
        except Exception as e:
            results["tests"]["retrieval_generation"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: Chat functionality
        try:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about deep learning."}
            ]
            chat_result = self.chat(test_messages, max_new_tokens=50)
            
            results["tests"]["chat"] = {
                "success": "response" in chat_result and chat_result["response"],
                "response_length": len(chat_result.get("response", "")),
                "n_docs_retrieved": len(chat_result.get("retrieved_docs", [])),
            }
        except Exception as e:
            results["tests"]["chat"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 4: Embedding
        try:
            test_text = "Test embedding functionality"
            embed_result = self.embed(test_text)
            
            results["tests"]["embedding"] = {
                "success": "embedding" in embed_result,
                "dimensions": embed_result.get("dimensions", 0),
                "text": test_text
            }
        except Exception as e:
            results["tests"]["embedding"] = {
                "success": False,
                "error": str(e)
            }
        
        # Overall success determination
        successful_tests = sum(1 for t in results["tests"].values() if t.get("success", False))
        results["success"] = successful_tests > 0
        results["success_rate"] = successful_tests / len(results["tests"])
        
        return results


class TestRagModel:
    """Test suite for the RAG model implementation."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.model_id = "facebook/rag-token-nq"  # Default test model
        self.low_memory_mode = True  # Use low memory mode for testing
    
    def run_tests(self):
        """Run all tests and return results."""
        results = {}
        
        # Test initialization
        init_result = self.test_initialization()
        results["initialization"] = init_result
        
        # If initialization failed, skip other tests
        if not init_result.get("success", False):
            return results
        
        # Test document retrieval
        results["document_retrieval"] = self.test_document_retrieval()
        
        # Test retrieval and generation
        results["retrieval_generation"] = self.test_retrieval_generation()
        
        # Test chat functionality
        results["chat"] = self.test_chat()
        
        # Test embeddings
        results["embedding"] = self.test_embedding()
        
        # Test low memory mode
        results["low_memory"] = self.test_low_memory_mode()
        
        # Determine overall success
        successful_tests = sum(1 for t in results.values() if t.get("success", False))
        results["overall_success"] = successful_tests / len(results)
        
        return results
    
    def test_initialization(self):
        """Test model initialization."""
        try:
            # Import the model class
            from transformers import RagTokenizer, RagTokenForGeneration
            
            # Initialize the model with minimal config
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Run basic self-test
            test_result = model.__test__()
            
            return {
                "success": test_result.get("success", False),
                "hardware_info": model.hardware_info,
                "details": test_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_document_retrieval(self):
        """Test document retrieval functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Perform document retrieval
            test_query = "What is machine learning?"
            result = model.retrieve_documents(test_query, n_docs=2)
            
            return {
                "success": "retrieved_docs" in result and len(result["retrieved_docs"]) > 0,
                "n_docs": len(result.get("retrieved_docs", [])),
                "query": test_query
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_retrieval_generation(self):
        """Test retrieval and generation functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Perform retrieval and generation
            test_query = "What are neural networks?"
            result = model.retrieve_and_generate(test_query, n_docs=2, max_new_tokens=30)
            
            return {
                "success": "answer" in result and result["answer"],
                "answer_length": len(result.get("answer", "")),
                "n_docs": len(result.get("retrieved_docs", [])),
                "query": test_query
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_chat(self):
        """Test chat functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Test with chat messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about deep learning."}
            ]
            
            result = model.chat(messages, max_new_tokens=30)
            
            return {
                "success": "response" in result and result["response"],
                "response_length": len(result.get("response", "")),
                "n_docs": len(result.get("retrieved_docs", [])),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_embedding(self):
        """Test embedding functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Generate embeddings
            test_text = "Test embedding functionality"
            result = model.embed(test_text)
            
            return {
                "success": "embedding" in result and isinstance(result["embedding"], list),
                "dimensions": result.get("dimensions", 0),
                "text": test_text
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_low_memory_mode(self):
        """Test low memory mode functionality."""
        try:
            # Initialize the model with low_memory_mode
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=True
            )
            
            # Check if initialization succeeded
            is_initialized = model.is_initialized
            
            # Get model parameters to check if low_memory settings were applied
            params = model.get_parameters()
            
            return {
                "success": is_initialized,
                "details": {
                    "hardware_info": model.hardware_info,
                    "parameters": params
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    # Run tests if executed directly
    tester = TestRagModel()
    results = tester.run_tests()
    
    print(json.dumps(results, indent=2))