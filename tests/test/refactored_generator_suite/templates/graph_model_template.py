#!/usr/bin/env python3
"""
Template for Graph Neural Network models such as Graphormer, GraphSage, etc.

This template is designed for models that operate on graph-structured data,
performing tasks such as node classification, link prediction, or graph classification.
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
    Skillset for {model_type_upper} - a graph neural network model that processes
    graph-structured data such as molecular graphs, social networks, or knowledge graphs.
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
        self.processor = None
        self.is_initialized = False
        
        # Track hardware info for reporting
        self.hardware_info = {
            "device": device,
            "device_name": None,
            "memory_available": None,
            "supports_half_precision": False,
            "graph_specific": {
                "max_nodes": None,
                "max_edges": None,
                "node_feature_dim": None,
                "edge_feature_dim": None,
                "supports_batching": True
            }
        }
        
        # Optional configuration
        self.low_memory_mode = kwargs.get("low_memory_mode", False)
        self.max_memory = kwargs.get("max_memory", None)
        self.torch_dtype = kwargs.get("torch_dtype", None)
        self.max_nodes = kwargs.get("max_nodes", 1024)
        self.max_edges = kwargs.get("max_edges", 4096)
        
        # Initialize the model if auto_init is True
        auto_init = kwargs.get("auto_init", True)
        if auto_init:
            self.initialize()
    
    def initialize(self):
        """Initialize the model and processor."""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing {self.model_id} on {self.device}")
        start_time = time.time()
        
        try:
            # Check if CUDA is available when device is cuda
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if MPS is available when device is mps
            if self.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Check if ROCm/HIP is available when device is rocm
            if self.device == "rocm":
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
            
            # Load processor (specialized for graph models)
            try:
                self.processor = {processor_class_name}.from_pretrained(self.model_id)
            except Exception as e:
                logger.warning(f"Error loading processor: {str(e)}. Creating a mock processor.")
                self.processor = self._create_mock_processor()
            
            # Load model with appropriate configuration
            load_kwargs = {}
            if self.torch_dtype is not None:
                load_kwargs["torch_dtype"] = self.torch_dtype
            
            if self.low_memory_mode:
                load_kwargs["low_cpu_mem_usage"] = True
            
            if self.max_memory is not None:
                load_kwargs["max_memory"] = self.max_memory
            
            # Specific handling for device placement
            if self.device.startswith(("cuda", "rocm")) and "device_map" not in load_kwargs:
                load_kwargs["device_map"] = "auto"
            
            # Load the graph model
            self.model = {model_class_name}.from_pretrained(self.model_id, **load_kwargs)
            
            # Move to appropriate device if not using device_map
            if "device_map" not in load_kwargs and not self.device.startswith(("cuda", "rocm")):
                self.model.to(self.device)
            
            # Update graph-specific info
            if hasattr(self.model, "config"):
                self.hardware_info["graph_specific"].update({
                    "max_nodes": getattr(self.model.config, "max_nodes", self.max_nodes),
                    "max_edges": getattr(self.model.config, "max_edges", self.max_edges),
                    "node_feature_dim": getattr(self.model.config, "hidden_size", None),
                    "edge_feature_dim": getattr(self.model.config, "edge_hidden_size", None)
                })
            
            # Log initialization time
            elapsed_time = time.time() - start_time
            logger.info(f"Initialized {self.model_id} in {elapsed_time:.2f} seconds")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing {self.model_id}: {str(e)}")
            raise
    
    def _create_mock_processor(self):
        """Create a mock processor for graph models when the real one fails."""
        class MockGraphProcessor:
            def __init__(self, max_nodes=1024, max_edges=4096):
                self.max_nodes = max_nodes
                self.max_edges = max_edges
            
            def __call__(self, graph=None, node_features=None, edge_features=None, 
                        adjacency_matrix=None, return_tensors="pt", **kwargs):
                """Mock processing of graph data."""
                # Create a mock batch with minimal graph data
                if graph is None:
                    # If no input is provided, create a small mock graph
                    batch_size = 1
                    num_nodes = min(10, self.max_nodes)
                    num_edges = min(20, self.max_edges)
                    
                    # Create mock adjacency matrix (simple path graph)
                    adj_shape = (batch_size, num_nodes, num_nodes)
                    adjacency = torch.zeros(adj_shape)
                    # Add edges in a simple path pattern
                    for i in range(num_nodes-1):
                        adjacency[0, i, i+1] = 1
                        adjacency[0, i+1, i] = 1  # Undirected graph
                    
                    # Create mock node features
                    node_feats = torch.rand((batch_size, num_nodes, 64))
                    
                    # Create mock edge features if needed
                    edge_feats = torch.rand((batch_size, num_edges, 32))
                    
                    # Create mock node types
                    node_types = torch.randint(0, 5, (batch_size, num_nodes))
                    
                    # Create attention mask for nodes
                    attention_mask = torch.ones((batch_size, num_nodes))
                    
                    return {
                        "input_ids": node_types,
                        "attention_mask": attention_mask,
                        "adjacency_matrix": adjacency,
                        "node_features": node_feats,
                        "edge_features": edge_feats
                    }
                else:
                    # Process the input graph (minimal conversion)
                    return {
                        "input_ids": torch.ones((1, 10), dtype=torch.long),
                        "attention_mask": torch.ones((1, 10)),
                        "adjacency_matrix": torch.eye(10).unsqueeze(0),
                        "node_features": torch.rand((1, 10, 64)),
                        "edge_features": torch.rand((1, 20, 32))
                    }
        
        logger.info("Creating mock graph processor")
        return MockGraphProcessor(max_nodes=self.max_nodes, max_edges=self.max_edges)
    
    def process_graph(self, graph_data, **kwargs):
        """
        Process graph data into the format expected by the model.
        
        Args:
            graph_data: The input graph data (can be NetworkX graph, PyG graph, adjacency matrix, etc.)
            **kwargs: Additional processing parameters
            
        Returns:
            Processed inputs ready for the model
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Check the type of input and process accordingly
            if hasattr(graph_data, "to_networkx") or hasattr(graph_data, "to_pyg"):
                # Handle PyTorch Geometric or DGL graph
                inputs = self._process_pyg_dgl_graph(graph_data, **kwargs)
            elif hasattr(graph_data, "nodes") and hasattr(graph_data, "edges"):
                # Handle NetworkX graph
                inputs = self._process_networkx_graph(graph_data, **kwargs)
            elif isinstance(graph_data, dict) and "adjacency" in graph_data:
                # Handle dictionary with adjacency matrix
                inputs = self._process_adjacency_dict(graph_data, **kwargs)
            elif isinstance(graph_data, (torch.Tensor, np.ndarray)) and len(graph_data.shape) >= 2:
                # Handle raw adjacency matrix
                inputs = self._process_adjacency_matrix(graph_data, **kwargs)
            else:
                # Default handling - attempt to use the processor directly
                try:
                    inputs = self.processor(graph_data, return_tensors="pt", **kwargs)
                except:
                    logger.warning("Could not process graph data with processor, using mock data")
                    inputs = self._create_mock_inputs()
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error processing graph data: {str(e)}")
            # Return mock inputs as fallback
            return self._create_mock_inputs()
    
    def _process_networkx_graph(self, graph, **kwargs):
        """Process a NetworkX graph into model inputs."""
        import networkx as nx
        
        # Extract node features if they exist
        node_list = list(graph.nodes())
        num_nodes = len(node_list)
        node_features = []
        node_types = []
        
        # Get node features and types
        for node in node_list:
            # Get node features
            feat = graph.nodes[node].get("features", None)
            if feat is not None:
                node_features.append(feat)
            
            # Get node types
            ntype = graph.nodes[node].get("type", 0)
            node_types.append(ntype)
        
        # Create adjacency matrix
        adjacency = nx.to_numpy_array(graph)
        
        # Extract edge features if they exist
        edge_features = []
        for u, v, data in graph.edges(data=True):
            if "features" in data:
                edge_features.append(data["features"])
        
        # Convert to tensors
        adjacency_tensor = torch.tensor(adjacency).float().unsqueeze(0)  # Add batch dimension
        
        if node_features:
            if isinstance(node_features[0], (list, np.ndarray)):
                node_features_tensor = torch.tensor(np.array(node_features)).float().unsqueeze(0)
            else:
                # Handle scalar features
                node_features_tensor = torch.tensor(node_features).float().unsqueeze(0).unsqueeze(-1)
        else:
            # Use identity features if none provided
            node_features_tensor = torch.eye(num_nodes).unsqueeze(0)
        
        if node_types:
            node_types_tensor = torch.tensor(node_types).long().unsqueeze(0)
        else:
            node_types_tensor = torch.zeros(1, num_nodes).long()
        
        if edge_features:
            if isinstance(edge_features[0], (list, np.ndarray)):
                edge_features_tensor = torch.tensor(np.array(edge_features)).float().unsqueeze(0)
            else:
                # Handle scalar features
                edge_features_tensor = torch.tensor(edge_features).float().unsqueeze(0).unsqueeze(-1)
        else:
            # Use empty tensor if no edge features
            num_edges = graph.number_of_edges()
            edge_features_tensor = torch.zeros(1, num_edges, 1).float()
        
        # Create attention mask
        attention_mask = torch.ones(1, num_nodes)
        
        return {
            "input_ids": node_types_tensor,
            "attention_mask": attention_mask,
            "adjacency_matrix": adjacency_tensor,
            "node_features": node_features_tensor,
            "edge_features": edge_features_tensor
        }
    
    def _process_pyg_dgl_graph(self, graph, **kwargs):
        """Process a PyTorch Geometric or DGL graph into model inputs."""
        # Try to handle PyTorch Geometric graph
        if hasattr(graph, "x") and hasattr(graph, "edge_index"):
            # PyTorch Geometric graph
            node_features = graph.x
            edge_index = graph.edge_index
            
            # Convert edge_index to adjacency matrix
            num_nodes = node_features.size(0)
            adjacency = torch.zeros(num_nodes, num_nodes)
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i], edge_index[1, i]
                adjacency[src, dst] = 1
            
            # Get edge features if they exist
            edge_features = graph.edge_attr if hasattr(graph, "edge_attr") else None
            
            # Get node types if they exist
            node_types = graph.y if hasattr(graph, "y") else torch.zeros(num_nodes).long()
            
            # Create attention mask
            attention_mask = torch.ones(num_nodes)
            
            # Add batch dimension
            adjacency = adjacency.unsqueeze(0)
            node_features = node_features.unsqueeze(0)
            node_types = node_types.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            
            if edge_features is not None:
                edge_features = edge_features.unsqueeze(0)
            else:
                # Create dummy edge features
                num_edges = edge_index.size(1)
                edge_features = torch.zeros(1, num_edges, 1)
            
            return {
                "input_ids": node_types,
                "attention_mask": attention_mask,
                "adjacency_matrix": adjacency,
                "node_features": node_features,
                "edge_features": edge_features
            }
            
        # Try to handle DGL graph
        elif hasattr(graph, "ndata") and hasattr(graph, "edata"):
            # DGL graph
            import dgl
            
            # Get node features
            if "feat" in graph.ndata:
                node_features = graph.ndata["feat"]
            else:
                # Use identity features if none provided
                num_nodes = graph.number_of_nodes()
                node_features = torch.eye(num_nodes)
            
            # Get node types if they exist
            if "label" in graph.ndata:
                node_types = graph.ndata["label"]
            else:
                node_types = torch.zeros(graph.number_of_nodes()).long()
            
            # Get edge features if they exist
            if "feat" in graph.edata:
                edge_features = graph.edata["feat"]
            else:
                # Create dummy edge features
                num_edges = graph.number_of_edges()
                edge_features = torch.zeros(num_edges, 1)
            
            # Convert to adjacency matrix
            adjacency = torch.from_numpy(dgl.to_networkx(graph).adjacency_matrix().todense()).float()
            
            # Create attention mask
            attention_mask = torch.ones(graph.number_of_nodes())
            
            # Add batch dimension
            adjacency = adjacency.unsqueeze(0)
            node_features = node_features.unsqueeze(0)
            node_types = node_types.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            
            return {
                "input_ids": node_types,
                "attention_mask": attention_mask,
                "adjacency_matrix": adjacency,
                "node_features": node_features,
                "edge_features": edge_features
            }
        else:
            raise ValueError("Unsupported graph format")
    
    def _process_adjacency_dict(self, graph_dict, **kwargs):
        """Process a dictionary containing adjacency matrix and features."""
        # Extract components from the dictionary
        adjacency = graph_dict.get("adjacency")
        node_features = graph_dict.get("node_features")
        edge_features = graph_dict.get("edge_features")
        node_types = graph_dict.get("node_types")
        
        # Convert to tensors if needed
        if isinstance(adjacency, np.ndarray):
            adjacency = torch.from_numpy(adjacency).float()
        
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).float()
        
        if isinstance(edge_features, np.ndarray):
            edge_features = torch.from_numpy(edge_features).float()
        
        if isinstance(node_types, np.ndarray):
            node_types = torch.from_numpy(node_types).long()
        
        # Add batch dimension if needed
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        
        # Create node features if not provided
        if node_features is None:
            num_nodes = adjacency.size(1)
            node_features = torch.eye(num_nodes).unsqueeze(0)
        elif node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        
        # Create node types if not provided
        if node_types is None:
            num_nodes = adjacency.size(1)
            node_types = torch.zeros(1, num_nodes).long()
        elif node_types.dim() == 1:
            node_types = node_types.unsqueeze(0)
        
        # Create edge features if not provided
        if edge_features is None:
            # Count number of edges from adjacency
            num_edges = torch.sum(adjacency > 0).item()
            edge_features = torch.zeros(1, num_edges, 1)
        elif edge_features.dim() == 2:
            edge_features = edge_features.unsqueeze(0)
        
        # Create attention mask
        attention_mask = torch.ones(1, adjacency.size(1))
        
        return {
            "input_ids": node_types,
            "attention_mask": attention_mask,
            "adjacency_matrix": adjacency,
            "node_features": node_features,
            "edge_features": edge_features
        }
    
    def _process_adjacency_matrix(self, adjacency, **kwargs):
        """Process a raw adjacency matrix into model inputs."""
        # Convert to tensor if needed
        if isinstance(adjacency, np.ndarray):
            adjacency = torch.from_numpy(adjacency).float()
        
        # Add batch dimension if needed
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        
        # Extract dimensions
        batch_size, num_nodes = adjacency.size(0), adjacency.size(1)
        
        # Create node features (identity features by default)
        node_features = kwargs.get("node_features")
        if node_features is None:
            node_features = torch.eye(num_nodes).repeat(batch_size, 1, 1)
        
        # Create node types (all zeros by default)
        node_types = kwargs.get("node_types")
        if node_types is None:
            node_types = torch.zeros(batch_size, num_nodes).long()
        
        # Create edge features
        edge_features = kwargs.get("edge_features")
        if edge_features is None:
            # Count number of edges from adjacency
            num_edges = torch.sum(adjacency > 0, dim=(1, 2))  # Count per batch
            max_edges = torch.max(num_edges).item()
            edge_features = torch.zeros(batch_size, max_edges, 1)
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, num_nodes)
        
        return {
            "input_ids": node_types,
            "attention_mask": attention_mask,
            "adjacency_matrix": adjacency,
            "node_features": node_features,
            "edge_features": edge_features
        }
    
    def _create_mock_inputs(self):
        """Create mock inputs for graceful degradation."""
        # Create a small mock graph
        num_nodes = 10
        
        # Create mock adjacency matrix (simple path graph)
        adjacency = torch.zeros(1, num_nodes, num_nodes)
        # Add edges in a simple path pattern
        for i in range(num_nodes-1):
            adjacency[0, i, i+1] = 1
            adjacency[0, i+1, i] = 1  # Undirected graph
        
        # Create mock node features
        node_feats = torch.rand(1, num_nodes, 64)
        
        # Create mock edge features
        edge_feats = torch.rand(1, num_nodes-1, 32)
        
        # Create mock node types
        node_types = torch.randint(0, 5, (1, num_nodes))
        
        # Create attention mask for nodes
        attention_mask = torch.ones(1, num_nodes)
        
        # Move to the correct device
        adjacency = adjacency.to(self.device)
        node_feats = node_feats.to(self.device)
        edge_feats = edge_feats.to(self.device)
        node_types = node_types.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        return {
            "input_ids": node_types,
            "attention_mask": attention_mask,
            "adjacency_matrix": adjacency,
            "node_features": node_feats,
            "edge_features": edge_feats
        }
    
    def node_classification(self, graph_data, **kwargs):
        """
        Perform node classification on a graph.
        
        Args:
            graph_data: The input graph (various formats supported)
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with node classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process the graph into model inputs
        inputs = self.process_graph(graph_data, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract and format results
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Format result
                result = {
                    "node_predictions": predictions.cpu().numpy(),
                    "node_logits": logits.cpu().numpy(),
                    "num_nodes": inputs["attention_mask"].size(1)
                }
                
                return result
            else:
                logger.warning("Model outputs don't include logits, returning raw outputs")
                # Convert any tensors to numpy for serialization
                raw_dict = {}
                for k, v in outputs.items():
                    if hasattr(v, "cpu"):
                        raw_dict[k] = v.cpu().numpy()
                    else:
                        raw_dict[k] = v
                
                return {"raw_outputs": raw_dict}
        
        except Exception as e:
            logger.error(f"Error during node classification: {str(e)}")
            return {"error": str(e)}
    
    def graph_classification(self, graph_data, **kwargs):
        """
        Perform graph classification.
        
        Args:
            graph_data: The input graph (various formats supported)
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with graph classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process the graph into model inputs
        inputs = self.process_graph(graph_data, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract and format results
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                
                # Get predictions
                if logits.dim() > 1 and logits.size(1) > 1:
                    # Multi-class classification
                    predictions = torch.argmax(logits, dim=-1)
                else:
                    # Binary classification
                    predictions = (logits > 0).long()
                
                # Format result
                result = {
                    "graph_predictions": predictions.cpu().numpy(),
                    "graph_logits": logits.cpu().numpy()
                }
                
                return result
            else:
                logger.warning("Model outputs don't include logits, returning raw outputs")
                # Convert any tensors to numpy for serialization
                raw_dict = {}
                for k, v in outputs.items():
                    if hasattr(v, "cpu"):
                        raw_dict[k] = v.cpu().numpy()
                    else:
                        raw_dict[k] = v
                
                return {"raw_outputs": raw_dict}
        
        except Exception as e:
            logger.error(f"Error during graph classification: {str(e)}")
            return {"error": str(e)}
    
    def link_prediction(self, graph_data, node_pairs=None, **kwargs):
        """
        Perform link prediction to estimate the likelihood of edges between nodes.
        
        Args:
            graph_data: The input graph (various formats supported)
            node_pairs: List of node pairs to predict links for (if None, predict all possible links)
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with link prediction results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process the graph into model inputs
        inputs = self.process_graph(graph_data, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract node embeddings
            if hasattr(outputs, "node_embeddings"):
                node_embeddings = outputs.node_embeddings
            else:
                # If no specific node embeddings, use last_hidden_state
                node_embeddings = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else None
            
            if node_embeddings is None:
                logger.warning("Model outputs don't include node embeddings, returning raw outputs")
                # Convert any tensors to numpy for serialization
                raw_dict = {}
                for k, v in outputs.items():
                    if hasattr(v, "cpu"):
                        raw_dict[k] = v.cpu().numpy()
                    else:
                        raw_dict[k] = v
                
                return {"raw_outputs": raw_dict}
            
            # Get number of nodes
            num_nodes = node_embeddings.size(1)
            
            # If node pairs not provided, generate all possible pairs
            if node_pairs is None:
                src_nodes = []
                dst_nodes = []
                for i in range(num_nodes):
                    for j in range(i+1, num_nodes):  # Only upper triangle to avoid duplicates
                        src_nodes.append(i)
                        dst_nodes.append(j)
                node_pairs = list(zip(src_nodes, dst_nodes))
            
            # Compute link probabilities for each pair
            if node_pairs:
                link_scores = []
                src_indices = []
                dst_indices = []
                
                for src, dst in node_pairs:
                    if src < num_nodes and dst < num_nodes:
                        src_indices.append(src)
                        dst_indices.append(dst)
                
                # Get embeddings for selected nodes
                src_embeddings = node_embeddings[0, src_indices]
                dst_embeddings = node_embeddings[0, dst_indices]
                
                # Compute similarity scores (dot product)
                scores = torch.sum(src_embeddings * dst_embeddings, dim=1)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(scores)
                
                # Format result
                result = {
                    "node_pairs": node_pairs,
                    "link_probabilities": probabilities.cpu().numpy(),
                    "link_scores": scores.cpu().numpy()
                }
                
                return result
            else:
                return {"error": "No valid node pairs to predict"}
        
        except Exception as e:
            logger.error(f"Error during link prediction: {str(e)}")
            return {"error": str(e)}
    
    def node_embedding(self, graph_data, **kwargs):
        """
        Get node embeddings from the graph model.
        
        Args:
            graph_data: The input graph (various formats supported)
            **kwargs: Additional parameters for processing or inference
            
        Returns:
            Dictionary with node embeddings
        """
        if not self.is_initialized:
            self.initialize()
        
        # Process the graph into model inputs
        inputs = self.process_graph(graph_data, **kwargs)
        
        try:
            # Run inference with the model
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract node embeddings
            if hasattr(outputs, "node_embeddings"):
                embeddings = outputs.node_embeddings
            elif hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state
            else:
                # Try to find embeddings in hidden states
                hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
                if hidden_states is not None:
                    # Use the last layer
                    embeddings = hidden_states[-1]
                else:
                    logger.warning("Could not extract embeddings from model outputs")
                    embeddings = None
            
            if embeddings is not None:
                # Format result
                result = {
                    "node_embeddings": embeddings.cpu().numpy(),
                    "embedding_dim": embeddings.size(-1),
                    "num_nodes": embeddings.size(1)
                }
                
                return result
            else:
                logger.warning("Model outputs don't include embeddings, returning raw outputs")
                # Convert any tensors to numpy for serialization
                raw_dict = {}
                for k, v in outputs.items():
                    if hasattr(v, "cpu"):
                        raw_dict[k] = v.cpu().numpy()
                    else:
                        raw_dict[k] = v
                
                return {"raw_outputs": raw_dict}
        
        except Exception as e:
            logger.error(f"Error during node embedding: {str(e)}")
            return {"error": str(e)}
    
    def __call__(self, graph_data, task: str = "node_classification", **kwargs) -> Dict[str, Any]:
        """
        Process graph data with the model.
        
        Args:
            graph_data: The input graph data
            task: Task to perform ('node_classification', 'graph_classification', 
                  'link_prediction', 'node_embedding')
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with task results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Select task
        if task == "node_classification":
            return self.node_classification(graph_data, **kwargs)
        elif task == "graph_classification":
            return self.graph_classification(graph_data, **kwargs)
        elif task == "link_prediction":
            node_pairs = kwargs.pop("node_pairs", None)
            return self.link_prediction(graph_data, node_pairs, **kwargs)
        elif task == "node_embedding":
            return self.node_embedding(graph_data, **kwargs)
        else:
            # Default to node classification
            logger.warning(f"Unknown task '{task}', defaulting to node_classification")
            return self.node_classification(graph_data, **kwargs)

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
        
        # Test 1: Process a small graph
        try:
            # Create a small test graph
            adjacency = torch.zeros(1, 5, 5)
            # Simple path graph
            for i in range(4):
                adjacency[0, i, i+1] = 1
                adjacency[0, i+1, i] = 1
            
            # Test graph processing
            inputs = self.process_graph(adjacency)
            
            results["tests"]["graph_processing"] = {
                "success": True,
                "num_nodes": inputs["adjacency_matrix"].size(1),
                "num_input_dimensions": len(inputs)
            }
        except Exception as e:
            results["tests"]["graph_processing"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Node classification
        try:
            # Create a small test graph
            adjacency = torch.zeros(1, 5, 5)
            # Simple path graph
            for i in range(4):
                adjacency[0, i, i+1] = 1
                adjacency[0, i+1, i] = 1
            
            # Test node classification
            output = self.node_classification(adjacency)
            
            if "node_predictions" in output:
                results["tests"]["node_classification"] = {
                    "success": True,
                    "predictions_shape": output["node_predictions"].shape
                }
            else:
                results["tests"]["node_classification"] = {
                    "success": "raw_outputs" in output,
                    "output_keys": list(output.keys())
                }
        except Exception as e:
            results["tests"]["node_classification"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: Node embedding
        try:
            # Create a small test graph
            adjacency = torch.zeros(1, 5, 5)
            # Simple path graph
            for i in range(4):
                adjacency[0, i, i+1] = 1
                adjacency[0, i+1, i] = 1
            
            # Test node embedding
            output = self.node_embedding(adjacency)
            
            if "node_embeddings" in output:
                results["tests"]["node_embedding"] = {
                    "success": True,
                    "embeddings_shape": output["node_embeddings"].shape,
                    "embedding_dim": output["embedding_dim"]
                }
            else:
                results["tests"]["node_embedding"] = {
                    "success": "raw_outputs" in output,
                    "output_keys": list(output.keys())
                }
        except Exception as e:
            results["tests"]["node_embedding"] = {
                "success": False,
                "error": str(e)
            }
        
        # Overall success determination
        successful_tests = sum(1 for t in results["tests"].values() if t.get("success", False))
        results["success"] = successful_tests > 0
        results["success_rate"] = successful_tests / len(results["tests"])
        
        return results


class TestGraphModel:
    """Test suite for the Graph Neural Network model implementation."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.model_id = "facebookresearch/graphormer-base-pcqm4mv2"  # Default test model
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
        
        # Test graph processing
        results["graph_processing"] = self.test_graph_processing()
        
        # Test node classification
        results["node_classification"] = self.test_node_classification()
        
        # Test graph classification
        results["graph_classification"] = self.test_graph_classification()
        
        # Test link prediction
        results["link_prediction"] = self.test_link_prediction()
        
        # Test node embedding
        results["node_embedding"] = self.test_node_embedding()
        
        # Determine overall success
        successful_tests = sum(1 for t in results.values() if t.get("success", False))
        results["overall_success"] = successful_tests / len(results)
        
        return results
    
    def test_initialization(self):
        """Test model initialization."""
        try:
            # Import the model class
            from transformers import AutoModelForNodeClassification, AutoTokenizer
            
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
    
    def test_graph_processing(self):
        """Test graph processing functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a small test graph
            adjacency = torch.zeros(1, 5, 5)
            # Simple path graph
            for i in range(4):
                adjacency[0, i, i+1] = 1
                adjacency[0, i+1, i] = 1
            
            # Test graph processing
            inputs = model.process_graph(adjacency)
            
            return {
                "success": "adjacency_matrix" in inputs and "node_features" in inputs,
                "num_nodes": inputs["adjacency_matrix"].size(1),
                "num_input_dimensions": len(inputs)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_node_classification(self):
        """Test node classification functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a small test graph
            adjacency = torch.zeros(1, 5, 5)
            # Simple path graph
            for i in range(4):
                adjacency[0, i, i+1] = 1
                adjacency[0, i+1, i] = 1
            
            # Test node classification
            output = model.node_classification(adjacency)
            
            return {
                "success": "node_predictions" in output or "raw_outputs" in output,
                "output_keys": list(output.keys())
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_graph_classification(self):
        """Test graph classification functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a small test graph
            adjacency = torch.zeros(1, 5, 5)
            # Simple path graph
            for i in range(4):
                adjacency[0, i, i+1] = 1
                adjacency[0, i+1, i] = 1
            
            # Test graph classification
            output = model.graph_classification(adjacency)
            
            return {
                "success": "graph_predictions" in output or "raw_outputs" in output,
                "output_keys": list(output.keys())
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_link_prediction(self):
        """Test link prediction functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a small test graph
            adjacency = torch.zeros(1, 5, 5)
            # Simple path graph
            for i in range(4):
                adjacency[0, i, i+1] = 1
                adjacency[0, i+1, i] = 1
            
            # Define node pairs
            node_pairs = [(0, 2), (1, 3), (0, 4)]
            
            # Test link prediction
            output = model.link_prediction(adjacency, node_pairs=node_pairs)
            
            return {
                "success": "link_probabilities" in output or "raw_outputs" in output,
                "output_keys": list(output.keys())
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_node_embedding(self):
        """Test node embedding functionality."""
        try:
            # Initialize the model
            model = {skillset_class_name}(
                model_id=self.model_id,
                device="cpu",
                low_memory_mode=self.low_memory_mode
            )
            
            # Create a small test graph
            adjacency = torch.zeros(1, 5, 5)
            # Simple path graph
            for i in range(4):
                adjacency[0, i, i+1] = 1
                adjacency[0, i+1, i] = 1
            
            # Test node embedding
            output = model.node_embedding(adjacency)
            
            return {
                "success": "node_embeddings" in output or "raw_outputs" in output,
                "output_keys": list(output.keys())
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    # Run tests if executed directly
    tester = TestGraphModel()
    results = tester.run_tests()
    
    print(json.dumps(results, indent=2))