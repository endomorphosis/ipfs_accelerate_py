"""
Embedding model template that inherits from the base template.
This template provides specialized functionality for embedding models like BERT.
"""

# Template metadata
TEMPLATE_VERSION = "1.0.0"
TEMPLATE_DESCRIPTION = "Template for embedding models like BERT, RoBERTa, etc."
INHERITS_FROM = "hf_template.py"  # Inherits from base template
SUPPORTS_HARDWARE = ["cpu", "cuda", "rocm", "mps"]  # Embedding models support all hardware
COMPATIBLE_FAMILIES = ["embedding"]  # Only compatible with embedding models
MODEL_REQUIREMENTS = {"embedding_dim": [128, 384, 768, 1024]}  # Common embedding dimensions

# Override sections for embedding models
SECTION_CLASS_DEFINITION = """class hf_{{ model_name }}:
    """{{ model_description }}"""
    
    # Model metadata
    MODEL_TYPE = "{{ model_type }}"
    MODEL_NAME = "{{ model_name }}"
    MODALITY = "{{ modality }}"
    SUPPORTS_QUANTIZATION = {{ supports_quantization }}
    REQUIRES_GPU = {{ requires_gpu }}
    
    # Embedding-specific metadata
    POOLING_STRATEGY = "mean"  # Default pooling strategy"""

SECTION_METHODS = """    def encode(self, text, **kwargs):
        """
        Encode text input to embeddings.
        
        Args:
            text: The text to encode (string or list of strings)
            **kwargs: Additional keyword arguments for encoding
            
        Returns:
            Embeddings for the input text
        """
        self._ensure_model_loaded()
        
        # Format input
        input_texts = self._format_inputs(text)
        
        # Tokenize the input
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, **kwargs)
        
        # Apply pooling strategy
        embeddings = self._pool_embeddings(outputs, inputs)
        
        return embeddings.cpu().numpy()
    
    def similarity(self, texts1, texts2):
        """
        Calculate similarity between two sets of texts.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            
        Returns:
            Similarity scores between text pairs
        """
        # Get embeddings for both sets
        embeddings1 = self.encode(texts1)
        embeddings2 = self.encode(texts2)
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        normalized_embeddings1 = embeddings1 / norm1
        normalized_embeddings2 = embeddings2 / norm2
        
        # Calculate cosine similarity
        if len(normalized_embeddings1) == 1 and len(normalized_embeddings2) > 1:
            # Compare one text to many
            similarities = np.dot(normalized_embeddings2, normalized_embeddings1.T).flatten()
        elif len(normalized_embeddings2) == 1 and len(normalized_embeddings1) > 1:
            # Compare many texts to one
            similarities = np.dot(normalized_embeddings1, normalized_embeddings2.T).flatten()
        elif len(normalized_embeddings1) == len(normalized_embeddings2):
            # Compare pairs of texts
            similarities = np.sum(normalized_embeddings1 * normalized_embeddings2, axis=1)
        else:
            # Compare all pairs
            similarities = np.dot(normalized_embeddings1, normalized_embeddings2.T)
        
        return similarities"""

SECTION_UTILITY_METHODS = """    def _load_model(self, **kwargs):
        """Load the model and tokenizer with embedding-specific settings"""
        try:
            # Try to use resource pool if available
            if RESOURCE_POOL_AVAILABLE:
                pool = get_global_resource_pool()
                
                # Define model constructor
                def create_model():
                    # For embedding models, we typically use AutoModel
                    from transformers import AutoModel, AutoConfig
                    
                    config = AutoConfig.from_pretrained(self.model_id)
                    model = AutoModel.from_pretrained(self.model_id, config=config, **kwargs)
                    model.to(self.device)
                    model.eval()
                    return model
                
                # Define tokenizer constructor
                def create_tokenizer():
                    from transformers import AutoTokenizer
                    return AutoTokenizer.from_pretrained(self.model_id)
                
                # Get or create model from pool
                self.model = pool.get_model(
                    model_type=self.MODEL_TYPE,
                    model_name=self.model_id,
                    constructor=create_model
                )
                
                # Get or create tokenizer from pool
                self.tokenizer = pool.get_tokenizer(
                    model_type=self.MODEL_TYPE,
                    model_name=self.model_id,
                    constructor=create_tokenizer
                )
            else:
                # Load directly without resource pooling
                from transformers import AutoModel, AutoConfig, AutoTokenizer
                
                self.config = AutoConfig.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id, config=self.config, **kwargs)
                self.model.to(self.device)
                self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            self.is_initialized = True
            self.logger.info(f"Model {self.model_id} loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_id}: {str(e)}")
            raise
    
    def _pool_embeddings(self, outputs, inputs):
        """
        Pool token embeddings to sentence embeddings.
        
        Args:
            outputs: Model outputs
            inputs: Model inputs with attention mask
            
        Returns:
            Pooled embeddings
        """
        # Extract embeddings from the model outputs
        if hasattr(outputs, "last_hidden_state"):
            # Apply pooling based on strategy
            if self.POOLING_STRATEGY == "mean":
                # Use mean pooling for embedding
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    # Apply attention mask for proper mean pooling
                    last_hidden = outputs.last_hidden_state
                    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    embeddings = torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                else:
                    # Fall back to simple mean
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            elif self.POOLING_STRATEGY == "cls":
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0]
            elif self.POOLING_STRATEGY == "max":
                # Use max pooling
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    # Apply attention mask for proper max pooling
                    last_hidden = outputs.last_hidden_state
                    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())
                    last_hidden[~mask.bool()] = -torch.inf  # Mask out padding tokens
                    embeddings = torch.max(last_hidden, dim=1)[0]
                else:
                    # Fall back to simple max
                    embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
            else:
                # Fall back to mean pooling
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        else:
            # Fall back to pooler output if available
            embeddings = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
        
        return embeddings
    
    def _format_inputs(self, inputs):
        """Format inputs for embedding models"""
        if isinstance(inputs, str):
            # Single text input
            return [inputs]
        elif isinstance(inputs, list):
            # List of text inputs
            return inputs
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")"""

SECTION_MAIN = """def main():
    """Example usage for embedding model"""
    import argparse
    import numpy as np
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f"Example usage of {hf_{{ model_name }}.__name__}")
    parser.add_argument("--model", type=str, default="{{ model_name }}", help="Model ID to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--text", type=str, default="Hello, world\!", help="Text to encode")
    parser.add_argument("--compare", type=str, default=None, help="Compare similarity with this text")
    args = parser.parse_args()
    
    # Create model instance
    model = hf_{{ model_name }}(model_id=args.model, device=args.device)
    
    # Encode text
    embeddings = model.encode(args.text)
    
    # Print results
    print(f"Text: {args.text}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding sample: {embeddings[0, :5]}")
    
    # Compare similarity if requested
    if args.compare:
        similarity = model.similarity(args.text, args.compare)[0]
        print(f"Comparison text: {args.compare}")
        print(f"Similarity: {similarity:.4f}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())"""
