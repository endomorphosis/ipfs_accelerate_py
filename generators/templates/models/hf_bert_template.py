"""
BERT-specific template that inherits from the embedding template.
This template provides specialized functionality for BERT models.
"""

# Template metadata
TEMPLATE_VERSION = "1.0.0"
TEMPLATE_DESCRIPTION = "Template for BERT models with specialized functionality"
INHERITS_FROM = "hf_embedding_template.py"  # Inherits from embedding template
SUPPORTS_HARDWARE = ["cpu", "cuda", "rocm", "mps", "openvino"]  # BERT supports all hardware
COMPATIBLE_FAMILIES = ["embedding"]  # Only compatible with embedding models
MODEL_REQUIREMENTS = {"embedding_dim": [768, 1024]}  # BERT embedding dimensions

# Override sections for BERT models
SECTION_CLASS_DEFINITION = """class hf_{{ model_name }}:
    """{{ model_description }}"""
    
    # Model metadata
    MODEL_TYPE = "bert"
    MODEL_NAME = "{{ model_name }}"
    MODALITY = "{{ modality }}"
    SUPPORTS_QUANTIZATION = {{ supports_quantization }}
    REQUIRES_GPU = {{ requires_gpu }}
    
    # BERT-specific metadata
    POOLING_STRATEGY = "cls"  # Default to CLS token for BERT
    SUPPORTS_MASKED_LM = True
    SUPPORTS_NSP = True  # Next Sentence Prediction"""

SECTION_METHODS = """    def encode(self, text, **kwargs):
        """
        Encode text input to embeddings using BERT model.
        
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
            return_tensors="pt",
            max_length=512  # BERT has a max sequence length of 512
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, **kwargs)
        
        # Apply pooling strategy (BERT typically uses CLS token)
        embeddings = self._pool_embeddings(outputs, inputs)
        
        return embeddings.cpu().numpy()
    
    def predict_masked_tokens(self, text, mask_token="[MASK]"):
        """
        Predict masked tokens in the input text.
        
        Args:
            text: Text with masked tokens (e.g., "The [MASK] is on the table.")
            mask_token: Token used for masking (default: "[MASK]")
            
        Returns:
            List of predictions for each masked token
        """
        self._ensure_model_loaded()
        
        # Check if the model has a masked language modeling head
        if not hasattr(self.model, "cls") or not hasattr(self.model.cls, "predictions"):
            # Try to load a masked language modeling version of the model
            self._load_masked_lm_model()
        
        # Tokenize the input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Find positions of mask tokens
        input_ids = inputs["input_ids"][0].tolist()
        mask_token_id = self.tokenizer.convert_tokens_to_ids(mask_token)
        mask_positions = [i for i, token_id in enumerate(input_ids) if token_id == mask_token_id]
        
        if not mask_positions:
            return {"error": f"No {mask_token} tokens found in input text"}
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get predictions for masked tokens
        predictions = []
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            for mask_pos in mask_positions:
                # Get top 5 predictions for each mask position
                mask_logits = logits[0, mask_pos]
                top_indices = torch.topk(mask_logits, 5).indices.tolist()
                top_tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in top_indices]
                
                # Add to predictions
                token_predictions = [
                    {"token": token, "score": float(mask_logits[idx].item())}
                    for idx, token in zip(top_indices, top_tokens)
                ]
                predictions.append({
                    "position": mask_pos,
                    "predictions": token_predictions
                })
        
        return predictions
    
    def similarity(self, texts1, texts2):
        """
        Calculate similarity between two sets of texts using BERT embeddings.
        
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
        """Load the BERT model and tokenizer"""
        try:
            # Try to use resource pool if available
            if RESOURCE_POOL_AVAILABLE:
                pool = get_global_resource_pool()
                
                # Define model constructor
                def create_model():
                    # For BERT models, we use AutoModel
                    from transformers import AutoModel, AutoConfig, BertModel
                    
                    config = AutoConfig.from_pretrained(self.model_id)
                    model = AutoModel.from_pretrained(self.model_id, config=config, **kwargs)
                    model.to(self.device)
                    model.eval()
                    return model
                
                # Define tokenizer constructor
                def create_tokenizer():
                    from transformers import AutoTokenizer, BertTokenizer
                    return AutoTokenizer.from_pretrained(self.model_id)
                
                # Get or create model from pool
                self.model = pool.get_model(
                    model_type="bert",  # Explicitly set as BERT
                    model_name=self.model_id,
                    constructor=create_model
                )
                
                # Get or create tokenizer from pool
                self.tokenizer = pool.get_tokenizer(
                    model_type="bert",  # Explicitly set as BERT
                    model_name=self.model_id,
                    constructor=create_tokenizer
                )
            else:
                # Load directly without resource pooling
                from transformers import AutoModel, AutoConfig, AutoTokenizer, BertModel, BertTokenizer
                
                self.config = AutoConfig.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id, config=self.config, **kwargs)
                self.model.to(self.device)
                self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            self.is_initialized = True
            self.logger.info(f"BERT model {self.model_id} loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading BERT model {self.model_id}: {str(e)}")
            raise
    
    def _load_masked_lm_model(self):
        """Load a masked language modeling version of the model"""
        try:
            from transformers import BertForMaskedLM, AutoModelForMaskedLM
            
            # Try to use resource pool if available
            if RESOURCE_POOL_AVAILABLE:
                pool = get_global_resource_pool()
                
                def create_mlm_model():
                    model = AutoModelForMaskedLM.from_pretrained(self.model_id)
                    model.to(self.device)
                    model.eval()
                    return model
                
                # Get or create MLM model from pool
                self.model = pool.get_model(
                    model_type="bert_mlm",  # Different type to avoid conflicts
                    model_name=self.model_id,
                    constructor=create_mlm_model
                )
            else:
                # Load directly without resource pooling
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
                self.model.to(self.device)
                self.model.eval()
            
            self.logger.info(f"Loaded masked language model version of {self.model_id}")
        except Exception as e:
            self.logger.error(f"Error loading masked language model: {str(e)}")
            raise
    
    def _pool_embeddings(self, outputs, inputs):
        """
        Pool token embeddings to sentence embeddings for BERT models.
        
        Args:
            outputs: Model outputs
            inputs: Model inputs with attention mask
            
        Returns:
            Pooled embeddings
        """
        # BERT-specific pooling prioritizes CLS token by default
        if self.POOLING_STRATEGY == "cls" and hasattr(outputs, "last_hidden_state"):
            # Use CLS token for BERT (first token)
            return outputs.last_hidden_state[:, 0]
        
        # Fall back to parent implementation for other strategies
        return super()._pool_embeddings(outputs, inputs)"""

SECTION_MAIN = """def main():
    """Example usage for BERT model"""
    import argparse
    import numpy as np
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f"Example usage of {hf_{{ model_name }}.__name__}")
    parser.add_argument("--model", type=str, default="{{ model_name }}", help="Model ID to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--text", type=str, default="Hello, world\!", help="Text to encode")
    parser.add_argument("--compare", type=str, default=None, help="Compare similarity with this text")
    parser.add_argument("--mask", type=str, default=None, help="Text with [MASK] token for prediction")
    args = parser.parse_args()
    
    # Create model instance
    model = hf_{{ model_name }}(model_id=args.model, device=args.device)
    
    # Process based on command
    if args.mask:
        # Predict masked tokens
        predictions = model.predict_masked_tokens(args.mask)
        print(f"Masked text: {args.mask}")
        print("Predictions:")
        for pred in predictions:
            print(f"  Position {pred['position']}:")
            for p in pred['predictions']:
                print(f"    {p['token']}: {p['score']:.4f}")
    else:
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
