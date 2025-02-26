import os
import json
from typing import Dict, Optional, List

class api_models:
    """API Models Registry
    
    This class manages the routing of model requests to appropriate API backends.
    It loads model lists from JSON files and provides lookup functionality to
    determine which backend should handle a given model.
    """
    
    def __init__(self):
        self.model_lists = {}
        self._load_model_lists()
        
    def _load_model_lists(self):
        """Load all model lists from the model_list directory"""
        model_list_dir = os.path.join(os.path.dirname(__file__), 'model_list')
        if not os.path.exists(model_list_dir):
            os.makedirs(model_list_dir)
            
        for filename in os.listdir(model_list_dir):
            if filename.endswith('.json'):
                backend_name = filename[:-5]  # Remove .json
                file_path = os.path.join(model_list_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        self.model_lists[backend_name] = json.load(f)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def get_backend_for_model(self, model_name: str) -> Optional[str]:
        """Determine which backend should handle a given model
        
        Args:
            model_name: Name of the model (e.g., "openai/gpt-4", "google/gemini-pro")
            
        Returns:
            str: Name of the backend that handles this model, or None if not found
        """
        # First try exact match
        for backend, models in self.model_lists.items():
            if model_name in models:
                return backend
        
        # Try matching by provider prefix
        provider = model_name.split('/')[0] if '/' in model_name else ''
        if provider:
            provider_map = {
                'openai': 'openai_api',
                'google': 'gemini',
                'anthropic': 'claude',
                'huggingface': 'hf_tgi',  # Default to TGI unless it's an embedding model
                'openvino': 'ovms',
                'groq': 'groq',
                'ollama': 'ollama'
            }
            if provider in provider_map:
                # Special case for Huggingface models
                if provider == 'huggingface':
                    # Check if it's an embedding model
                    if any(term in model_name.lower() for term in ['embedding', 'encoder', 'sentence']):
                        return 'hf_tei'
                return provider_map[provider]
                
        return None
    
    def get_models_for_backend(self, backend_name: str) -> List[str]:
        """Get list of models supported by a specific backend
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            list: List of model names supported by this backend
        """
        return self.model_lists.get(backend_name, [])
