import os
import json
import logging
from typing import Dict, Optional, List

try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        try:
            from common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False

if HAVE_STORAGE_WRAPPER:
    try:
        _storage = get_storage_wrapper(auto_detect_ci=True)
    except Exception:
        _storage = None
else:
    _storage = None

class api_models:
    """API Models Registry
    
    This class manages the routing of model requests to appropriate API backends.
    It loads model lists from JSON files and provides lookup functionality to
    determine which backend should handle a given model.
    """
    
    def __init__(self, resources: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """Initialize the api_models registry
        
        Args:
            resources: Optional dictionary containing shared resources
            metadata: Optional dictionary containing configuration metadata
        """
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Load model lists from json files
        self.model_lists = {}
        model_list_dir = os.path.join(os.path.dirname(__file__), 'model_list')
        
        for filename in os.listdir(model_list_dir):
            if filename.endswith('.json'):
                api_name = os.path.splitext(filename)[0]
                filepath = os.path.join(model_list_dir, filename)
                
                # Try distributed storage first
                if _storage and _storage.is_distributed:
                    try:
                        content = _storage.read_file(filepath)
                        if content:
                            self.model_lists[api_name] = json.loads(content)
                            continue
                    except Exception:
                        pass
                
                # Fallback to local filesystem
                try:
                    with open(filepath, 'r') as f:
                        self.model_lists[api_name] = json.load(f)
                except json.JSONDecodeError as e:
                    logging.error(f"Error loading {filename}: {e}")
                    self.model_lists[api_name] = []
                except Exception as e:
                    logging.error(f"Error loading {filename}: {e}")
                    self.model_lists[api_name] = []

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

    def get_models(self, api_name: str) -> List[str]:
        """Get list of models supported by a specific API
        
        Args:
            api_name: Name of the API to get models for
            
        Returns:
            List of model names supported by that API
        """
        return self.model_lists.get(api_name, [])

    def is_compatible_model(self, api_name: str, model_name: str) -> bool:
        """Check if a model is compatible with a specific API
        
        Args:
            api_name: Name of the API to check
            model_name: Name of the model to check
            
        Returns:
            True if the model is compatible with the API, False otherwise
        """
        return model_name in self.model_lists.get(api_name, [])
