#!/usr/bin/env python3
"""
Dependency Manager

This module provides dependency checking and mocking capabilities for the generator system.
"""

import os
import sys
import logging
import importlib
from typing import Dict, Any, Optional, List, Union
from unittest.mock import MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyManager:
    """
    Dependency manager for checking and mocking dependencies.
    
    This class provides capabilities to check if dependencies are available and mock them if needed.
    """
    
    def __init__(self, mock_config=None):
        """
        Initialize the dependency manager.
        
        Args:
            mock_config: Configuration for mocking dependencies
        """
        self.mock_config = mock_config or {}
        self.cache = {}
        self.mocks = {}
    
    def check(self, dependency: str) -> Dict[str, Any]:
        """
        Check if a dependency is available.
        
        Args:
            dependency: Dependency name
            
        Returns:
            Dictionary with dependency status
        """
        # Return cached result if available
        if dependency in self.cache:
            return self.cache[dependency]
            
        # Check if dependency should be mocked
        if self.should_mock(dependency):
            result = {"available": False, "mocked": True}
            self.cache[dependency] = result
            return result
            
        # Check if dependency is available
        checker_method = getattr(self, f"check_{dependency}", None)
        if checker_method:
            # Use specific checker method
            result = checker_method()
        else:
            # Use generic checker
            result = self._check_generic(dependency)
            
        # Cache result
        self.cache[dependency] = result
        return result
    
    def check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Check all dependencies.
        
        Returns:
            Dictionary with all dependency statuses
        """
        results = {}
        
        # Check all known dependencies
        for dependency in self.get_known_dependencies():
            results[dependency] = self.check(dependency)
            
        return results
    
    def should_mock(self, dependency: str) -> bool:
        """
        Check if a dependency should be mocked.
        
        Args:
            dependency: Dependency name
            
        Returns:
            True if the dependency should be mocked, False otherwise
        """
        # Check environment variable
        env_var = f"MOCK_{dependency.upper()}"
        if os.environ.get(env_var, "").lower() == "true":
            return True
            
        # Check mock_config
        if self.mock_config.get("enabled", False) and self.mock_config.get(dependency, False):
            return True
            
        return False
    
    def get_mock(self, dependency: str) -> Any:
        """
        Get a mock for a dependency.
        
        Args:
            dependency: Dependency name
            
        Returns:
            Mock object for the dependency
        """
        # Return cached mock if available
        if dependency in self.mocks:
            return self.mocks[dependency]
            
        # Create mock
        mock_creator = getattr(self, f"mock_{dependency}", None)
        if mock_creator:
            # Use specific mock creator
            mock = mock_creator()
        else:
            # Use generic mock
            mock = MagicMock()
            mock.__version__ = "MOCK"
            
        # Cache mock
        self.mocks[dependency] = mock
        return mock
    
    def reset_cache(self) -> None:
        """Reset the dependency check cache."""
        self.cache = {}
    
    def get_known_dependencies(self) -> List[str]:
        """
        Get a list of known dependencies.
        
        Returns:
            List of dependency names
        """
        # Get all check_* methods
        dependencies = []
        for name in dir(self):
            if name.startswith("check_") and callable(getattr(self, name)):
                dependencies.append(name[6:])  # Remove "check_" prefix
                
        return dependencies
    
    def _check_generic(self, dependency: str) -> Dict[str, Any]:
        """
        Generic dependency checker.
        
        Args:
            dependency: Dependency name
            
        Returns:
            Dictionary with dependency status
        """
        try:
            module = importlib.import_module(dependency)
            
            # Get version if available
            version = getattr(module, "__version__", None)
            if version is None:
                version = getattr(module, "version", None)
                
            return {
                "available": True,
                "mocked": False,
                "version": version
            }
        except ImportError:
            return {
                "available": False,
                "mocked": False,
                "error": "Import failed"
            }
        except Exception as e:
            return {
                "available": False,
                "mocked": False,
                "error": str(e)
            }
    
    # Specific dependency checkers
    
    def check_torch(self) -> Dict[str, Any]:
        """
        Check if torch is available.
        
        Returns:
            Dictionary with torch status
        """
        try:
            import torch
            
            # Get device information
            cuda_available = False
            cuda_version = None
            cuda_device_count = 0
            
            if hasattr(torch, "cuda"):
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    cuda_version = torch.version.cuda
                    cuda_device_count = torch.cuda.device_count()
                    
            # Get MPS information
            mps_available = False
            if hasattr(torch.backends, "mps"):
                mps_available = torch.backends.mps.is_available()
                
            return {
                "available": True,
                "mocked": False,
                "version": torch.__version__,
                "cuda_available": cuda_available,
                "cuda_version": cuda_version,
                "cuda_device_count": cuda_device_count,
                "mps_available": mps_available
            }
        except ImportError:
            return {
                "available": False,
                "mocked": False,
                "error": "Import failed"
            }
        except Exception as e:
            return {
                "available": False,
                "mocked": False,
                "error": str(e)
            }
    
    def check_transformers(self) -> Dict[str, Any]:
        """
        Check if transformers is available.
        
        Returns:
            Dictionary with transformers status
        """
        try:
            import transformers
            
            return {
                "available": True,
                "mocked": False,
                "version": transformers.__version__
            }
        except ImportError:
            return {
                "available": False,
                "mocked": False,
                "error": "Import failed"
            }
        except Exception as e:
            return {
                "available": False,
                "mocked": False,
                "error": str(e)
            }
    
    def check_tokenizers(self) -> Dict[str, Any]:
        """
        Check if tokenizers is available.
        
        Returns:
            Dictionary with tokenizers status
        """
        try:
            import tokenizers
            
            return {
                "available": True,
                "mocked": False,
                "version": tokenizers.__version__
            }
        except ImportError:
            return {
                "available": False,
                "mocked": False,
                "error": "Import failed"
            }
        except Exception as e:
            return {
                "available": False,
                "mocked": False,
                "error": str(e)
            }
    
    def check_sentencepiece(self) -> Dict[str, Any]:
        """
        Check if sentencepiece is available.
        
        Returns:
            Dictionary with sentencepiece status
        """
        try:
            import sentencepiece
            
            return {
                "available": True,
                "mocked": False,
                "version": sentencepiece.__version__
            }
        except ImportError:
            return {
                "available": False,
                "mocked": False,
                "error": "Import failed"
            }
        except Exception as e:
            return {
                "available": False,
                "mocked": False,
                "error": str(e)
            }
    
    def check_openvino(self) -> Dict[str, Any]:
        """
        Check if openvino is available.
        
        Returns:
            Dictionary with openvino status
        """
        try:
            import openvino
            
            # Get available devices
            try:
                from openvino.runtime import Core
                core = Core()
                devices = core.available_devices
            except:
                devices = []
                
            return {
                "available": True,
                "mocked": False,
                "version": openvino.__version__,
                "devices": devices
            }
        except ImportError:
            return {
                "available": False,
                "mocked": False,
                "error": "Import failed"
            }
        except Exception as e:
            return {
                "available": False,
                "mocked": False,
                "error": str(e)
            }
    
    # Mock creators
    
    def mock_torch(self) -> MagicMock:
        """
        Create a mock for torch.
        
        Returns:
            Mock torch module
        """
        mock = MagicMock()
        mock.__version__ = "MOCK"
        
        # Mock CUDA
        mock.cuda = MagicMock()
        mock.cuda.is_available = lambda: False
        mock.cuda.device_count = lambda: 0
        
        # Mock device
        mock.device = lambda device_str: device_str
        
        # Mock tensor creation
        def mock_tensor(data, *args, **kwargs):
            result = MagicMock()
            result.shape = getattr(data, "shape", None)
            result.dtype = kwargs.get("dtype", None)
            result.device = kwargs.get("device", "cpu")
            
            # Add some common tensor methods
            result.to = lambda device, *_, **__: result
            result.detach = lambda: result
            result.numpy = lambda: data if hasattr(data, "shape") else data
            result.item = lambda: data[0] if hasattr(data, "__getitem__") else data
            
            return result
            
        mock.tensor = mock_tensor
        
        # Mock no_grad context manager
        mock.no_grad = MagicMock()
        mock.no_grad.return_value.__enter__ = lambda *args: None
        mock.no_grad.return_value.__exit__ = lambda *args: None
        
        return mock
    
    def mock_transformers(self) -> MagicMock:
        """
        Create a mock for transformers.
        
        Returns:
            Mock transformers module
        """
        mock = MagicMock()
        mock.__version__ = "MOCK"
        
        # Mock pipeline
        def mock_pipeline(task, model=None, **kwargs):
            pipeline_mock = MagicMock()
            
            # Define return values for different tasks
            if task == "fill-mask":
                pipeline_mock.return_value = [{"token_str": "Paris", "score": 0.9}]
            elif task == "text-generation":
                pipeline_mock.return_value = [{"generated_text": "This is a generated text."}]
            elif task == "text-classification":
                pipeline_mock.return_value = [{"label": "POSITIVE", "score": 0.9}]
            elif task == "token-classification":
                pipeline_mock.return_value = [{"word": "Paris", "score": 0.9, "entity": "LOC"}]
            elif task == "question-answering":
                pipeline_mock.return_value = {"answer": "Paris", "score": 0.9, "start": 15, "end": 20}
            elif task == "image-classification":
                pipeline_mock.return_value = [{"label": "cat", "score": 0.9}]
            elif task == "automatic-speech-recognition":
                pipeline_mock.return_value = {"text": "This is a transcription."}
            else:
                pipeline_mock.return_value = {}
                
            return pipeline_mock
            
        mock.pipeline = mock_pipeline
        
        # Mock model loading
        model_mock = MagicMock()
        
        # Define forward method with proper return type
        logits_mock = MagicMock()
        model_mock.return_value = MagicMock(logits=logits_mock)
        
        # Configure AutoModel classes
        mock.AutoModelForSequenceClassification.from_pretrained = lambda *args, **kwargs: model_mock
        mock.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: model_mock
        mock.AutoModelForMaskedLM.from_pretrained = lambda *args, **kwargs: model_mock
        mock.AutoModelForQuestionAnswering.from_pretrained = lambda *args, **kwargs: model_mock
        mock.AutoModelForTokenClassification.from_pretrained = lambda *args, **kwargs: model_mock
        
        # Mock tokenizer
        tokenizer_mock = MagicMock()
        tokenizer_mock.encode = lambda text, **kwargs: [101, 102, 103]
        tokenizer_mock.decode = lambda ids, **kwargs: "Decoded text"
        tokenizer_mock.tokenize = lambda text, **kwargs: ["[CLS]", "token", "[SEP]"]
        tokenizer_mock.convert_tokens_to_ids = lambda tokens, **kwargs: [101, 102, 103]
        
        # Special tokens
        tokenizer_mock.pad_token = "[PAD]"
        tokenizer_mock.pad_token_id = 0
        tokenizer_mock.eos_token = "[EOS]"
        tokenizer_mock.eos_token_id = 102
        tokenizer_mock.bos_token = "[BOS]"
        tokenizer_mock.bos_token_id = 101
        tokenizer_mock.mask_token = "[MASK]"
        tokenizer_mock.mask_token_id = 103
        
        # Configure AutoTokenizer
        mock.AutoTokenizer.from_pretrained = lambda *args, **kwargs: tokenizer_mock
        
        return mock
    
    def mock_tokenizers(self) -> MagicMock:
        """
        Create a mock for tokenizers.
        
        Returns:
            Mock tokenizers module
        """
        mock = MagicMock()
        mock.__version__ = "MOCK"
        
        return mock
    
    def mock_sentencepiece(self) -> MagicMock:
        """
        Create a mock for sentencepiece.
        
        Returns:
            Mock sentencepiece module
        """
        mock = MagicMock()
        mock.__version__ = "MOCK"
        
        return mock
    
    def mock_openvino(self) -> MagicMock:
        """
        Create a mock for openvino.
        
        Returns:
            Mock openvino module
        """
        mock = MagicMock()
        mock.__version__ = "MOCK"
        
        # Mock Core class
        core_mock = MagicMock()
        core_mock.available_devices = ["CPU"]
        mock.runtime = MagicMock()
        mock.runtime.Core = lambda: core_mock
        
        return mock