"""Integration between ipfs_accelerate_py and ipfs_transformers_py."""

import os
import logging
import tempfile
from typing import Dict, Any, Optional, List, Union

# Import the modified ipfs_transformers_py
try:
    from ipfs_transformers_py.ipfs_transformers_py.ipfs_transformers import AutoModel
    #from ipfs_transformers_py.ipfs_transformers_py.ipfs_kit_bridge import IPFSKitBridge
    from ipfs_kit_py.ipfs_kit_py.high_level_api import IPFSSimpleAPI
    TRANSFORMERS_AVAILABLE = True
    print("`from ipfs_transformers_py.ipfs_transformers_py.ipfs_transformers import AutoModel` is successful")
except ImportError:
    try:
        # Try alternative import paths
        from ipfs_transformers_py.ipfs_transformers import AutoModel
        #from ipfs_transformers_py.ipfs_kit_bridge import IPFSKitBridge
        from ipfs_kit_py.high_level_api import IPFSSimpleAPI
        TRANSFORMERS_AVAILABLE = True
        print("`from ipfs_transformers_py.ipfs_transformers import AutoModel` is successful")
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        logging.warning("ipfs_transformers_py not available. Using regular transformers.")
        try:
            from transformers import AutoModel as HFAutoModel
            HAS_TRANSFORMERS = True
        except ImportError:
            HAS_TRANSFORMERS = False
            logging.error("Neither ipfs_transformers_py nor transformers is available.")


# Try to import ipfs_kit_py for IPFS functionality
try:
    import ipfs_kit_py
    from ipfs_kit_py.high_level_api import IPFSSimpleAPI
    HAS_IPFS_KIT = True
except ImportError:
    HAS_IPFS_KIT = False
    logging.warning("ipfs_kit_py not available. IPFS functionality will be limited.")

# Try to import storage wrapper
try:
    from .common.storage_wrapper import get_storage_wrapper
    HAS_STORAGE_WRAPPER = True
except ImportError:
    HAS_STORAGE_WRAPPER = False
    get_storage_wrapper = None
    logging.debug("Storage wrapper not available for transformers integration")


# Create a simple implementation of IPFSKitBridge if the real one doesn't exist
class IPFSKitBridge:
    """Bridge between ipfs_transformers and ipfs_kit_py."""

    def __init__(self, config=None):
        """Initialize the bridge with config."""
        self.config = config or {}
        self.ipfs_api = None
        self._storage_wrapper = None

        # Initialize storage wrapper for distributed filesystem (with gating)
        if HAS_STORAGE_WRAPPER:
            try:
                self._storage_wrapper = get_storage_wrapper(auto_detect_ci=True)
                if self._storage_wrapper.is_distributed:
                    logging.info("Transformers bridge using distributed storage backend")
                else:
                    logging.debug("Transformers bridge using local filesystem")
            except Exception as e:
                logging.debug(f"Storage wrapper initialization skipped: {e}")

        if HAS_IPFS_KIT:
            try:
                # Get configuration
                role = self.config.get("role", "leecher")
                async_backend = self.config.get("async_backend", "anyio")

                # Initialize the API
                self.ipfs_api = IPFSSimpleAPI(role=role, async_backend=async_backend)
                logging.info(f"Initialized IPFS API with role={role}")
            except Exception as e:
                logging.error(f"Failed to initialize IPFS API: {e}")

    def get_from_ipfs(self, cid: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Get content from IPFS by CID (with distributed storage integration)."""
        # Try storage wrapper first if available
        if self._storage_wrapper and self._storage_wrapper.is_distributed:
            try:
                data = self._storage_wrapper.read_file(cid)
                if data:
                    # Create output directory and write file
                    if output_dir is None:
                        output_dir = tempfile.mkdtemp()
                    
                    output_path = os.path.join(output_dir, "content")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(data)
                    
                    return {"success": True, "path": output_path, "source": "distributed_storage"}
            except Exception as e:
                logging.debug(f"Failed to get from distributed storage: {e}")
        
        # Try IPFS API fallback
        if not HAS_IPFS_KIT or not self.ipfs_api:
            return {"success": False, "error": "IPFS functionality not available"}

        try:
            # Create temp dir if not provided
            if output_dir is None:
                output_dir = tempfile.mkdtemp()

            # Download from IPFS
            path = self.ipfs_api.get_to_file(cid, output_dir)

            return {"success": True, "path": path, "source": "ipfs_api"}
        except Exception as e:
            logging.error(f"Error getting model from IPFS: {e}")
            return {"success": False, "error": str(e)}

    def add_to_ipfs(self, path: str) -> Dict[str, Any]:
        """Add content to IPFS (with distributed storage integration)."""
        # Try storage wrapper first if available
        if self._storage_wrapper and self._storage_wrapper.is_distributed:
            try:
                if os.path.isfile(path):
                    with open(path, 'rb') as f:
                        data = f.read()
                    cid = self._storage_wrapper.write_file(
                        data,
                        filename=os.path.basename(path),
                        pin=True
                    )
                    return {"success": True, "cid": cid, "source": "distributed_storage"}
            except Exception as e:
                logging.debug(f"Failed to add to distributed storage: {e}")
        
        # Try IPFS API fallback
        if not HAS_IPFS_KIT or not self.ipfs_api:
            return {"success": False, "error": "IPFS functionality not available"}

        try:
            if os.path.isdir(path):
                cid = self.ipfs_api.add_directory(path)
            else:
                cid = self.ipfs_api.add_file(path)

            return {"success": True, "cid": cid, "source": "ipfs_api"}
        except Exception as e:
            logging.error(f"Error adding to IPFS: {e}")
            return {"success": False, "error": str(e)}


class TransformersModelProvider:
    """Provider class that loads models using ipfs_transformers_py."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration options."""
        self.config = config or {}
        self.ipfs_bridge = None

        # Initialize the IPFS bridge if transformers are available
        if TRANSFORMERS_AVAILABLE:
            self.ipfs_bridge = IPFSKitBridge(self.config.get("ipfs_kit", {}))

        # Track loaded models
        self.loaded_models = {}

    def is_available(self) -> bool:
        """Check if transformers integration is available."""
        return TRANSFORMERS_AVAILABLE or HAS_TRANSFORMERS

    def load_model(
        self,
        model_name: str,
        model_type: str = "text_generation",
        use_ipfs: bool = True,
        ipfs_cid: Optional[str] = None,
        s3_config: Optional[Dict[str, str]] = None,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load a model using ipfs_transformers_py with hardware acceleration.

        Args:
            model_name: Name of the model or HF model ID
            model_type: Type of model (text_generation, text_embedding, etc.)
            use_ipfs: Whether to try IPFS sources
            ipfs_cid: Specific IPFS CID (if known)
            s3_config: S3 configuration for S3 sources
            device: Device to load the model on (auto, cpu, cuda, mps, etc.)
            **kwargs: Additional arguments to pass to the model loader

        Returns:
            Dictionary with loaded model and metadata
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "No transformers library available"
            }

        try:
            # Configure hardware-specific options
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Add device to kwargs
            kwargs["device_map"] = device

            # For MPS (Apple Silicon), set torch dtype
            if device == "mps":
                import torch
                kwargs["torch_dtype"] = torch.float16

            # Load the model - use ipfs_transformers_py if available
            if TRANSFORMERS_AVAILABLE:
                if ipfs_cid:
                    # Load directly from IPFS
                    model = AutoModel.from_ipfs(ipfs_cid, **kwargs)
                else:
                    # Auto-download from fastest source
                    model = AutoModel.from_auto_download(
                        model_name=model_name,
                        s3cfg=s3_config,
                        **kwargs
                    )
            else:
                # Fallback to regular transformers
                model = HFAutoModel.from_pretrained(model_name, **kwargs)

            # Generate a unique ID for this model
            import uuid
            model_id = str(uuid.uuid4())

            # Store the model for future reference
            self.loaded_models[model_id] = {
                "model": model,
                "name": model_name,
                "type": model_type,
                "device": device
            }

            # Return success result
            return {
                "success": True,
                "model_id": model_id,
                "model": model,
                "device": device
            }

        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def run_inference_new(self, model_id: str, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run inference on a loaded model with improved error handling."""
        if model_id not in self.loaded_models:
            return {
                "success": False,
                "error": f"Model with ID {model_id} not found"
            }

        try:
            model_info = self.loaded_models[model_id]
            model = model_info["model"]
            model_type = model_info.get("type", "text_generation")

            # Debug: Print input shape
            print(f"Input keys: {inputs.keys()}")

            # Run inference
            import torch
            with torch.no_grad():
                outputs = model(**inputs)

            # Debug: Print outputs type
            print(f"Output type: {type(outputs)}")
            if isinstance(outputs, tuple):
                print(f"Tuple length: {len(outputs)}")
                for i, item in enumerate(outputs):
                    print(f"  Item {i} type: {type(item)}")

            # Create result with success flag
            result = {
                "success": True,
                "model_id": model_id,
                "outputs": {}
            }

            # Handle different output types
            if hasattr(outputs, "last_hidden_state"):
                # Transformers output object with attributes
                result["outputs"]["last_hidden_state"] = outputs.last_hidden_state

                # Copy any other useful attributes
                for attr in ["logits", "hidden_states", "attentions", "cross_attentions"]:
                    if hasattr(outputs, attr):
                        result["outputs"][attr] = getattr(outputs, attr)

            elif isinstance(outputs, tuple):
                # Handle tuple outputs correctly
                for i, item in enumerate(outputs):
                    result["outputs"][f"output_{i}"] = item

                # Typical convention for embeddings is that they're the first element
                if len(outputs) > 0 and model_type == "text_embedding":
                    result["outputs"]["last_hidden_state"] = outputs[0]

            elif isinstance(outputs, list):
                # Handle list outputs correctly - use integer indices
                for i, item in enumerate(outputs):
                    result["outputs"][f"output_{i}"] = item

            elif isinstance(outputs, torch.Tensor):
                # Single tensor output
                result["outputs"]["tensor_output"] = outputs

                # For embedding models, treat this as the embedding
                if model_type == "text_embedding":
                    result["outputs"]["last_hidden_state"] = outputs

            elif hasattr(outputs, "to_dict"):
                # Some HF outputs have to_dict() method
                dict_output = outputs.to_dict()
                for key, value in dict_output.items():
                    result["outputs"][key] = value

            elif hasattr(outputs, "items"):
                # It's already a dict-like object
                for key, value in outputs.items():
                    result["outputs"][key] = value

            else:
                # Unknown output type, store as is
                result["outputs"]["raw_output"] = outputs

            return result

        except Exception as e:
            logging.error(f"Error running inference: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def run_inference(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on a loaded model.

        Args:
            model_id: ID of the loaded model
            inputs: Input data for the model
            **kwargs: Additional inference parameters

        Returns:
            Dictionary with inference results
        """
        if model_id not in self.loaded_models:
            return {
                "success": False,
                "error": f"Model with ID {model_id} not found"
            }

        try:
            model_info = self.loaded_models[model_id]
            model = model_info["model"]
            model_type = model_info.get("type", "text_generation")

            # Run inference
            import torch
            with torch.no_grad():
                outputs = model(**inputs, **kwargs)

            # Process outputs based on model type
            result = {
                "success": True,
                "model_id": model_id,
            }

            # Convert model outputs to dictionary while preserving all data
            outputs_dict = {}

            # Try different methods to get a dictionary from outputs
            if hasattr(outputs, "to_dict"):
                # Some HF outputs have to_dict() method
                outputs_dict = outputs.to_dict()
            elif hasattr(outputs, "_asdict"):
                # namedtuples have _asdict()
                outputs_dict = outputs._asdict()
            elif hasattr(outputs, "items"):
                # It's already a dict-like object
                outputs_dict = dict(outputs.items())
            elif isinstance(outputs, (tuple, list)):
                # If it's a tuple/list, preserve the structure
                for i, item in enumerate(outputs):
                    outputs_dict[f"output_{i}"] = item
            else:
                # Single output (like a tensor), use as is
                outputs_dict["output"] = outputs

            # Add additional model-type specific handling
            if model_type == "text_embedding":
                # Make sure last_hidden_state is accessible
                if hasattr(outputs, "last_hidden_state"):
                    outputs_dict["last_hidden_state"] = outputs.last_hidden_state
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    # Some models return (last_hidden_state, ...) tuple
                    outputs_dict["last_hidden_state"] = outputs[0]

            # Add the outputs dictionary to the result
            result["outputs"] = outputs_dict

            # Add processed results based on model type
            if model_info["type"] == "text_generation":
                if hasattr(outputs, "logits"):
                    result["logits"] = outputs.logits.tolist()
            elif model_info["type"] == "text_embedding":
                if hasattr(outputs, "last_hidden_state"):
                    result["embeddings"] = outputs.last_hidden_state.tolist()

            return result

        except Exception as e:
            logging.error(f"Error running inference: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload a model to free up memory."""
        if model_id not in self.loaded_models:
            return {
                "success": False,
                "error": f"Model with ID {model_id} not found"
            }

        try:
            # Get the model info
            model_info = self.loaded_models[model_id]

            # Delete the model and clear from GPU if applicable
            import gc
            import torch

            # Delete from our tracking dict
            del self.loaded_models[model_id]

            # Run garbage collection and empty CUDA cache if available
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "success": True,
                "message": f"Model {model_id} unloaded successfully"
            }

        except Exception as e:
            logging.error(f"Error unloading model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
