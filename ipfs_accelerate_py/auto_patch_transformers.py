"""
Auto-Patching System for Transformers with Distributed Storage Support

This module provides automatic monkey-patching of HuggingFace transformers
to integrate distributed filesystem support via storage_wrapper and ipfs_kit_py.

Inspired by ipfs_transformers_py pattern but customized for ipfs_accelerate_py:
- Automatically patches transformers.AutoModel classes
- Integrates with storage_wrapper for distributed caching
- Provides CI/CD gating and fallback
- Zero code changes required in worker skillsets

Usage:
    # Apply patches before importing worker skillsets
    from ipfs_accelerate_py import auto_patch_transformers
    auto_patch_transformers.apply()
    
    # Now all imports of transformers will use patched versions
    from worker.skillset import hf_bert  # Uses patched transformers automatically
    
    # Disable patching
    auto_patch_transformers.disable()
"""

import importlib
import importlib.abc
import importlib.util
from importlib.machinery import PathFinder
import os
import sys
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)

try:
    from .deps_resolver import deps_get as _deps_get
    from .deps_resolver import deps_set as _deps_set
    from .deps_resolver import resolve_module as _resolve_module
except Exception:  # pragma: no cover
    _deps_get = None  # type: ignore
    _deps_set = None  # type: ignore
    _resolve_module = None  # type: ignore


_default_deps: object | None = None


def set_default_deps(deps: object | None) -> None:
    """Set a default deps container used by lazy patch hooks."""
    global _default_deps
    _default_deps = deps

# Track patching state
_patching_enabled = False
_original_from_pretrained = {}
_patch_applied = False
_lazy_hook_installed = False
_lazy_attr_hook_installed = False

_CLASSES_TO_PATCH = [
    'AutoModel',
    'AutoModelForCausalLM',
    'AutoModelForSeq2SeqLM',
    'AutoModelForSequenceClassification',
    'AutoModelForTokenClassification',
    'AutoModelForQuestionAnswering',
    'AutoModelForMaskedLM',
    'AutoModelForImageClassification',
    'AutoModelForObjectDetection',
    'AutoModelForImageSegmentation',
    'AutoModelForSemanticSegmentation',
    'AutoModelForInstanceSegmentation',
    'AutoModelForUniversalSegmentation',
    'AutoModelForZeroShotImageClassification',
    'AutoModelForDepthEstimation',
    'AutoModelForVideoClassification',
    'AutoModelForVision2Seq',
    'AutoModelForVisualQuestionAnswering',
    'AutoModelForDocumentQuestionAnswering',
    'AutoModelForMaskedImageModeling',
    'AutoModelForAudioClassification',
    'AutoModelForAudioFrameClassification',
    'AutoModelForCTC',
    'AutoModelForSpeechSeq2Seq',
    'AutoModelForAudioXVector',
    'AutoModelForTextToSpectrogram',
    'AutoModelForTextToWaveform',
    'AutoModelForTableQuestionAnswering',
    'AutoTokenizer',
    'AutoProcessor',
    'AutoConfig',
    'AutoFeatureExtractor',
    'AutoImageProcessor',
]


def _install_transformers_attr_hook(transformers_module) -> None:
    """Patch transformers lazily when a supported class is first accessed."""
    global _lazy_attr_hook_installed

    if _lazy_attr_hook_installed:
        return

    original_getattr = getattr(transformers_module, "__getattr__", None)
    if original_getattr is None:
        return

    def _patched_getattr(name: str):
        attr = original_getattr(name)
        if name in _CLASSES_TO_PATCH and hasattr(attr, "from_pretrained"):
            try:
                patch_transformers_class(attr, f"transformers.{name}")
                logger.info(f"Patched transformers.{name} on first access")
            except Exception as e:
                logger.warning(f"Failed to patch {name} on access: {e}")
        return attr

    transformers_module.__getattr__ = _patched_getattr  # type: ignore[assignment]
    _lazy_attr_hook_installed = True


class _TransformersLazyPatchHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Meta path hook that patches transformers on first import."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "transformers":
            return None

        # IMPORTANT: do *not* call importlib.util.find_spec() here.
        # That function consults sys.meta_path, which includes this hook,
        # leading to infinite recursion / RecursionError during import.
        spec = PathFinder.find_spec(fullname, path)
        if spec is None:
            return None

        # Wrap the loader to apply patches after module execution.
        original_loader = spec.loader

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                if original_loader is not None and hasattr(original_loader, "create_module"):
                    return original_loader.create_module(spec)
                return None

            def exec_module(self, module):
                if original_loader is not None and hasattr(original_loader, "exec_module"):
                    original_loader.exec_module(module)
                else:
                    importlib.import_module(fullname)

                try:
                    _install_transformers_attr_hook(module)
                    apply(patch_loaded_only=True, deps=_default_deps, transformers_module=module)
                except Exception as e:
                    logger.warning(f"Failed to auto-apply transformers patches: {e}")

        spec.loader = _Loader()
        return spec


def is_patching_enabled() -> bool:
    """Check if patching is currently enabled."""
    return _patching_enabled


def should_patch() -> bool:
    """
    Determine if patching should be applied based on environment.
    
    Returns:
        True if patching should be applied, False otherwise
    """
    # Check explicit disable flags
    if os.environ.get('TRANSFORMERS_PATCH_DISABLE', '').lower() in ('1', 'true', 'yes'):
        logger.debug("Transformers patching disabled via TRANSFORMERS_PATCH_DISABLE")
        return False
    
    if os.environ.get('IPFS_KIT_DISABLE', '').lower() in ('1', 'true', 'yes'):
        logger.debug("Transformers patching disabled via IPFS_KIT_DISABLE")
        return False
    
    if os.environ.get('STORAGE_FORCE_LOCAL', '').lower() in ('1', 'true', 'yes'):
        logger.debug("Transformers patching disabled via STORAGE_FORCE_LOCAL")
        return False
    
    # Auto-detect CI/CD environment
    if os.environ.get('CI', ''):
        logger.debug("CI environment detected, disabling transformers patching")
        return False
    
    # Default to enabled
    return True


def create_patched_from_pretrained(original_from_pretrained: Callable, class_name: str) -> Callable:
    """
    Create a patched version of from_pretrained that uses distributed storage.
    
    Args:
        original_from_pretrained: Original from_pretrained method
        class_name: Name of the class being patched (for logging)
    
    Returns:
        Patched from_pretrained method
    """
    @wraps(original_from_pretrained)
    def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        """
        Patched from_pretrained that integrates with storage_wrapper.
        
        This method:
        1. Checks if storage_wrapper is available and distributed storage is enabled
        2. If yes, modifies cache_dir to use storage_wrapper's cache directory
        3. Falls back to original behavior if storage_wrapper unavailable
        """
        # Try to use storage_wrapper for cache_dir
        try:
            from .common.storage_wrapper import get_storage_wrapper
            
            storage = get_storage_wrapper(auto_detect_ci=True)
            
            if storage and storage.is_distributed:
                # Get the distributed cache directory
                cache_dir = str(storage.get_cache_dir())
                
                # If user didn't specify cache_dir, use ours
                if 'cache_dir' not in kwargs:
                    kwargs['cache_dir'] = cache_dir
                    logger.debug(
                        f"{class_name}.from_pretrained using distributed cache: {cache_dir}"
                    )
                else:
                    # User specified cache_dir, respect it but log
                    logger.debug(
                        f"{class_name}.from_pretrained using user-specified cache_dir: {kwargs['cache_dir']}"
                    )
            else:
                logger.debug(
                    f"{class_name}.from_pretrained using standard cache (distributed storage unavailable)"
                )
        except ImportError:
            # storage_wrapper not available, use original behavior
            logger.debug(
                f"{class_name}.from_pretrained using standard cache (storage_wrapper not available)"
            )
        except Exception as e:
            # Any other error, fall back gracefully
            logger.debug(
                f"{class_name}.from_pretrained storage_wrapper error (falling back): {e}"
            )
        
        # Call original from_pretrained with potentially modified kwargs
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    
    return patched_from_pretrained


def patch_transformers_class(cls, class_name: str) -> None:
    """
    Patch a single transformers class to use distributed storage.
    
    Args:
        cls: The class to patch
        class_name: Name of the class (for logging and tracking)
    """
    if hasattr(cls, 'from_pretrained') and callable(cls.from_pretrained):
        # Store original method
        if class_name not in _original_from_pretrained:
            _original_from_pretrained[class_name] = cls.from_pretrained
        
        # Apply patch
        cls.from_pretrained = classmethod(
            create_patched_from_pretrained(
                _original_from_pretrained[class_name].__func__,
                class_name
            )
        )
        logger.info(f"Patched {class_name}.from_pretrained for distributed storage")


def apply(
    patch_loaded_only: bool = False,
    *,
    deps: object | None = None,
    transformers_module=None,
) -> bool:
    """
    Apply patches to transformers classes.
    
    This method patches all AutoModel classes in the transformers library
    to automatically use distributed storage when available.
    
    Returns:
        True if patches were applied, False otherwise
    """
    global _patching_enabled, _patch_applied

    if deps is None:
        deps = _default_deps
    
    # Check if we should patch
    if not should_patch():
        logger.info("Transformers auto-patching disabled by environment")
        return False
    
    # Check if already patched (module-global or deps-cached)
    if _patch_applied:
        if deps is not None and callable(_deps_set):
            try:
                _deps_set(deps, "ipfs_accelerate_py::transformers_patched", True)
                if "transformers" in sys.modules:
                    _deps_set(deps, "pip::transformers", sys.modules["transformers"])
            except Exception:
                pass
        logger.debug("Transformers already patched, skipping")
        return True
    if deps is not None and callable(_deps_get):
        try:
            if _deps_get(deps, "ipfs_accelerate_py::transformers_patched"):
                _patching_enabled = True
                _patch_applied = True
                logger.debug("Transformers already patched (deps cache), skipping")
                return True
        except Exception:
            pass
    
    transformers = None
    if transformers_module is not None:
        transformers = transformers_module
    elif callable(_resolve_module):
        transformers = _resolve_module("transformers", deps=deps, cache_key="pip::transformers")
    if transformers is None:
        try:
            import transformers  # type: ignore
        except ImportError:
            logger.warning("transformers not available, cannot apply patches")
            return False
    
    _install_transformers_attr_hook(transformers)
    
    patched_count = 0
    for class_name in _CLASSES_TO_PATCH:
        if patch_loaded_only and class_name not in transformers.__dict__:
            continue
        if hasattr(transformers, class_name):
            cls = getattr(transformers, class_name)
            try:
                patch_transformers_class(cls, f"transformers.{class_name}")
                patched_count += 1
            except Exception as e:
                logger.warning(f"Failed to patch {class_name}: {e}")
        else:
            logger.debug(f"{class_name} not found in transformers, skipping")
    
    _patching_enabled = True
    _patch_applied = True

    if deps is not None and callable(_deps_set):
        try:
            _deps_set(deps, "ipfs_accelerate_py::transformers_patched", True)
            if transformers is not None:
                _deps_set(deps, "pip::transformers", transformers)
        except Exception:
            pass
    
    if patched_count:
        logger.info(
            f"Successfully patched {patched_count} transformers classes for distributed storage"
        )
    return True


def install_lazy_patch_hook() -> None:
    """Install a meta path hook to patch transformers on first import."""
    global _lazy_hook_installed

    if _lazy_hook_installed:
        return

    if "transformers" in sys.modules:
        try:
            _install_transformers_attr_hook(sys.modules["transformers"])
            apply(patch_loaded_only=True, deps=_default_deps, transformers_module=sys.modules["transformers"])
        except Exception as e:
            logger.warning(f"Failed to auto-apply transformers patches: {e}")
        return

    sys.meta_path.insert(0, _TransformersLazyPatchHook())
    _lazy_hook_installed = True
    logger.info("Installed lazy transformers patch hook")


def restore() -> bool:
    """
    Restore original transformers methods.
    
    This undoes the patching and restores the original behavior.
    
    Returns:
        True if restoration was successful, False otherwise
    """
    global _patching_enabled, _patch_applied
    
    if not _patch_applied:
        logger.debug("No patches to restore")
        return False
    
    try:
        import transformers
    except ImportError:
        logger.warning("transformers not available, cannot restore")
        return False
    
    restored_count = 0
    for class_full_name, original_method in _original_from_pretrained.items():
        # Parse class name
        if '.' in class_full_name:
            class_name = class_full_name.split('.')[-1]
        else:
            class_name = class_full_name
        
        if hasattr(transformers, class_name):
            cls = getattr(transformers, class_name)
            cls.from_pretrained = classmethod(original_method)
            restored_count += 1
            logger.debug(f"Restored {class_full_name}.from_pretrained")
    
    _patching_enabled = False
    _patch_applied = False
    _original_from_pretrained.clear()
    
    logger.info(f"Restored {restored_count} transformers classes to original state")
    return True


def disable() -> None:
    """Disable patching and restore original behavior."""
    restore()


def get_status() -> Dict[str, Any]:
    """
    Get current patching status.
    
    Returns:
        Dictionary with patching status information
    """
    return {
        'enabled': _patching_enabled,
        'applied': _patch_applied,
        'patched_classes': list(_original_from_pretrained.keys()),
        'should_patch': should_patch(),
    }


# Install lazy patch hook on import if environment allows
if should_patch():
    install_lazy_patch_hook()
