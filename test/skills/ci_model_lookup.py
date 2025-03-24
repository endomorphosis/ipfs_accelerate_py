#!/usr/bin/env python3

"""
CI/CD Integration for HuggingFace Model Lookup System.

This script:
1. Provides CI/CD friendly model lookup with environment variable controls
2. Enables automated registry updates on a scheduled basis
3. Skips API calls in CI/CD environments unless explicitly requested
4. Verifies model availability without making expensive API calls

Usage:
    python ci_model_lookup.py --update-registry [--force-api] [--verify-only]
"""

import os
import sys
import json
import logging
import argparse
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_FILE = CURRENT_DIR / "huggingface_model_types.json"
MAX_REGISTRY_AGE_DAYS = 7  # Update registry if older than this

# Environment variables
ENV_USE_STATIC_REGISTRY = "USE_STATIC_REGISTRY"  # Set to "true" to skip API calls
ENV_FORCE_API_CALLS = "FORCE_API_CALLS"  # Set to "true" to force API calls
ENV_MAX_REGISTRY_AGE = "MAX_REGISTRY_AGE_DAYS"  # Override max registry age
ENV_REGISTRY_PATH = "HF_REGISTRY_PATH"  # Override registry file path

def should_use_static_registry():
    """Check if we should use static registry (no API calls)."""
    env_value = os.environ.get(ENV_USE_STATIC_REGISTRY, "").lower()
    return env_value in ("true", "1", "yes")

def should_force_api_calls():
    """Check if we should force API calls."""
    env_value = os.environ.get(ENV_FORCE_API_CALLS, "").lower()
    return env_value in ("true", "1", "yes")

def get_max_registry_age():
    """Get maximum registry age in days."""
    try:
        return int(os.environ.get(ENV_MAX_REGISTRY_AGE, MAX_REGISTRY_AGE_DAYS))
    except (ValueError, TypeError):
        return MAX_REGISTRY_AGE_DAYS

def get_registry_path():
    """Get registry file path, considering environment variables."""
    env_path = os.environ.get(ENV_REGISTRY_PATH)
    if env_path:
        return Path(env_path)
    return REGISTRY_FILE

def load_registry_data():
    """Load model registry data from JSON file."""
    registry_path = get_registry_path()
    try:
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded model registry from {registry_path}")
                return data
        else:
            logger.warning(f"Registry file {registry_path} not found, creating new one")
            return {}
    except Exception as e:
        logger.error(f"Error loading registry file: {e}")
        return {}

def save_registry_data(data):
    """Save model registry data to JSON file."""
    registry_path = get_registry_path()
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved model registry to {registry_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving registry file: {e}")
        return False

def registry_needs_update():
    """Check if registry needs update based on age."""
    registry_path = get_registry_path()
    max_age_days = get_max_registry_age()
    
    if not os.path.exists(registry_path):
        logger.info("Registry file doesn't exist, needs update")
        return True
    
    try:
        # Check file modification time
        mtime = os.path.getmtime(registry_path)
        mtime_dt = datetime.fromtimestamp(mtime)
        now = datetime.now()
        age_days = (now - mtime_dt).days
        
        if age_days > max_age_days:
            logger.info(f"Registry is {age_days} days old (max: {max_age_days}), needs update")
            return True
        
        # Also check individual model entries
        registry_data = load_registry_data()
        if not registry_data:
            return True
        
        # Check timestamps of individual entries
        for model_type, data in registry_data.items():
            updated_at = data.get("updated_at", "")
            try:
                # Parse ISO format timestamp
                updated_dt = datetime.fromisoformat(updated_at)
                entry_age_days = (now - updated_dt).days
                if entry_age_days > max_age_days:
                    logger.info(f"Entry for {model_type} is {entry_age_days} days old, needs update")
                    return True
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp for {model_type}: {updated_at}")
                return True
        
        logger.info(f"Registry is up to date (age: {age_days} days)")
        return False
    except Exception as e:
        logger.error(f"Error checking registry age: {e}")
        return True

def verify_registry_models(registry_data=None):
    """Verify models in registry are valid without API calls."""
    if registry_data is None:
        registry_data = load_registry_data()
    
    if not registry_data:
        logger.warning("Registry is empty, nothing to verify")
        return False
    
    # Import verify function if available
    try:
        sys.path.insert(0, str(CURRENT_DIR))
        from find_models import try_model_access
        has_verification = True
    except ImportError:
        logger.warning("Could not import try_model_access function, using basic verification")
        has_verification = False
    
    verification_results = {}
    
    # Verify each model type's default model
    for model_type, data in registry_data.items():
        default_model = data.get("default_model")
        if not default_model:
            logger.warning(f"No default model for {model_type}")
            verification_results[model_type] = False
            continue
        
        try:
            if has_verification:
                # Use the imported verification function
                result = try_model_access(default_model)
                if result.get("error"):
                    logger.warning(f"Model {default_model} verification failed: {result['error']}")
                    verification_results[model_type] = False
                else:
                    logger.info(f"Model {default_model} verified successfully")
                    verification_results[model_type] = True
            else:
                # Basic string format verification
                if '/' in default_model and len(default_model.split('/')) == 2:
                    logger.info(f"Model {default_model} format looks valid")
                    verification_results[model_type] = True
                else:
                    logger.warning(f"Model {default_model} format looks suspicious")
                    verification_results[model_type] = False
        except Exception as e:
            logger.error(f"Error verifying {default_model}: {e}")
            verification_results[model_type] = False
    
    # Summary
    total = len(verification_results)
    passed = sum(1 for result in verification_results.values() if result)
    
    logger.info(f"Verified {passed}/{total} models ({passed/total*100:.1f}%)")
    return passed == total

def update_registry_from_script(model_types=None, force=False):
    """Update registry using the expand_model_registry.py script."""
    expand_registry_script = CURRENT_DIR / "expand_model_registry.py"
    
    if not os.path.exists(expand_registry_script):
        logger.error(f"expand_model_registry.py not found at {expand_registry_script}")
        return False
    
    try:
        # Determine which model types to update
        if model_types is None:
            # Get all model types from ARCHITECTURE_TYPES
            try:
                sys.path.insert(0, str(CURRENT_DIR))
                from find_models import ARCHITECTURE_TYPES
                all_types = []
                for models in ARCHITECTURE_TYPES.values():
                    all_types.extend(models)
                model_types = sorted(list(set(all_types)))  # Remove duplicates
            except ImportError:
                logger.warning("Could not import ARCHITECTURE_TYPES, using default types")
                model_types = ["bert", "gpt2", "t5", "vit", "roberta", "distilbert", 
                              "gpt-j", "gpt-neo", "whisper", "llama", "clip"]
        
        # Update in batches to avoid overwhelming the API
        batch_size = 3  # Update 3 types at a time
        batches = [model_types[i:i+batch_size] for i in range(0, len(model_types), batch_size)]
        
        success_count = 0
        for batch in batches:
            logger.info(f"Updating models: {', '.join(batch)}")
            
            # Run the script for each model type in batch
            for model_type in batch:
                cmd = [sys.executable, str(expand_registry_script), "--model-type", model_type]
                if force:
                    cmd.append("--force")
                
                import subprocess
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        logger.info(f"✅ Successfully updated {model_type}")
                        success_count += 1
                    else:
                        logger.error(f"❌ Failed to update {model_type}: {result.stderr}")
                except Exception as e:
                    logger.error(f"Error updating {model_type}: {e}")
            
            # Add a slight delay between batches to avoid rate limiting
            if len(batches) > 1:
                delay = random.uniform(1.0, 3.0)
                logger.info(f"Sleeping for {delay:.1f}s before next batch")
                time.sleep(delay)
        
        logger.info(f"Updated {success_count}/{len(model_types)} model types")
        return success_count > 0
    
    except Exception as e:
        logger.error(f"Error updating registry: {e}")
        return False

def backup_registry():
    """Create a backup of the registry file."""
    registry_path = get_registry_path()
    if not os.path.exists(registry_path):
        logger.warning(f"Registry file {registry_path} not found, nothing to backup")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = str(registry_path) + f".bak.{timestamp}"
    
    try:
        import shutil
        shutil.copy2(registry_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return False

def ci_update_registry(force_api=False, verify_only=False, model_types=None):
    """Update registry in CI/CD environment."""
    # Skip API calls if configured and not forced
    if should_use_static_registry() and not force_api:
        logger.info("Using static registry (API calls disabled)")
        
        # Just verify the existing registry
        registry_data = load_registry_data()
        return verify_registry_models(registry_data)
    
    # Check if update is needed
    if not registry_needs_update() and not force_api:
        logger.info("Registry is up to date, skipping update")
        
        # Verify the existing registry
        registry_data = load_registry_data()
        return verify_registry_models(registry_data)
    
    # Backup the registry before updating
    backup_registry()
    
    # If verify only, just verify without updating
    if verify_only:
        logger.info("Verifying registry without updating")
        registry_data = load_registry_data()
        return verify_registry_models(registry_data)
    
    # Update the registry
    logger.info("Updating registry with latest model data")
    success = update_registry_from_script(model_types, force=force_api)
    
    # Verify after update
    if success:
        logger.info("Verifying updated registry")
        registry_data = load_registry_data()
        return verify_registry_models(registry_data)
    
    return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="CI/CD integration for model lookup")
    parser.add_argument("--update-registry", action="store_true", help="Update registry if needed")
    parser.add_argument("--force-api", action="store_true", help="Force API calls even in CI/CD")
    parser.add_argument("--verify-only", action="store_true", help="Only verify without updating")
    parser.add_argument("--models", type=str, help="Comma-separated list of model types to update")
    
    args = parser.parse_args()
    
    # Parse model types if provided
    model_types = None
    if args.models:
        model_types = [m.strip() for m in args.models.split(",")]
    
    # Determine if we're in a CI/CD environment
    is_ci = any(env in os.environ for env in [
        "CI", "GITHUB_ACTIONS", "GITLAB_CI", "TRAVIS", "CIRCLECI", "APPVEYOR", 
        "TF_BUILD", "JENKINS_URL"
    ])
    
    if is_ci:
        logger.info("Running in CI/CD environment")
    
    # Update registry if requested
    if args.update_registry:
        success = ci_update_registry(
            force_api=args.force_api or should_force_api_calls(),
            verify_only=args.verify_only,
            model_types=model_types
        )
        
        return 0 if success else 1
    else:
        # Just verify the registry
        registry_data = load_registry_data()
        success = verify_registry_models(registry_data)
        
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())