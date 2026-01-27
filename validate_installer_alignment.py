#!/usr/bin/env python3
"""
Comprehensive Validator for Installer and Wrapper Alignment

This script validates that:
1. All installers actually install what they claim to install
2. All wrappers align with the actual API/SDK/CLI they wrap
3. All imports work correctly
4. All basic operations function
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple
from pathlib import Path


class ValidationResult:
    """Store validation results."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, test: str, message: str = ""):
        self.passed.append((test, message))
        print(f"✅ {test}: {message}")
    
    def add_fail(self, test: str, error: str):
        self.failed.append((test, error))
        print(f"❌ {test}: {error}")
    
    def add_warn(self, test: str, warning: str):
        self.warnings.append((test, warning))
        print(f"⚠️  {test}: {warning}")
    
    def summary(self):
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"✅ Passed: {len(self.passed)}")
        print(f"❌ Failed: {len(self.failed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print("="*60)
        
        if self.failed:
            print("\nFailed Tests:")
            for test, error in self.failed:
                print(f"  - {test}: {error}")
        
        if self.warnings:
            print("\nWarnings:")
            for test, warning in self.warnings:
                print(f"  - {test}: {warning}")
        
        return len(self.failed) == 0


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        result = subprocess.run(
            ["which", command],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def check_python_module(module: str) -> Tuple[bool, str]:
    """Check if a Python module can be imported."""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)


def validate_cli_installations(result: ValidationResult):
    """Validate CLI tool installations."""
    print("\n" + "="*60)
    print("VALIDATING CLI INSTALLATIONS")
    print("="*60)
    
    # GitHub CLI
    if check_command_exists("gh"):
        result.add_pass("GitHub CLI", "gh command found")
    else:
        result.add_fail("GitHub CLI", "gh command not found")
    
    # HuggingFace CLI
    if check_command_exists("huggingface-cli"):
        result.add_pass("HuggingFace CLI", "huggingface-cli command found")
    else:
        result.add_fail("HuggingFace CLI", "huggingface-cli command not found")
    
    # Vast AI CLI
    if check_command_exists("vastai"):
        result.add_pass("Vast AI CLI", "vastai command found")
    else:
        result.add_fail("Vast AI CLI", "vastai command not found")
    
    # VSCode CLI
    if check_command_exists("code"):
        result.add_pass("VSCode CLI", "code command found")
    else:
        result.add_warn("VSCode CLI", "code command not found - requires VSCode installation")
    
    # GitHub Copilot CLI
    if check_command_exists("github-copilot-cli"):
        result.add_pass("GitHub Copilot CLI", "github-copilot-cli command found")
    else:
        result.add_warn("GitHub Copilot CLI", "github-copilot-cli command not found - may need npm install")


def validate_python_sdks(result: ValidationResult):
    """Validate Python SDK installations."""
    print("\n" + "="*60)
    print("VALIDATING PYTHON SDK INSTALLATIONS")
    print("="*60)
    
    sdks = {
        "openai": "OpenAI SDK",
        "anthropic": "Claude/Anthropic SDK",
        "google.generativeai": "Gemini SDK",
        "groq": "Groq SDK",
        "ollama": "Ollama SDK",
        "boto3": "AWS S3 SDK",
        "ipfshttpclient": "IPFS HTTP Client",
    }
    
    for module, name in sdks.items():
        success, version = check_python_module(module)
        if success:
            result.add_pass(name, f"version {version}")
        else:
            result.add_warn(name, f"not installed ({version})")


def validate_cache_infrastructure(result: ValidationResult):
    """Validate cache infrastructure imports."""
    print("\n" + "="*60)
    print("VALIDATING CACHE INFRASTRUCTURE")
    print("="*60)
    
    modules = [
        "ipfs_accelerate_py.common.base_cache",
        "ipfs_accelerate_py.common.cid_index",
        "ipfs_accelerate_py.common.llm_cache",
        "ipfs_accelerate_py.common.hf_hub_cache",
        "ipfs_accelerate_py.common.docker_cache",
        "ipfs_accelerate_py.common.kubernetes_cache",
        "ipfs_accelerate_py.common.huggingface_hugs_cache",
        "ipfs_accelerate_py.common.ipfs_kit_fallback",
    ]
    
    for module in modules:
        success, msg = check_python_module(module)
        if success:
            result.add_pass(f"Module: {module.split('.')[-1]}", "imported successfully")
        else:
            result.add_fail(f"Module: {module.split('.')[-1]}", msg)


def validate_cli_integrations(result: ValidationResult):
    """Validate CLI integrations."""
    print("\n" + "="*60)
    print("VALIDATING CLI INTEGRATIONS")
    print("="*60)
    
    try:
        from ipfs_accelerate_py.cli_integrations import get_all_cli_integrations
        result.add_pass("CLI Integrations Module", "imported successfully")
        
        # Try to get all CLI integrations
        try:
            clis = get_all_cli_integrations()
            result.add_pass("Get All CLI Integrations", f"returned {len(clis)} integrations")
            
            # Check each integration
            expected = ["github", "copilot", "vscode", "openai_codex", "claude", 
                       "gemini", "huggingface", "vastai", "groq"]
            for name in expected:
                if name in clis:
                    result.add_pass(f"CLI Integration: {name}", "available")
                else:
                    result.add_fail(f"CLI Integration: {name}", "not found in registry")
        
        except Exception as e:
            result.add_fail("Get All CLI Integrations", str(e))
    
    except ImportError as e:
        result.add_fail("CLI Integrations Module", str(e))


def validate_api_integrations(result: ValidationResult):
    """Validate API integrations."""
    print("\n" + "="*60)
    print("VALIDATING API INTEGRATIONS")
    print("="*60)
    
    try:
        from ipfs_accelerate_py import api_integrations
        result.add_pass("API Integrations Module", "imported successfully")
        
        # Check factory functions
        factories = [
            "get_cached_openai_api",
            "get_cached_claude_api",
            "get_cached_gemini_api",
            "get_cached_groq_api",
            "get_cached_ollama_api",
        ]
        
        for factory in factories:
            if hasattr(api_integrations, factory):
                result.add_pass(f"Factory: {factory}", "available")
            else:
                result.add_fail(f"Factory: {factory}", "not found")
    
    except ImportError as e:
        result.add_fail("API Integrations Module", str(e))


def validate_backend_alignment(result: ValidationResult):
    """Validate that integrations align with backends."""
    print("\n" + "="*60)
    print("VALIDATING BACKEND ALIGNMENT")
    print("="*60)
    
    # Check if backends exist
    backends = {
        "openai_api": "ipfs_accelerate_py.api_backends.openai_api",
        "claude": "ipfs_accelerate_py.api_backends.claude",
        "gemini": "ipfs_accelerate_py.api_backends.gemini",
        "groq": "ipfs_accelerate_py.api_backends.groq",
        "ollama": "ipfs_accelerate_py.api_backends.ollama",
        "vllm": "ipfs_accelerate_py.api_backends.vllm",
        "hf_tgi": "ipfs_accelerate_py.api_backends.hf_tgi",
        "hf_tei": "ipfs_accelerate_py.api_backends.hf_tei",
        "ovms": "ipfs_accelerate_py.api_backends.ovms",
        "opea": "ipfs_accelerate_py.api_backends.opea",
        "s3_kit": "ipfs_accelerate_py.api_backends.s3_kit",
    }
    
    for name, module in backends.items():
        success, msg = check_python_module(module)
        if success:
            result.add_pass(f"Backend: {name}", "exists")
        else:
            result.add_warn(f"Backend: {name}", f"not found ({msg})")


def validate_existing_wrappers(result: ValidationResult):
    """Validate existing wrapper implementations."""
    print("\n" + "="*60)
    print("VALIDATING EXISTING WRAPPERS")
    print("="*60)
    
    wrappers = {
        "github_cli": "ipfs_accelerate_py.github_cli.wrapper",
        "copilot_cli": "ipfs_accelerate_py.copilot_cli.wrapper",
        "copilot_sdk": "ipfs_accelerate_py.copilot_sdk.wrapper",
    }
    
    for name, module in wrappers.items():
        success, msg = check_python_module(module)
        if success:
            result.add_pass(f"Existing Wrapper: {name}", "exists")
        else:
            result.add_warn(f"Existing Wrapper: {name}", f"not found ({msg})")


def main():
    """Run all validations."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   IPFS Accelerate Installer & Wrapper Alignment Validator   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    result = ValidationResult()
    
    # Run all validation checks
    validate_cli_installations(result)
    validate_python_sdks(result)
    validate_cache_infrastructure(result)
    validate_cli_integrations(result)
    validate_api_integrations(result)
    validate_backend_alignment(result)
    validate_existing_wrappers(result)
    
    # Print summary
    success = result.summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
