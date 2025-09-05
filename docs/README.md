# 📚 IPFS Accelerate Python Documentation

This directory contains comprehensive documentation for the IPFS Accelerate Python framework.

## 📖 Documentation Index

### **Core Documentation**
- [Installation Guide](./INSTALLATION_GUIDE.md) - Complete installation instructions
- [Testing README](./TESTING_README.md) - Testing framework and procedures
- [Installation Troubleshooting](./INSTALLATION_TROUBLESHOOTING_GUIDE.md) - Common issues and solutions

### **Implementation Guides**
- [AI MCP Server Implementation](./AI_MCP_SERVER_IMPLEMENTATION.md) - MCP server setup and usage
- [AI Model Discovery](./AI_MODEL_DISCOVERY_README.md) - Model discovery and management
- [Model Manager](./MODEL_MANAGER_README.md) - Model management system
- [IPFS MCP Integration Plan](./IPFS_ACCELERATE_MCP_INTEGRATION_PLAN.md) - Integration architecture

### **Platform & Migration Guides**
- [WebGPU WebNN Migration](./WEBGPU_WEBNN_MIGRATION_PLAN.md) - Migration strategy
- [WebNN WebGPU Implementation](./WEBNN_WEBGPU_IMPLEMENTATION_SUMMARY.md) - Implementation details
- [WebNN WebGPU README](./WEBNN_WEBGPU_README.md) - Platform-specific documentation
- [Windows Compatibility](./WINDOWS_COMPATIBILITY.md) - Windows support information
- [Python 3.12 Compatibility](./PYTHON312_COMPATIBILITY_SUMMARY.md) - Python version compatibility

### **Testing & Verification**
- [Kitchen Sink Testing Plan](./KITCHEN_SINK_TESTING_PLAN.md) - Comprehensive testing strategy
- [Kitchen Sink Visual Verification](./KITCHEN_SINK_VISUAL_VERIFICATION.md) - Visual testing procedures
- [Complete System Verification](./COMPLETE_SYSTEM_VERIFICATION_REPORT.md) - System-wide verification
- [Alternative Visual Verification](./ALTERNATIVE_VISUAL_VERIFICATION_REPORT.md) - Alternative testing methods
- [Kitchen Sink Enhancement Report](./KITCHEN_SINK_ENHANCEMENT_REPORT.md) - UI/UX improvements

### **Repository & Hub Integration**
- [HuggingFace Repository Integration](./HUGGINGFACE_REPOSITORY_INTEGRATION.md) - HF Hub integration
- [GitHub Workflow Fixes](./GITHUB_WORKFLOW_FIXES.md) - CI/CD improvements
- [CI/CD Updates Summary](./CI_CD_UPDATES_SUMMARY.md) - Workflow enhancements

### **Implementation Reports**
- [Implementation Completion Summary](./IMPLEMENTATION_COMPLETION_SUMMARY.md) - Final implementation status
- [Implementation Plan](./IMPLEMENTATION_PLAN.md) - Original planning document
- [Improvement Implementation Plan](./IMPROVEMENT_IMPLEMENTATION_PLAN.md) - Enhancement roadmap
- [Final Success Summary](./FINAL_SUCCESS_SUMMARY.md) - Project completion summary
- [Comprehensive Documentation Update Summary](./COMPREHENSIVE_DOCUMENTATION_UPDATE_SUMMARY.md) - Documentation updates

### **Publishing & Distribution**
- [PyPI Publishing Guide](./PYPI_PUBLISHING_GUIDE.md) - Package publishing instructions

## 🏗️ Project Organization

The documentation is organized alongside the main project structure:

```
ipfs_accelerate_py/
├── docs/                    # 📚 Documentation (this directory)
├── examples/               # 🎯 Example implementations
├── tests/                  # 🧪 Test suites
├── tools/                  # 🔧 Utility tools and scripts
├── ipfs_accelerate_py/     # 📦 Main package
└── README.md              # 📖 Main project documentation
```

## 🚀 Getting Started

1. Start with the [main README](../README.md) for an overview
2. Follow the [Installation Guide](./INSTALLATION_GUIDE.md) to set up the framework
3. Review the [Testing README](./TESTING_README.md) to understand the test suite
4. Explore specific implementation guides based on your needs

## 🔧 CLI Usage

The framework provides a comprehensive CLI tool accessible via `ipfs_accelerate`:

```bash
# Text processing
ipfs_accelerate text generate --prompt "Hello world"
ipfs_accelerate text classify --text "Great product!"

# Audio processing
ipfs_accelerate audio transcribe --audio-file speech.wav

# Vision processing  
ipfs_accelerate vision classify --image-file cat.jpg

# System information
ipfs_accelerate system list-models
```

See the [main README](../README.md) for complete CLI documentation.