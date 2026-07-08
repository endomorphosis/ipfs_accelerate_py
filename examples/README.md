# 🎯 IPFS Accelerate Python Examples

This directory contains comprehensive examples and demonstrations of the IPFS Accelerate Python framework capabilities.

## 📂 Examples Overview

### **AI Implementation Examples**
- [AI Implementation Showcase](./ai_implementation_showcase.py) - Complete system demonstration
- [AI MCP Demo](./ai_mcp_demo.py) - MCP server demonstration with AI inference
- [AI Model Discovery Example](./ai_model_discovery_example.py) - Model discovery and search

### **Complete System Demonstrations**
- [Comprehensive AI Demo](./comprehensive_ai_demo.py) - Full-featured AI capabilities demo
- [Demonstration Example](./demonstration_example.py) - Basic usage demonstration
- [Model Manager Example](./model_manager_example.py) - Model management system demo

### **Kitchen Sink Applications**
- [Kitchen Sink Demo](./kitchen_sink_demo.py) - Web-based testing interface
- [Kitchen Sink App](./kitchen_sink_app.py) - Complete application example
- [SDK Demo](./sdk_demo.py) - JavaScript SDK demonstration

## 🚀 Running Examples

### **Basic Usage**
```bash
# AI implementation showcase
cd examples
python ai_implementation_showcase.py

# Complete AI demonstration  
python comprehensive_ai_demo.py

# Model manager example
python model_manager_example.py
```

### **Web-based Examples**
```bash
# Kitchen sink testing interface
python kitchen_sink_demo.py
# Open http://localhost:8080 in your browser

# SDK demonstration
python sdk_demo.py
# Demonstrates JavaScript SDK usage
```

### **MCP Server Examples**
```bash
# AI MCP server demonstration
python ai_mcp_demo.py
# Shows MCP server capabilities with AI inference

# Model discovery example
python ai_model_discovery_example.py
# Demonstrates model search and discovery
```

## 🎯 Example Categories

### **🤖 AI Inference Examples**
These examples demonstrate various AI inference capabilities:
- Text generation and classification
- Audio transcription and synthesis
- Image classification and generation
- Multimodal processing (image captioning, VQA)
- Specialized processing (code generation, time series)

### **🏗️ System Architecture Examples**
Examples showing system architecture and integration:
- MCP server setup and usage
- Model manager configuration
- Hardware acceleration examples
- IPFS integration demonstrations

### **🌐 Web Interface Examples**
Browser-based examples and interfaces:
- Kitchen Sink testing dashboard
- JavaScript SDK usage
- Real-time inference examples
- Interactive model exploration

## 📋 Prerequisites

Before running examples, ensure you have:

1. **Installed the framework:**
   ```bash
   pip install ipfs_accelerate_py
   ```

2. **Required dependencies:**
   ```bash
   pip install ipfs_accelerate_py[full]  # For complete functionality
   ```

3. **Optional dependencies for specific examples:**
   ```bash
   pip install ipfs_accelerate_py[webnn]  # For web-based examples
   pip install ipfs_accelerate_py[mcp]    # For MCP server examples
   ```

## 🔧 CLI Integration

All examples can be replicated using the CLI tool:

```bash
# Instead of running Python examples, use the CLI:
ipfs_accelerate text generate --prompt "Hello world"
ipfs_accelerate system list-models
ipfs_accelerate vision classify --image-file cat.jpg
```

See the [main README](../README.md) for complete CLI documentation.

## 📚 Additional Resources

- [Documentation](../docs/) - Complete documentation
- [Tests](../tests/) - Test suite examples
- [Scripts](../scripts/) - Utility tools and scripts
- [Main README](../README.md) - Project overview

## 🐛 Troubleshooting

If you encounter issues running examples:

1. Check the [Installation Troubleshooting Guide](../docs/INSTALLATION_TROUBLESHOOTING_GUIDE.md)
2. Ensure all dependencies are installed
3. Review example-specific comments for requirements
4. Check the [Testing README](../docs/TESTING_README.md) for debugging tips