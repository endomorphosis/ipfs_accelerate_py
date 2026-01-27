# IPFS Accelerate MCP Setup Guide

This guide will help you set up and use the IPFS Accelerate MCP (Model Context Protocol) servers with VS Code and the web dashboard.

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn fastmcp psutil numpy
```

### 2. Test the Servers

#### JSON-RPC Server with Web Dashboard
```bash
cd /path/to/ipfs_accelerate_py
python mcp_jsonrpc_server.py --port 8003
```

Then open your browser to: http://localhost:8003

#### VS Code MCP Integration
```bash
cd /path/to/ipfs_accelerate_py
python vscode_mcp_server.py
```

#### CLI Tools
```bash
# Test comprehensive server
python tools/comprehensive_mcp_server.py --help

# Test standalone server
python -m ipfs_accelerate_py.mcp.standalone --help
```

## VS Code Configuration

To use the MCP server with VS Code:

1. Install the MCP extension for VS Code
2. Add this configuration to your VS Code settings or MCP config file:

```json
{
  "mcpServers": {
    "ipfs-accelerate": {
      "command": "python",
      "args": ["/path/to/ipfs_accelerate_py/vscode_mcp_server.py"],
      "cwd": "/path/to/ipfs_accelerate_py"
    }
  }
}
```

Replace `/path/to/ipfs_accelerate_py` with the actual path to your repository.

## Web Dashboard Features

The web dashboard provides a comprehensive interface for testing AI models:

- **Text Generation**: Test language model text generation
- **Text Classification**: Classify text into categories
- **Text Embeddings**: Generate vector embeddings for text
- **Audio Processing**: Transcribe audio files
- **Vision Models**: Image classification and generation
- **Multimodal**: Visual question answering (coming soon)
- **Specialized Tools**: Code generation and more
- **Model Recommendations**: Get AI model suggestions
- **Model Manager**: Search and manage AI models

## Available CLI Commands

### JSON-RPC Server
```bash
python mcp_jsonrpc_server.py --help
python mcp_jsonrpc_server.py --port 8003 --verbose
```

### Comprehensive MCP Server
```bash
python tools/comprehensive_mcp_server.py --help
python tools/comprehensive_mcp_server.py --transport stdio  # For VS Code
python tools/comprehensive_mcp_server.py --transport sse --port 8004  # For HTTP
```

### Standalone Server
```bash
python -m ipfs_accelerate_py.mcp.standalone --help
python -m ipfs_accelerate_py.mcp.standalone --fastapi --port 8005
```

## API Testing

You can test the JSON-RPC API directly:

```bash
curl -X POST http://localhost:8003/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "list_models", "id": 1}'
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all required packages:
   ```bash
   pip install fastapi uvicorn fastmcp psutil numpy
   ```

2. **Port Already in Use**: Change the port number:
   ```bash
   python mcp_jsonrpc_server.py --port 8004
   ```

3. **VS Code Connection Issues**: 
   - Make sure the MCP extension is installed
   - Check that the path in the configuration is correct
   - Try running the server manually first to check for errors

### Verification

To verify everything is working:

1. Start the JSON-RPC server: `python mcp_jsonrpc_server.py --port 8003`
2. Open http://localhost:8003 in your browser
3. Test the "Text Generation" tab with a sample prompt
4. You should see a JSON response with generated text

## Support

If you encounter issues:

1. Check the console/terminal output for error messages
2. Verify all dependencies are installed
3. Make sure no other services are using the same ports
4. Check file paths in configuration files

## Features Status

✅ **Working Features:**
- JSON-RPC server with web dashboard
- All API endpoints (28 methods available)
- Static file serving
- CLI tools
- VS Code MCP wrapper
- Dashboard UI with multiple model categories

⚠️ **Known Limitations:**
- Some optional dependencies may not be available (DuckDB, IPFS, etc.)
- Models are currently mock/demo responses
- Full model integration requires additional setup

🔧 **In Progress:**
- Multimodal processing features
- Real model loading and inference
- Advanced hardware acceleration