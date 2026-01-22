# 🔧 IPFS Accelerate Python Tools & Utilities

This directory contains utility tools and scripts for development, testing, and system administration.

## 📂 Tools Overview

### **System Verification Tools**
- `comprehensive_system_verifier.py` - Complete system verification and validation
- `complete_system_verification.py` - System-wide verification checks
- `comprehensive_system_tester.py` - Comprehensive testing framework

### **AI and Model Management Tools**
- `comprehensive_mcp_server.py` - Complete MCP server implementation
- `huggingface_search_engine.py` - HuggingFace model search and discovery
- `model_manager_integration.py` - Model management integration tools

### **Kitchen Sink Testing Tools**
- `comprehensive_kitchen_sink_app.py` - Complete kitchen sink testing application
- `kitchen_sink_demo.py` - Kitchen sink demonstration interface
- `kitchen_sink_app.py` - Kitchen sink web application
- `kitchen_sink_pipeline_tester.py` - Pipeline testing utilities

### **Visual Testing Tools**
- `enhanced_visual_tester.py` - Enhanced visual testing framework
- `alternative_browser_tester.py` - Alternative browser testing methods
- `selenium_screenshot_tester.py` - Selenium-based screenshot testing
- `simple_screenshot_test.py` - Simple screenshot testing utilities

### **Dependency and Setup Tools**
- `comprehensive_dependency_installer.py` - Complete dependency management
- `dependency_installer.py` - Basic dependency installation
- `setup_environment.py` - Environment setup utilities
- `setup_package_structure.py` - Package structure setup
- `check_imports.py` - Import validation utilities

### **Documentation and Analysis Tools**
- `comprehensive_pipeline_documenter.py` - Pipeline documentation generator
- `final_screenshot_summary.py` - Screenshot analysis and summary
- `process_wheels.py` - Wheel processing utilities

### **SDK and Browser Tools**
- `sdk_browser_automation_test.py` - SDK browser automation
- `sdk_dashboard_app.py` - SDK dashboard application
- `sdk_dashboard_test.py` - SDK dashboard testing
- `sdk_demo.py` - SDK demonstration

### **Alternative Testing Tools**
- `alternative_visual_verifier.py` - Alternative visual verification
- `simple_server_test.py` - Simple server testing utilities

## 🚀 Using the Tools

### **System Verification**
```bash
# Complete system verification
python tools/comprehensive_system_verifier.py

# Basic system verification
python tools/complete_system_verification.py

# Comprehensive testing
python tools/comprehensive_system_tester.py
```

### **Model Management**
```bash
# Search HuggingFace models
python tools/huggingface_search_engine.py --query "bert" --task text-classification

# Run MCP server
python tools/comprehensive_mcp_server.py --port 8080

# Model manager integration
python tools/model_manager_integration.py --action list-models
```

### **Kitchen Sink Testing**
```bash
# Launch kitchen sink application
python tools/comprehensive_kitchen_sink_app.py

# Run kitchen sink demo
python tools/kitchen_sink_demo.py
# Open http://localhost:8080 in browser

# Test specific pipelines
python tools/kitchen_sink_pipeline_tester.py --pipeline text-generation
```

### **Visual Testing**
```bash
# Enhanced visual testing
python tools/enhanced_visual_tester.py --capture-screenshots

# Alternative browser testing
python tools/alternative_browser_tester.py --browser chrome

# Selenium screenshot testing
python tools/selenium_screenshot_tester.py --url http://localhost:8080
```

### **Dependency Management**
```bash
# Install all dependencies
python tools/comprehensive_dependency_installer.py

# Install basic dependencies
python tools/dependency_installer.py --minimal

# Check import issues
python tools/check_imports.py
```

### **SDK and Browser Automation**
```bash
# SDK browser automation
python tools/sdk_browser_automation_test.py

# SDK dashboard
python tools/sdk_dashboard_app.py
# Open http://localhost:8080 for dashboard

# SDK demonstration
python tools/sdk_demo.py
```

## 🎯 Tool Categories

### **🔍 Verification and Testing**
Tools for system verification and comprehensive testing:
- System-wide verification checks
- Component integration testing
- Performance validation
- Feature compatibility testing

### **🤖 AI and Model Tools**
Tools for AI model management and inference:
- Model discovery and search
- MCP server management
- Model compatibility assessment
- Performance optimization

### **🌐 Web Interface Tools**
Tools for web-based testing and demonstration:
- Kitchen sink testing interface
- Browser automation frameworks
- Visual testing and screenshots
- Interactive demonstrations

### **⚙️ Development Utilities**
Tools for development and maintenance:
- Dependency management
- Environment setup
- Package structure management
- Documentation generation

## 📋 Prerequisites

Before using tools, ensure you have:

1. **Installed the framework:**
   ```bash
   pip install ipfs_accelerate_py[tools]
   ```

2. **Required tool dependencies:**
   ```bash
   pip install ipfs_accelerate_py[full]  # For complete functionality
   pip install ipfs_accelerate_py[webnn] # For browser tools
   pip install ipfs_accelerate_py[mcp]   # For MCP tools
   ```

3. **Optional dependencies for specific tools:**
   ```bash
   pip install selenium playwright  # For visual testing tools
   pip install fastapi uvicorn      # For web-based tools
   ```

## 🔧 Configuration

Many tools support configuration through:

### **Command Line Arguments**
```bash
# Most tools support --help for options
python tools/[tool_name].py --help

# Common arguments:
--port 8080              # Specify port for web tools
--browser chrome         # Specify browser for testing
--output-dir results/    # Specify output directory
--verbose               # Enable detailed logging
```

### **Environment Variables**
```bash
export IPFS_ACCELERATE_PORT=8080
export IPFS_ACCELERATE_DEBUG=true
export IPFS_ACCELERATE_BROWSER=chrome
```

### **Configuration Files**
Some tools support configuration files (usually JSON or YAML):
```bash
python tools/comprehensive_system_verifier.py --config config.json
```

## 🐛 Troubleshooting Tools

If tools encounter issues:

1. **Check dependencies:**
   ```bash
   python tools/check_imports.py
   ```

2. **Install missing dependencies:**
   ```bash
   python tools/comprehensive_dependency_installer.py
   ```

3. **Verify system setup:**
   ```bash
   python tools/complete_system_verification.py
   ```

4. **Check browser setup (for visual tools):**
   ```bash
   python tools/simple_server_test.py
   ```

## 📊 Tool Output

Tools generate various types of output:

### **Reports and Logs**
- HTML reports with detailed analysis
- JSON data files for programmatic access
- Log files for debugging and monitoring

### **Screenshots and Media**
- PNG screenshots for visual testing
- Performance graphs and charts
- Interactive dashboards

### **Data Files**
- Database files for persistent storage
- CSV files for data analysis
- Configuration files for reproducibility

## 📚 Additional Resources

- [Documentation](../docs/) - Complete documentation
- [Examples](../examples/) - Example implementations
- [Tests](../tests/) - Test suite
- [Main README](../README.md) - Project overview
- [CLI Documentation](../README.md#cli-usage) - Command line interface