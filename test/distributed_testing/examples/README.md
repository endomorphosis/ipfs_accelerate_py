# Distributed Testing Framework - Examples

This directory contains example scripts that demonstrate the usage of the Distributed Testing Framework and its plugins.

## Available Examples

### Plugin Example

**File**: `plugin_example.py`

This script demonstrates how to use the plugin architecture of the Distributed Testing Framework:

- Creating and initializing a coordinator with plugin support
- Discovering and loading plugins
- Configuring plugins with specific settings
- Handling events through the plugin system
- Creating a custom notification plugin
- Simulating worker registration and task execution

### Running the Example

```bash
# Make sure you're in the distributed_testing directory
cd /path/to/distributed_testing

# Run the plugin example
python examples/plugin_example.py
```

## Creating Your Own Examples

Feel free to create additional examples in this directory to demonstrate specific aspects of the framework. When creating examples:

1. Use clear, descriptive file names
2. Include comprehensive docstrings explaining the purpose
3. Add comments to explain key sections of code
4. Handle errors gracefully
5. Include logging for better understanding

## Documentation

For more information, refer to:

- [Integration and Extensibility Guide](../INTEGRATION_EXTENSIBILITY_GUIDE.md): Comprehensive guide to the plugin architecture
- [Distributed Testing Design](../DISTRIBUTED_TESTING_DESIGN.md): Overview of the framework architecture
- [Integration Plugins README](../integration/README.md): Description of available integration plugins