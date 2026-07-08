#!/usr/bin/env python3
"""
MCP Dashboard Coverage Analysis Script

This script analyzes the MCP server to identify all available tools and compare
them with what's exposed in the JavaScript SDK and dashboard.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def extract_mcp_tools() -> Dict[str, List[str]]:
    """Extract all MCP tools from the codebase."""
    tools_dir = Path(__file__).parent.parent / "ipfs_accelerate_py" / "mcp" / "tools"
    tools_by_file = {}
    
    for py_file in tools_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        content = py_file.read_text()
        
        # Find @mcp.tool decorators
        pattern = r'@mcp\.tool\([^)]*\)\s*(?:async\s+)?def\s+(\w+)'
        matches = re.findall(pattern, content)
        
        if matches:
            tools_by_file[py_file.name] = matches
    
    return tools_by_file

def extract_sdk_methods() -> List[str]:
    """Extract all SDK methods from mcp-sdk.js."""
    sdk_path = Path(__file__).parent.parent / "ipfs_accelerate_py" / "static" / "js" / "mcp-sdk.js"
    
    if not sdk_path.exists():
        return []
    
    content = sdk_path.read_text()
    
    # Find async methods
    pattern = r'async\s+(\w+)\s*\('
    matches = re.findall(pattern, content)
    
    # Filter out private methods
    methods = [m for m in matches if not m.startswith('_')]
    
    return sorted(set(methods))

def extract_unified_tools() -> List[str]:
    """Extract unified tools from unified_tools.py."""
    unified_path = Path(__file__).parent.parent / "ipfs_accelerate_py" / "mcp" / "unified_tools.py"
    
    if not unified_path.exists():
        return []
    
    content = unified_path.read_text()
    
    # Find function definitions for unified tools
    pattern = r'def\s+(github_|docker_|hardware_|runner_|ipfs_files_|network_)(\w+)'
    matches = re.findall(pattern, content)
    
    tools = [f"{prefix}{name}" for prefix, name in matches]
    
    return sorted(set(tools))

def categorize_tools(tools: List[str]) -> Dict[str, List[str]]:
    """Categorize tools by prefix/type."""
    categories = {
        'GitHub': [],
        'Docker': [],
        'Hardware': [],
        'Runner': [],
        'IPFS Files': [],
        'Network': [],
        'Models': [],
        'Inference': [],
        'Workflows': [],
        'Endpoints': [],
        'Status': [],
        'Dashboard': [],
        'System': [],
        'Other': []
    }
    
    for tool in tools:
        tool_lower = tool.lower()
        
        if 'github' in tool_lower:
            categories['GitHub'].append(tool)
        elif 'docker' in tool_lower:
            categories['Docker'].append(tool)
        elif 'hardware' in tool_lower:
            categories['Hardware'].append(tool)
        elif 'runner' in tool_lower or 'autoscaler' in tool_lower:
            categories['Runner'].append(tool)
        elif 'ipfs' in tool_lower or 'files' in tool_lower:
            categories['IPFS Files'].append(tool)
        elif 'network' in tool_lower or 'peer' in tool_lower or 'swarm' in tool_lower:
            categories['Network'].append(tool)
        elif 'model' in tool_lower or 'search' in tool_lower or 'recommend' in tool_lower:
            categories['Models'].append(tool)
        elif 'inference' in tool_lower or 'generate' in tool_lower or 'classify' in tool_lower:
            categories['Inference'].append(tool)
        elif 'workflow' in tool_lower:
            categories['Workflows'].append(tool)
        elif 'endpoint' in tool_lower:
            categories['Endpoints'].append(tool)
        elif 'status' in tool_lower or 'health' in tool_lower:
            categories['Status'].append(tool)
        elif 'dashboard' in tool_lower or 'stats' in tool_lower:
            categories['Dashboard'].append(tool)
        elif 'log' in tool_lower or 'system' in tool_lower:
            categories['System'].append(tool)
        else:
            categories['Other'].append(tool)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def main():
    """Main analysis function."""
    print("=" * 80)
    print("MCP Dashboard Coverage Analysis")
    print("=" * 80)
    print()
    
    # Extract MCP tools
    print("📦 Analyzing MCP Tools...")
    mcp_tools_by_file = extract_mcp_tools()
    all_mcp_tools = []
    for tools in mcp_tools_by_file.values():
        all_mcp_tools.extend(tools)
    all_mcp_tools = sorted(set(all_mcp_tools))
    
    print(f"   Found {len(all_mcp_tools)} MCP tools in {len(mcp_tools_by_file)} files")
    print()
    
    # Extract unified tools
    print("🔧 Analyzing Unified Tools...")
    unified_tools = extract_unified_tools()
    print(f"   Found {len(unified_tools)} unified tools")
    print()
    
    # Extract SDK methods
    print("📱 Analyzing JavaScript SDK...")
    sdk_methods = extract_sdk_methods()
    print(f"   Found {len(sdk_methods)} SDK methods")
    print()
    
    # Combine all tools
    all_tools = sorted(set(all_mcp_tools + unified_tools))
    
    # Categorize
    print("📊 Tool Categories:")
    print("-" * 80)
    categories = categorize_tools(all_tools)
    
    total_categorized = 0
    for category, tools in sorted(categories.items()):
        print(f"   {category:20s}: {len(tools):3d} tools")
        total_categorized += len(tools)
    
    print(f"   {'-'*20}   {'-'*8}")
    print(f"   {'TOTAL':20s}: {total_categorized:3d} tools")
    print()
    
    # Coverage analysis
    print("🎯 Coverage Analysis:")
    print("-" * 80)
    
    # Tools with SDK methods
    tools_with_sdk = []
    tools_without_sdk = []
    
    for tool in all_tools:
        # Check if tool has corresponding SDK method
        # Convert tool name to camelCase for SDK method name
        tool_parts = tool.replace('_', ' ').split()
        sdk_method = tool_parts[0].lower() + ''.join(word.capitalize() for word in tool_parts[1:])
        
        if sdk_method in sdk_methods:
            tools_with_sdk.append(tool)
        else:
            tools_without_sdk.append(tool)
    
    coverage_percent = (len(tools_with_sdk) / len(all_tools) * 100) if all_tools else 0
    
    print(f"   Total Tools:        {len(all_tools)}")
    print(f"   Tools with SDK:     {len(tools_with_sdk)} ({coverage_percent:.1f}%)")
    print(f"   Tools without SDK:  {len(tools_without_sdk)}")
    print()
    
    if tools_without_sdk:
        print("⚠️  Tools Missing SDK Methods:")
        print("-" * 80)
        for tool in sorted(tools_without_sdk)[:20]:  # Show first 20
            print(f"   - {tool}")
        if len(tools_without_sdk) > 20:
            print(f"   ... and {len(tools_without_sdk) - 20} more")
        print()
    
    # SDK methods without tools
    sdk_only = []
    for method in sdk_methods:
        # Check if this SDK method has a corresponding tool
        found = False
        for tool in all_tools:
            tool_parts = tool.replace('_', ' ').split()
            expected_method = tool_parts[0].lower() + ''.join(word.capitalize() for word in tool_parts[1:])
            if method == expected_method:
                found = True
                break
        
        if not found:
            # These might be utility methods
            if method not in ['request', 'notify', 'batch', 'callTool', 'callToolsBatch', 
                             'listMethods', 'getServerInfo', 'ping', 'waitForServer',
                             'getModel', 'addModel', 'listModels']:
                sdk_only.append(method)
    
    if sdk_only:
        print("ℹ️  SDK-Only Methods (may be wrappers or utilities):")
        print("-" * 80)
        for method in sorted(sdk_only)[:15]:
            print(f"   - {method}")
        print()
    
    # Summary
    print("=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"   Total MCP Tools:      {len(all_tools)}")
    print(f"   SDK Methods:          {len(sdk_methods)}")
    print(f"   Coverage:             {coverage_percent:.1f}%")
    print(f"   Unified Tools:        {len(unified_tools)}")
    print(f"   Tool Categories:      {len(categories)}")
    print()
    print("✅ Analysis complete!")
    print()
    
    # Save detailed report
    report = {
        'total_tools': len(all_tools),
        'total_sdk_methods': len(sdk_methods),
        'coverage_percent': coverage_percent,
        'categories': {k: len(v) for k, v in categories.items()},
        'tools_without_sdk': tools_without_sdk,
        'sdk_only_methods': sdk_only,
        'all_tools': all_tools,
        'sdk_methods': sdk_methods
    }
    
    report_path = Path(__file__).parent.parent / "mcp_coverage_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_path}")
    print()

if __name__ == "__main__":
    main()
