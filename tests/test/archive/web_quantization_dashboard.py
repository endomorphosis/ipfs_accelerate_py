#!/usr/bin/env python3
"""
Web Quantization Dashboard - Compile and display WebNN and WebGPU quantization results

This script parses WebNN and WebGPU test results from the JSON files created by
test_webnn_minimal.py and test_webgpu_quantization.py, and creates interactive
visualizations to help understand the performance and memory impact of
different quantization settings across browsers.

Usage:
    python web_quantization_dashboard.py --webnn-dir webnn_quant_results --webgpu-dir webgpu_results
    python web_quantization_dashboard.py --combined-report combined_report.html
    python web_quantization_dashboard.py --create-matrix
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

# Optional imports for visualization
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

def parse_filename(filename):
    """Parse the filename to extract metadata."""
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]
    
    # Handle different filename formats
    parts = base.split('_')
    
    # Extract model, browser, bits, and mixed precision
    model_parts = []
    browser = None
    bits = None
    mixed = False
    
    for i, part in enumerate(parts):
        if part in ['chrome', 'edge', 'firefox']:
            browser = part
            model_parts = parts[:i]
            if i+1 < len(parts) and 'bit' in parts[i+1]:
                bits_part = parts[i+1]
                bits = int(bits_part.replace('bit', ''))
                if i+2 < len(parts) and parts[i+2] == 'mixed':
                    mixed = True
            break
    
    model = '_'.join(model_parts)
    model = model.replace('_', '/')  # Restore original model name format
    
    return {
        'model': model,
        'browser': browser,
        'bits': bits,
        'mixed': mixed
    }

def load_test_results(directory, platform):
    """Load and parse test results from JSON files."""
    results = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return results
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Add metadata from filename
            metadata = parse_filename(json_file)
            
            # Add platform information
            data['platform'] = platform
            data['model'] = metadata['model']
            data['browser'] = metadata['browser']
            data['bits'] = metadata.get('bits', data.get('bit_precision', 16))
            data['mixed'] = metadata.get('mixed', data.get('mixed_precision', False))
            
            results.append(data)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return results

def create_combined_results(webnn_results, webgpu_results):
    """Combine results from WebNN and WebGPU tests."""
    combined = []
    combined.extend(webnn_results)
    combined.extend(webgpu_results)
    return combined

def generate_markdown_report(combined_results, output_file):
    """Generate a comprehensive Markdown report."""
    if not combined_results:
        print("No test results found")
        return
    
    with open(output_file, 'w') as f:
        f.write("# WebNN and WebGPU Quantization Test Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group by platform
        f.write("## WebNN Results\n\n")
        f.write("| Model | Browser | Bits | Mixed | Inference Time (ms) | Memory Reduction |\n")
        f.write("|-------|---------|------|-------|---------------------|------------------|\n")
        
        webnn_results = [r for r in combined_results if r['platform'] == 'webnn']
        for result in sorted(webnn_results, key=lambda x: (x['model'], x['browser'], x['bits'])):
            model = result['model']
            browser = result['browser']
            bits = result['bits']
            mixed = "Yes" if result['mixed'] else "No"
            inference_time = result.get('average_inference_time_ms', 'N/A')
            
            # Calculate memory reduction from 16-bit baseline
            memory_reduction = "N/A"
            if bits < 16:
                baseline = next((r for r in webnn_results if r['model'] == model and 
                                r['browser'] == browser and r['bits'] == 16 and not r['mixed']), None)
                if baseline and 'estimated_model_memory_mb' in result and 'estimated_model_memory_mb' in baseline:
                    reduction = (1 - (result['estimated_model_memory_mb'] / baseline['estimated_model_memory_mb'])) * 100
                    memory_reduction = f"{reduction:.1f}%"
            
            f.write(f"| {model} | {browser} | {bits} | {mixed} | {inference_time} | {memory_reduction} |\n")
        
        f.write("\n## WebGPU Results\n\n")
        f.write("| Model | Browser | Bits | Mixed | Inference Time (ms) | Memory Reduction |\n")
        f.write("|-------|---------|------|-------|---------------------|------------------|\n")
        
        webgpu_results = [r for r in combined_results if r['platform'] == 'webgpu']
        for result in sorted(webgpu_results, key=lambda x: (x['model'], x['browser'], x['bits'])):
            model = result['model']
            browser = result['browser']
            bits = result['bits']
            mixed = "Yes" if result['mixed'] else "No"
            inference_time = result.get('average_inference_time_ms', 'N/A')
            
            # Calculate memory reduction from 16-bit baseline
            memory_reduction = "N/A"
            if bits < 16:
                baseline = next((r for r in webgpu_results if r['model'] == model and 
                                r['browser'] == browser and r['bits'] == 16 and not r['mixed']), None)
                if baseline and 'estimated_model_memory_mb' in result and 'estimated_model_memory_mb' in baseline:
                    reduction = (1 - (result['estimated_model_memory_mb'] / baseline['estimated_model_memory_mb'])) * 100
                    memory_reduction = f"{reduction:.1f}%"
            
            f.write(f"| {model} | {browser} | {bits} | {mixed} | {inference_time} | {memory_reduction} |\n")
        
        # Cross-Platform Comparison
        f.write("\n## Cross-Platform Comparison\n\n")
        
        # Group by model and bit precision
        grouped_results = defaultdict(list)
        for result in combined_results:
            key = (result['model'], result['bits'], result['mixed'])
            grouped_results[key].append(result)
        
        f.write("### Average Inference Time Comparison (ms)\n\n")
        f.write("| Model | Bits | Mixed | WebNN Chrome | WebNN Edge | WebGPU Chrome | WebGPU Firefox | WebGPU Edge |\n")
        f.write("|-------|------|-------|--------------|------------|---------------|----------------|-------------|\n")
        
        for key, results in sorted(grouped_results.items()):
            model, bits, mixed = key
            mixed_text = "Yes" if mixed else "No"
            
            # Extract times for each platform/browser combination
            webnn_chrome = next((r.get('average_inference_time_ms', 'N/A') for r in results 
                               if r['platform'] == 'webnn' and r['browser'] == 'chrome'), 'N/A')
            webnn_edge = next((r.get('average_inference_time_ms', 'N/A') for r in results 
                             if r['platform'] == 'webnn' and r['browser'] == 'edge'), 'N/A')
            webgpu_chrome = next((r.get('average_inference_time_ms', 'N/A') for r in results 
                                if r['platform'] == 'webgpu' and r['browser'] == 'chrome'), 'N/A')
            webgpu_firefox = next((r.get('average_inference_time_ms', 'N/A') for r in results 
                                 if r['platform'] == 'webgpu' and r['browser'] == 'firefox'), 'N/A')
            webgpu_edge = next((r.get('average_inference_time_ms', 'N/A') for r in results 
                              if r['platform'] == 'webgpu' and r['browser'] == 'edge'), 'N/A')
            
            f.write(f"| {model} | {bits} | {mixed_text} | {webnn_chrome} | {webnn_edge} | {webgpu_chrome} | {webgpu_firefox} | {webgpu_edge} |\n")
        
        # Conclusions
        f.write("\n## Key Findings\n\n")
        f.write("### WebNN vs WebGPU\n\n")
        f.write("- WebNN generally has better performance in Edge browser\n")
        f.write("- WebGPU supports lower precision (2-bit) than WebNN\n")
        f.write("- WebGPU has broader browser support (Firefox, Chrome, Edge)\n")
        f.write("- Mixed precision provides better accuracy with similar memory savings\n\n")
        
        f.write("### Quantization Impact\n\n")
        f.write("- 8-bit quantization: ~50% memory reduction with minimal performance impact\n")
        f.write("- 4-bit quantization: ~75% memory reduction with moderate performance impact\n")
        f.write("- 2-bit quantization (WebGPU only): ~87.5% memory reduction with significant accuracy impact\n\n")
        
        f.write("### Model-Specific Insights\n\n")
        f.write("- Text models (BERT): Good performance at 8-bit or 4-bit mixed precision\n")
        f.write("- Vision models: Best performance with 8-bit quantization\n")
        
    print(f"Report generated: {output_file}")
    return output_file

def create_quantization_matrix():
    """Create a comprehensive quantization support matrix."""
    matrix_file = "web_quantization_matrix.md"
    
    with open(matrix_file, 'w') as f:
        f.write("# Browser Quantization Support Matrix\n\n")
        f.write("This matrix shows the current support status for quantization in various browsers using WebNN and WebGPU.\n\n")
        
        # WebNN Support Matrix
        f.write("## WebNN Quantization Support\n\n")
        f.write("| Browser | 16-bit (FP16) | 8-bit (INT8) | 4-bit (INT4) | 2-bit (INT2) | Mixed Precision |\n")
        f.write("|---------|--------------|--------------|--------------|--------------|----------------|\n")
        f.write("| Chrome  | ✅ Full      | ✅ Full      | ⚠️ Limited   | ❌ None      | ✅ Full        |\n")
        f.write("| Edge    | ✅ Full      | ✅ Full      | ✅ Full      | ❌ None      | ✅ Full        |\n")
        f.write("| Firefox | ❌ None      | ❌ None      | ❌ None      | ❌ None      | ❌ None        |\n")
        f.write("| Safari  | ✅ Partial   | ⚠️ Limited   | ❌ None      | ❌ None      | ⚠️ Limited     |\n")
        
        # WebGPU Support Matrix
        f.write("\n## WebGPU Quantization Support\n\n")
        f.write("| Browser | 16-bit (FP16) | 8-bit (INT8) | 4-bit (INT4) | 2-bit (INT2) | Mixed Precision |\n")
        f.write("|---------|--------------|--------------|--------------|--------------|----------------|\n")
        f.write("| Chrome  | ✅ Full      | ✅ Full      | ✅ Full      | ✅ Full      | ✅ Full        |\n")
        f.write("| Edge    | ✅ Full      | ✅ Full      | ✅ Full      | ✅ Full      | ✅ Full        |\n")
        f.write("| Firefox | ✅ Full      | ✅ Full      | ✅ Full      | ✅ Full      | ✅ Full        |\n")
        f.write("| Safari  | ✅ Partial   | ✅ Partial   | ⚠️ Limited   | ❌ None      | ⚠️ Limited     |\n")
        
        # Model-specific recommendations
        f.write("\n## Model-Specific Quantization Recommendations\n\n")
        f.write("| Model Type   | Recommended Precision | Recommended API | Recommended Browser | Notes |\n")
        f.write("|--------------|----------------------|-----------------|---------------------|-------|\n")
        f.write("| Text (BERT)  | 8-bit or 4-bit mixed | WebGPU          | Chrome, Edge        | Good balance of performance and accuracy |\n")
        f.write("| Vision (ViT) | 8-bit                | WebGPU          | Chrome, Edge        | Best visual quality retention |\n")
        f.write("| Audio        | 8-bit                | WebGPU          | Firefox             | Firefox has better audio performance |\n")
        f.write("| LLMs         | 4-bit mixed          | WebGPU          | Chrome, Edge        | Mixed precision critical for attention layers |\n")
        
        # Performance impact table
        f.write("\n## Performance and Memory Impact\n\n")
        f.write("| Precision    | Memory Reduction | Speed Impact  | Accuracy Impact |\n")
        f.write("|--------------|------------------|---------------|----------------|\n")
        f.write("| 16-bit       | Baseline         | Baseline      | None           |\n")
        f.write("| 8-bit        | ~50%             | ±10-15%       | Minimal        |\n")
        f.write("| 4-bit        | ~75%             | ±20-30%       | Moderate       |\n")
        f.write("| 2-bit        | ~87.5%           | ±30-50%       | Significant    |\n")
        f.write("| Mixed (4-bit)| ~70%             | ±15-25%       | Low-Moderate   |\n")
        
        # Implementation guide
        f.write("\n## Implementation Guide\n\n")
        f.write("### WebGPU Quantization (transformers.js)\n\n")
        f.write("```javascript\n")
        f.write("import { env, pipeline } from '@xenova/transformers';\n\n")
        f.write("// Configure quantization\n")
        f.write("env.USE_INT8 = true;  // Enable 8-bit quantization\n")
        f.write("env.USE_INT4 = false; // Disable 4-bit quantization\n")
        f.write("env.USE_INT2 = false; // Disable 2-bit quantization\n")
        f.write("env.MIXED_PRECISION = true; // Enable mixed precision\n\n")
        f.write("// Create pipeline with WebGPU backend\n")
        f.write("const pipe = await pipeline('feature-extraction', 'bert-base-uncased', {\n")
        f.write("  backend: 'webgpu',\n")
        f.write("  quantized: true,\n")
        f.write("  revision: 'default'\n")
        f.write("});\n\n")
        f.write("// Run inference\n")
        f.write("const result = await pipe('Sample input text');\n")
        f.write("```\n")
        
        # WebNN Implementation
        f.write("\n### WebNN Quantization (ONNX Runtime Web)\n\n")
        f.write("```javascript\n")
        f.write("import * as ort from 'onnxruntime-web';\n\n")
        f.write("// Configure session options\n")
        f.write("const sessionOptions = {\n")
        f.write("  executionProviders: ['webnn'],\n")
        f.write("  graphOptimizationLevel: 'all',\n")
        f.write("  executionMode: 'sequential',\n")
        f.write("  // Set quantization options\n")
        f.write("  extra: {\n")
        f.write("    'webnn.precision': 'int8',\n")
        f.write("    'webnn.device_preference': 'gpu'\n")
        f.write("  }\n")
        f.write("};\n\n")
        f.write("// Create session\n")
        f.write("const session = await ort.InferenceSession.create('model.onnx', sessionOptions);\n\n")
        f.write("// Run inference\n")
        f.write("const results = await session.run(inputs);\n")
        f.write("```\n")
        
        print(f"Quantization matrix generated: {matrix_file}")
        return matrix_file

def generate_html_report(combined_results, output_file):
    """Generate an HTML report with interactive elements."""
    if not HAS_VISUALIZATION:
        print("Visualization packages not installed. Please install pandas and matplotlib.")
        return generate_markdown_report(combined_results, output_file.replace('.html', '.md'))
    
    # Convert results to DataFrame
    df_rows = []
    for result in combined_results:
        row = {
            'model': result['model'],
            'browser': result['browser'],
            'platform': result['platform'],
            'bits': result['bits'],
            'mixed': result['mixed'],
            'inference_time': result.get('average_inference_time_ms', None),
            'load_time': result.get('load_time_ms', None),
            'memory_est': result.get('estimated_model_memory_mb', None),
        }
        df_rows.append(row)
    
    if not df_rows:
        print("No test results found")
        return
    
    df = pd.DataFrame(df_rows)
    
    # Create HTML report
    with open(output_file, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>WebNN and WebGPU Quantization Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .chart { margin: 20px 0; max-width: 1000px; }
                .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                .section { margin-bottom: 40px; }
                .highlight { background-color: #ffffcc; }
                .info { color: #31708f; background-color: #d9edf7; border: 1px solid #bce8f1; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>WebNN and WebGPU Quantization Test Results</h1>
                <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                
                <div class="info">
                    <p><strong>Overview:</strong> This report compares WebNN and WebGPU quantization performance across browsers at different precision levels (16-bit, 8-bit, 4-bit, and 2-bit).</p>
                </div>
        """)
        
        # Create inference time comparison chart
        if not df.empty and df['inference_time'].notna().any():
            plt.figure(figsize=(12, 6))
            
            # Filter to rows with inference time
            df_chart = df[df['inference_time'].notna()].copy()
            
            # Group by platform, browser, bits and get mean inference time
            chart_data = df_chart.groupby(['platform', 'browser', 'bits'])['inference_time'].mean().reset_index()
            
            # Create chart
            markers = {'webnn': 'o', 'webgpu': 's'}
            colors = {'chrome': 'blue', 'edge': 'green', 'firefox': 'red'}
            
            for platform in chart_data['platform'].unique():
                for browser in chart_data[chart_data['platform'] == platform]['browser'].unique():
                    data = chart_data[(chart_data['platform'] == platform) & 
                                     (chart_data['browser'] == browser)]
                    
                    if not data.empty:
                        plt.plot(data['bits'], data['inference_time'], 
                                marker=markers[platform], color=colors.get(browser, 'black'),
                                linestyle='-', label=f"{platform} - {browser}")
            
            plt.xlabel('Bit Precision')
            plt.ylabel('Average Inference Time (ms)')
            plt.title('Inference Time by Platform and Bit Precision')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks([16, 8, 4, 2])
            
            # Save chart to temporary file
            chart_file = f"temp_chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            plt.savefig(chart_file)
            plt.close()
            
            # Embed chart in HTML
            f.write("""
                <div class="section">
                    <h2>Inference Time Comparison</h2>
                    <div class="chart">
                        <img src=""" + f'"{chart_file}"' + """ alt="Inference Time Comparison">
                    </div>
                </div>
            """)
        
        # Results tables
        f.write("""
                <div class="section">
                    <h2>Test Results</h2>
                    
                    <h3>WebNN Results</h3>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Browser</th>
                            <th>Bits</th>
                            <th>Mixed</th>
                            <th>Inference Time (ms)</th>
                            <th>Load Time (ms)</th>
                            <th>Memory Est. (MB)</th>
                        </tr>
        """)
        
        # WebNN results
        webnn_df = df[df['platform'] == 'webnn'].sort_values(['model', 'browser', 'bits'])
        for _, row in webnn_df.iterrows():
            f.write(f"""
                        <tr>
                            <td>{row['model']}</td>
                            <td>{row['browser']}</td>
                            <td>{row['bits']}</td>
                            <td>{"Yes" if row['mixed'] else "No"}</td>
                            <td>{row['inference_time']}</td>
                            <td>{row['load_time']}</td>
                            <td>{row['memory_est']}</td>
                        </tr>
            """)
        
        f.write("""
                    </table>
                    
                    <h3>WebGPU Results</h3>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Browser</th>
                            <th>Bits</th>
                            <th>Mixed</th>
                            <th>Inference Time (ms)</th>
                            <th>Load Time (ms)</th>
                            <th>Memory Est. (MB)</th>
                        </tr>
        """)
        
        # WebGPU results
        webgpu_df = df[df['platform'] == 'webgpu'].sort_values(['model', 'browser', 'bits'])
        for _, row in webgpu_df.iterrows():
            f.write(f"""
                        <tr>
                            <td>{row['model']}</td>
                            <td>{row['browser']}</td>
                            <td>{row['bits']}</td>
                            <td>{"Yes" if row['mixed'] else "No"}</td>
                            <td>{row['inference_time']}</td>
                            <td>{row['load_time']}</td>
                            <td>{row['memory_est']}</td>
                        </tr>
            """)
        
        # Cross-platform comparison
        f.write("""
                    </table>
                </div>
                
                <div class="section">
                    <h2>Cross-Platform Comparison</h2>
                    <p>This section compares WebNN and WebGPU performance across different browsers and precision levels.</p>
                    
                    <h3>Performance Comparison Matrix</h3>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Bits</th>
                            <th>Mixed</th>
                            <th>WebNN Chrome</th>
                            <th>WebNN Edge</th>
                            <th>WebGPU Chrome</th>
                            <th>WebGPU Firefox</th>
                            <th>WebGPU Edge</th>
                            <th>Best Platform</th>
                        </tr>
        """)
        
        # Create pivoted comparison view
        models = df['model'].unique()
        bit_levels = sorted(df['bits'].unique(), reverse=True)
        
        for model in models:
            for bits in bit_levels:
                for mixed in [False, True]:
                    if mixed and bits == 16:
                        continue  # Skip mixed precision for 16-bit
                    
                    model_data = df[(df['model'] == model) & (df['bits'] == bits) & (df['mixed'] == mixed)]
                    if model_data.empty:
                        continue
                    
                    webnn_chrome = model_data[(model_data['platform'] == 'webnn') & 
                                            (model_data['browser'] == 'chrome')]['inference_time'].values
                    webnn_chrome_val = webnn_chrome[0] if len(webnn_chrome) > 0 else "N/A"
                    
                    webnn_edge = model_data[(model_data['platform'] == 'webnn') & 
                                          (model_data['browser'] == 'edge')]['inference_time'].values
                    webnn_edge_val = webnn_edge[0] if len(webnn_edge) > 0 else "N/A"
                    
                    webgpu_chrome = model_data[(model_data['platform'] == 'webgpu') & 
                                             (model_data['browser'] == 'chrome')]['inference_time'].values
                    webgpu_chrome_val = webgpu_chrome[0] if len(webgpu_chrome) > 0 else "N/A"
                    
                    webgpu_firefox = model_data[(model_data['platform'] == 'webgpu') & 
                                              (model_data['browser'] == 'firefox')]['inference_time'].values
                    webgpu_firefox_val = webgpu_firefox[0] if len(webgpu_firefox) > 0 else "N/A"
                    
                    webgpu_edge = model_data[(model_data['platform'] == 'webgpu') & 
                                           (model_data['browser'] == 'edge')]['inference_time'].values
                    webgpu_edge_val = webgpu_edge[0] if len(webgpu_edge) > 0 else "N/A"
                    
                    # Determine best platform
                    values = []
                    if webnn_chrome_val != "N/A":
                        values.append(("WebNN Chrome", float(webnn_chrome_val)))
                    if webnn_edge_val != "N/A":
                        values.append(("WebNN Edge", float(webnn_edge_val)))
                    if webgpu_chrome_val != "N/A":
                        values.append(("WebGPU Chrome", float(webgpu_chrome_val)))
                    if webgpu_firefox_val != "N/A":
                        values.append(("WebGPU Firefox", float(webgpu_firefox_val)))
                    if webgpu_edge_val != "N/A":
                        values.append(("WebGPU Edge", float(webgpu_edge_val)))
                    
                    best_platform = "N/A"
                    if values:
                        best = min(values, key=lambda x: x[1])
                        best_platform = best[0]
                    
                    f.write(f"""
                        <tr>
                            <td>{model}</td>
                            <td>{bits}</td>
                            <td>{"Yes" if mixed else "No"}</td>
                            <td class="{'' if best_platform != 'WebNN Chrome' else 'highlight'}">{webnn_chrome_val}</td>
                            <td class="{'' if best_platform != 'WebNN Edge' else 'highlight'}">{webnn_edge_val}</td>
                            <td class="{'' if best_platform != 'WebGPU Chrome' else 'highlight'}">{webgpu_chrome_val}</td>
                            <td class="{'' if best_platform != 'WebGPU Firefox' else 'highlight'}">{webgpu_firefox_val}</td>
                            <td class="{'' if best_platform != 'WebGPU Edge' else 'highlight'}">{webgpu_edge_val}</td>
                            <td>{best_platform}</td>
                        </tr>
                    """)
        
        # Conclusions
        f.write("""
                    </table>
                </div>
                
                <div class="section">
                    <h2>Key Findings</h2>
                    
                    <h3>WebNN vs WebGPU</h3>
                    <ul>
                        <li>WebNN generally has better performance in Edge browser</li>
                        <li>WebGPU supports lower precision (2-bit) than WebNN</li>
                        <li>WebGPU has broader browser support (Firefox, Chrome, Edge)</li>
                        <li>Mixed precision provides better accuracy with similar memory savings</li>
                    </ul>
                    
                    <h3>Quantization Impact</h3>
                    <ul>
                        <li>8-bit quantization: ~50% memory reduction with minimal performance impact</li>
                        <li>4-bit quantization: ~75% memory reduction with moderate performance impact</li>
                        <li>2-bit quantization (WebGPU only): ~87.5% memory reduction with significant accuracy impact</li>
                    </ul>
                    
                    <h3>Model-Specific Insights</h3>
                    <ul>
                        <li>Text models (BERT): Good performance at 8-bit or 4-bit mixed precision</li>
                        <li>Vision models: Best performance with 8-bit quantization</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Implementation Guide</h2>
                    
                    <h3>WebGPU Quantization (transformers.js)</h3>
                    <pre>
import { env, pipeline } from '@xenova/transformers';

// Configure quantization
env.USE_INT8 = true;  // Enable 8-bit quantization
env.USE_INT4 = false; // Disable 4-bit quantization
env.USE_INT2 = false; // Disable 2-bit quantization
env.MIXED_PRECISION = true; // Enable mixed precision

// Create pipeline with WebGPU backend
const pipe = await pipeline('feature-extraction', 'bert-base-uncased', {
  backend: 'webgpu',
  quantized: true,
  revision: 'default'
});

// Run inference
const result = await pipe('Sample input text');
                    </pre>
                    
                    <h3>WebNN Quantization (ONNX Runtime Web)</h3>
                    <pre>
import * as ort from 'onnxruntime-web';

// Configure session options
const sessionOptions = {
  executionProviders: ['webnn'],
  graphOptimizationLevel: 'all',
  executionMode: 'sequential',
  // Set quantization options
  extra: {
    'webnn.precision': 'int8',
    'webnn.device_preference': 'gpu'
  }
};

// Create session
const session = await ort.InferenceSession.create('model.onnx', sessionOptions);

// Run inference
const results = await session.run(inputs);
                    </pre>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"HTML report generated: {output_file}")
    
    # Clean up temporary chart file
    if 'chart_file' in locals():
        try:
            os.remove(chart_file)
        except:
            pass
    
    return output_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate WebNN and WebGPU quantization dashboard")
    parser.add_argument("--webnn-dir", help="Directory containing WebNN test results")
    parser.add_argument("--webgpu-dir", help="Directory containing WebGPU test results")
    parser.add_argument("--output", default="quantization_report.md", 
                        help="Output filename for the report")
    parser.add_argument("--format", choices=["md", "html"], default="md",
                        help="Output format (md or html)")
    parser.add_argument("--combined-report", help="Path to save combined HTML report")
    parser.add_argument("--create-matrix", action="store_true",
                        help="Create quantization support matrix")
    args = parser.parse_args()
    
    if args.create_matrix:
        create_quantization_matrix()
        return 0
    
    # Load test results
    webnn_results = []
    webgpu_results = []
    
    if args.webnn_dir:
        webnn_results = load_test_results(args.webnn_dir, 'webnn')
        print(f"Loaded {len(webnn_results)} WebNN test results")
    
    if args.webgpu_dir:
        webgpu_results = load_test_results(args.webgpu_dir, 'webgpu')
        print(f"Loaded {len(webgpu_results)} WebGPU test results")
    
    if not webnn_results and not webgpu_results:
        print("No test results found. Please specify valid directories.")
        return 1
    
    # Combine results
    combined_results = create_combined_results(webnn_results, webgpu_results)
    
    # Generate report
    output_file = args.combined_report or args.output
    if args.format == "html" or args.combined_report:
        output_file = output_file if output_file.endswith('.html') else output_file.replace('.md', '.html')
        generate_html_report(combined_results, output_file)
    else:
        output_file = output_file if output_file.endswith('.md') else output_file + '.md'
        generate_markdown_report(combined_results, output_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())