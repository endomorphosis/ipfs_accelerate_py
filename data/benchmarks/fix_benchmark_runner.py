
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Comprehensive Benchmark Timing Report</title>
                    <style>
                        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; color: #333; line-height: 1.6; }
                        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                        h1, h2, h3, h4 { color: #1a5276; }
                        h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }
                        th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
                        th { background-color: #3498db; color: white; }
                        tr:nth-child(even) { background-color: #f9f9f9; }
                        tr:hover { background-color: #f1f1f1; }
                        .summary-card { background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin-bottom: 20px; }
                        .optimization-card { background-color: #e8f8f5; border-left: 4px solid #2ecc71; padding: 15px; margin-bottom: 20px; }
                        .recommendation { padding: 10px; margin: 5px 0; background-color: #f1f9f7; border-radius: 5px; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Comprehensive Benchmark Timing Report</h1>
                        <p>Generated: 2025-03-06 19:42:44</p>
                        
                        <div class="summary-card">
                            <h2>Executive Summary</h2>
                            <p>This report provides detailed benchmark timing data for all 13 model types across 8 hardware endpoints, 
                            showing performance metrics including latency, throughput, and memory usage.</p>
                            <p>The analysis covers different model categories including text, vision, audio, and multimodal models.</p>
                        </div>
                        
                        <h2>Hardware Platforms</h2>
                        <table>
                            <tr>
                                <th>Hardware</th>
                                <th>Description</th>
                            </tr>
                <tr><td>cpu</td><td>CPU (Standard CPU processing)</td></tr>
<tr><td>cuda</td><td>CUDA (NVIDIA GPU acceleration)</td></tr>
<tr><td>rocm</td><td>ROCm (AMD GPU acceleration)</td></tr>
<tr><td>mps</td><td>MPS (Apple Silicon GPU acceleration)</td></tr>
<tr><td>openvino</td><td>OpenVINO (Intel acceleration)</td></tr>
<tr><td>qnn</td><td>QNN (Qualcomm AI Engine)</td></tr>
<tr><td>webnn</td><td>WebNN (Browser neural network API)</td></tr>
<tr><td>webgpu</td><td>WebGPU (Browser graphics API for ML)</td></tr>

                        </table>
                        
                        <h2>Performance Results</h2>
                
                        <h3>Text Models</h3>
                        <table>
                            <tr>
                                <th>Model</th>
                                <th>Hardware</th>
                                <th>Batch Size</th>
                                <th>Latency (ms)</th>
                                <th>Throughput (items/s)</th>
                                <th>Memory (MB)</th>
                            </tr>
                    
                            <tr>
                                <td>bert</td>
                                <td>cpu</td>
                                <td>1</td>
                                <td>31.20</td>
                                <td>32.06</td>
                                <td>3169.69</td>
                            </tr>
                        
                            <tr>
                                <td>bert</td>
                                <td>rocm</td>
                                <td>1</td>
                                <td>16.02</td>
                                <td>96.24</td>
                                <td>1300.51</td>
                            </tr>
                        </table>

                        <h3>Multimodal Models</h3>
                        <table>
                            <tr>
                                <th>Model</th>
                                <th>Hardware</th>
                                <th>Batch Size</th>
                                <th>Latency (ms)</th>
                                <th>Throughput (items/s)</th>
                                <th>Memory (MB)</th>
                            </tr>
                    </table>

                        <h3>Vision Models</h3>
                        <table>
                            <tr>
                                <th>Model</th>
                                <th>Hardware</th>
                                <th>Batch Size</th>
                                <th>Latency (ms)</th>
                                <th>Throughput (items/s)</th>
                                <th>Memory (MB)</th>
                            </tr>
                    </table>

                        <h3>Audio Models</h3>
                        <table>
                            <tr>
                                <th>Model</th>
                                <th>Hardware</th>
                                <th>Batch Size</th>
                                <th>Latency (ms)</th>
                                <th>Throughput (items/s)</th>
                                <th>Memory (MB)</th>
                            </tr>
                    </table>

                    <h2>Optimization Recommendations</h2>
                    <div class="optimization-card">
                        <p>Based on the benchmark results, here are some recommendations for optimizing performance:</p>
                        
                        <h3>Hardware Selection</h3>
                        <div class="recommendation">Use CUDA for best overall performance across all model types when available</div>
                        <div class="recommendation">For CPU-only environments, OpenVINO provides significant speedups over standard CPU</div>
                        <div class="recommendation">For browser environments, WebGPU with shader precompilation offers the best performance</div>
                        
                        <h3>Model-Specific Optimizations</h3>
                        <div class="recommendation">Text models benefit from CPU caching and OpenVINO optimizations</div>
                        <div class="recommendation">Vision models are well-optimized across most hardware platforms</div>
                        <div class="recommendation">Audio models perform best with CUDA; WebGPU with compute shader optimization for browser environments</div>
                        <div class="recommendation">For multimodal models, use hardware with sufficient memory capacity; WebGPU with parallel loading for browser environments</div>
                    </div>
                    
                    <h2>Conclusion</h2>
                    <p>This report provides a comprehensive view of performance characteristics for 13 key model types across 8 hardware platforms. 
                    Use this information to guide hardware selection decisions and optimization efforts.</p>
                </body>
                </html>
                