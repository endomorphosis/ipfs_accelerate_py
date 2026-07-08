#!/usr/bin/env python3
"""
Test the unified MCP dashboard
"""

import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import time

class TestDashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>IPFS Accelerate MCP Dashboard - Test</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: rgba(255,255,255,0.2); border: none; color: white; cursor: pointer; border-radius: 5px; }
        .tab.active { background: rgba(255,255,255,0.4); }
        .content { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: rgba(255,255,255,0.2); border-radius: 8px; }
        .btn { padding: 10px 20px; background: #10b981; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #059669; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 IPFS Accelerate MCP Dashboard</h1>
            <p>Unified AI Model Inference Platform with MCP Protocol Integration</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">📊 Overview</button>
            <button class="tab" onclick="showTab('inference')">🤖 Text Generation</button>
            <button class="tab" onclick="showTab('models')">🎯 Model Manager</button>
            <button class="tab" onclick="showTab('testing')">🧪 Model Testing</button>
            <button class="tab" onclick="showTab('queue')">📈 Queue Monitor</button>
        </div>
        
        <div class="content">
            <div id="overview" class="tab-content">
                <h2>System Overview</h2>
                <div class="metric">⏱️ Uptime: <span id="uptime">8s</span></div>
                <div class="metric">🖥️ Active Endpoints: 4</div>
                <div class="metric">📋 Queue Size: 8</div>
                <div class="metric">⚡ Processing Tasks: 3</div>
                <p>MCP server is running and ready to accept tool calls via JavaScript SDK.</p>
            </div>
            
            <div id="inference" class="tab-content" style="display: none;">
                <h2>Text Generation</h2>
                <textarea id="promptInput" placeholder="Enter your prompt..." style="width: 100%; height: 100px; margin-bottom: 10px;"></textarea>
                <br>
                <button class="btn" onclick="generateText()">✨ Generate Text</button>
                <div id="textResult" style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 5px; display: none;"></div>
            </div>
            
            <div id="models" class="tab-content" style="display: none;">
                <h2>HuggingFace Model Manager</h2>
                <input type="text" id="searchInput" placeholder="Search models..." style="width: 70%; padding: 10px; margin-right: 10px;">
                <button class="btn" onclick="searchModels()">🔍 Search</button>
                <div id="modelResults" style="margin-top: 20px;"></div>
            </div>
            
            <div id="testing" class="tab-content" style="display: none;">
                <h2>Model Testing & Validation</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px;">
                        <h3>📝 Text Generation</h3>
                        <button class="btn" onclick="runTest('text-generation', 'creative-writing')">Creative Writing Test</button>
                        <button class="btn" onclick="runTest('text-generation', 'conversation')">Conversation Test</button>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 8px;">
                        <h3>🔍 Classification</h3>
                        <button class="btn" onclick="runTest('classification', 'sentiment')">Sentiment Analysis</button>
                        <button class="btn" onclick="runTest('classification', 'topic')">Topic Classification</button>
                    </div>
                </div>
                <div id="testResults" style="margin-top: 20px; display: none;"></div>
            </div>
            
            <div id="queue" class="tab-content" style="display: none;">
                <h2>Queue Monitor</h2>
                <button class="btn" onclick="getQueueStatus()">📊 View Queue Status</button>
                <button class="btn" onclick="getModelQueues()">🎯 Model Queues</button>
                <div id="queueResults" style="margin-top: 20px;"></div>
            </div>
        </div>
    </div>
    
    <script>
        // MCP Client Mock for Testing
        class MCPClient {
            async callTool(toolName, params = {}) {
                // Simulate MCP tool calls with realistic responses
                await new Promise(resolve => setTimeout(resolve, 500)); // Simulate network delay
                
                const responses = {
                    'run_inference': {
                        success: true,
                        generated_text: `Generated response to: "${params.prompt}". This demonstrates the MCP tool integration working with the unified dashboard.`,
                        model: 'gpt2',
                        processing_time: 1.2
                    },
                    'search_models': {
                        success: true,
                        models: [
                            { id: 'gpt2', name: 'GPT-2', type: 'text-generation', downloads: 45000000 },
                            { id: 'bert-base-uncased', name: 'BERT Base', type: 'fill-mask', downloads: 15000000 }
                        ],
                        total: 2
                    },
                    'run_model_test': {
                        success: true,
                        test_type: params.test_type,
                        test_name: params.test_name,
                        results: {
                            accuracy: 0.92,
                            latency: 1.4,
                            throughput: 45,
                            success_rate: 0.98
                        }
                    },
                    'get_queue_status': {
                        success: true,
                        summary: { total_endpoints: 4, active_endpoints: 3, total_queue_size: 8, processing_tasks: 3 },
                        endpoints: [
                            { id: 'local_gpu_1', status: 'active', queue_size: 3, model_type: 'text-generation' },
                            { id: 'cloud_endpoint_1', status: 'active', queue_size: 5, model_type: 'embedding' }
                        ]
                    }
                };
                
                return responses[toolName] || { error: 'Tool not found' };
            }
        }
        
        const mcpClient = new MCPClient();
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => tab.style.display = 'none');
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).style.display = 'block';
            event.target.classList.add('active');
        }
        
        async function generateText() {
            const prompt = document.getElementById('promptInput').value;
            const resultDiv = document.getElementById('textResult');
            
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }
            
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '⏳ Generating text via MCP tool...';
            
            try {
                const result = await mcpClient.callTool('run_inference', { prompt });
                resultDiv.innerHTML = `
                    <strong>Generated Text:</strong><br>
                    <em>${result.generated_text}</em><br><br>
                    <small>Model: ${result.model} | Processing Time: ${result.processing_time}s</small>
                `;
            } catch (error) {
                resultDiv.innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        async function searchModels() {
            const query = document.getElementById('searchInput').value;
            const resultsDiv = document.getElementById('modelResults');
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            resultsDiv.innerHTML = '⏳ Searching models via MCP tool...';
            
            try {
                const result = await mcpClient.callTool('search_models', { query });
                resultsDiv.innerHTML = result.models.map(model => `
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; margin: 10px 0; border-radius: 5px;">
                        <strong>${model.name}</strong> (${model.id})<br>
                        Type: ${model.type} | Downloads: ${model.downloads.toLocaleString()}
                    </div>
                `).join('');
            } catch (error) {
                resultsDiv.innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        async function runTest(testType, testName) {
            const resultsDiv = document.getElementById('testResults');
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '⏳ Running test via MCP tool...';
            
            try {
                const result = await mcpClient.callTool('run_model_test', { test_type: testType, test_name: testName });
                resultsDiv.innerHTML = `
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px;">
                        <strong>${testType} - ${testName}</strong><br>
                        Accuracy: ${(result.results.accuracy * 100).toFixed(1)}% | 
                        Latency: ${result.results.latency.toFixed(2)}s | 
                        Throughput: ${result.results.throughput} tokens/sec | 
                        Success Rate: ${(result.results.success_rate * 100).toFixed(1)}%
                    </div>
                `;
            } catch (error) {
                resultsDiv.innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        async function getQueueStatus() {
            const resultsDiv = document.getElementById('queueResults');
            resultsDiv.innerHTML = '⏳ Getting queue status via MCP tool...';
            
            try {
                const result = await mcpClient.callTool('get_queue_status');
                resultsDiv.innerHTML = `
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px;">
                        <strong>Queue Summary</strong><br>
                        Total Endpoints: ${result.summary.total_endpoints} | 
                        Active: ${result.summary.active_endpoints} | 
                        Queue Size: ${result.summary.total_queue_size} | 
                        Processing: ${result.summary.processing_tasks}<br><br>
                        ${result.endpoints.map(ep => `<div>📍 ${ep.id}: ${ep.status} (${ep.queue_size} queued, ${ep.model_type})</div>`).join('')}
                    </div>
                `;
            } catch (error) {
                resultsDiv.innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        async function getModelQueues() {
            const resultsDiv = document.getElementById('queueResults');
            resultsDiv.innerHTML = '⏳ Getting model queues via MCP tool...';
            
            try {
                const result = await mcpClient.callTool('get_model_queues');
                resultsDiv.innerHTML = `
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px;">
                        <strong>Model Queue Information</strong><br>
                        Queue data filtered by model types would appear here.
                    </div>
                `;
            } catch (error) {
                resultsDiv.innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        // Update uptime counter
        let uptime = 0;
        setInterval(() => {
            uptime++;
            document.getElementById('uptime').textContent = `${uptime}s`;
        }, 1000);
        
        console.log('IPFS Accelerate MCP Dashboard Test loaded successfully');
        console.log('This demonstrates the unified dashboard with MCP tool integration');
    </script>
</body>
</html>
"""
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

def main():
    port = 8001
    server = HTTPServer(('localhost', port), TestDashboardHandler)
    print(f"🚀 Test Dashboard running at http://localhost:{port}")
    print("This demonstrates the unified MCP dashboard with testing functionality")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped")
        server.shutdown()

if __name__ == '__main__':
    main()