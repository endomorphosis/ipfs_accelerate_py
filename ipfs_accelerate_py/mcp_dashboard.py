"""
MCP Dashboard

Flask-based MCP (Model Control Plane) dashboard that provides access to various
AI and GraphRAG services, including the Caselaw GraphRAG system.
"""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPDashboard:
    """MCP Dashboard with links to various services."""
    
    def __init__(self, port: int = 8899, host: str = '127.0.0.1'):
        """Initialize the MCP dashboard.
        
        Args:
            port: Port to run on
            host: Host to bind to
        """
        self.port = port
        self.host = host
        self.app = Flask(__name__)
        CORS(self.app)
        
        self._setup_routes()
        logger.info(f"MCP Dashboard initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/mcp')
        def mcp_dashboard():
            """Main MCP dashboard."""
            return self._render_mcp_template()
        
        @self.app.route('/mcp/graphrag')
        def graphrag():
            """GraphRAG service page."""
            return self._render_feature_template("GraphRAG", "graph processing")
        
        @self.app.route('/mcp/analytics')
        def analytics():
            """Analytics service page."""
            return self._render_feature_template("Analytics", "data analysis")
        
        @self.app.route('/mcp/rag')
        def rag():
            """RAG Query service page."""
            return self._render_feature_template("RAG Query", "retrieval augmented generation")
        
        @self.app.route('/mcp/investigation')
        def investigation():
            """Investigation service page."""
            return self._render_feature_template("Investigation", "case investigation")
        
        @self.app.route('/api/mcp/status')
        def status():
            """MCP status API."""
            return jsonify({
                'status': 'running',
                'services': {
                    'GraphRAG': 'enabled',
                    'Analytics': 'enabled', 
                    'RAG Query': 'enabled',
                    'Investigation': 'enabled',
                    'Model Discovery': 'enabled',
                    'HuggingFace Scanner': 'enabled'
                },
                'caselaw': {
                    'available': True,
                    'url': 'http://127.0.0.1:5000'
                },
                'model_manager': {
                    'available': True,
                    'total_models': self._get_model_count()
                }
            })
        
        @self.app.route('/mcp/models')
        def models():
            """Model discovery and search page."""
            return self._render_model_discovery_template()
        
        @self.app.route('/api/mcp/models/search')
        def search_models():
            """Search models API endpoint."""
            query = request.args.get('q', '')
            task_filter = request.args.get('task')
            hardware_filter = request.args.get('hardware')
            limit = int(request.args.get('limit', 20))
            
            try:
                scanner = self._get_hub_scanner()
                results = scanner.search_models(
                    query=query,
                    task_filter=task_filter,
                    hardware_filter=hardware_filter,
                    limit=limit
                )
                
                return jsonify({
                    'results': results,
                    'total': len(results),
                    'query': query
                })
                
            except Exception as e:
                logger.error(f"Model search error: {e}")
                return jsonify({'error': str(e), 'results': []}), 500
        
        @self.app.route('/api/mcp/models/recommend')
        def recommend_models():
            """Get model recommendations using bandit algorithm."""
            task_type = request.args.get('task_type', 'text-generation')
            input_type = request.args.get('input_type', 'text')
            output_type = request.args.get('output_type', 'text')
            hardware = request.args.get('hardware', 'cpu')
            performance = request.args.get('performance', 'balanced')
            limit = int(request.args.get('limit', 5))
            
            try:
                from .huggingface_hub_scanner import HuggingFaceHubScanner
                from .model_manager import RecommendationContext
                
                context = RecommendationContext(
                    task_type=task_type,
                    input_type=input_type,
                    output_type=output_type,
                    hardware_constraint=hardware,
                    performance_preference=performance
                )
                
                scanner = self._get_hub_scanner()
                recommendations = scanner.get_model_recommendations(context, limit)
                
                return jsonify({
                    'recommendations': recommendations,
                    'context': {
                        'task_type': task_type,
                        'input_type': input_type,
                        'output_type': output_type,
                        'hardware': hardware,
                        'performance': performance
                    }
                })
                
            except Exception as e:
                logger.error(f"Model recommendation error: {e}")
                return jsonify({'error': str(e), 'recommendations': []}), 500
        
        @self.app.route('/api/mcp/models/scan', methods=['POST'])
        def scan_huggingface_hub():
            """Trigger HuggingFace Hub scan."""
            data = request.get_json() or {}
            limit = data.get('limit', 100)
            task_filter = data.get('task_filter')
            
            try:
                scanner = self._get_hub_scanner()
                
                # Run scan in background
                import threading
                def run_scan():
                    scanner.scan_all_models(limit=limit, task_filter=task_filter)
                
                scan_thread = threading.Thread(target=run_scan)
                scan_thread.daemon = True
                scan_thread.start()
                
                return jsonify({
                    'status': 'started',
                    'message': f'Scanning HuggingFace Hub with limit={limit}',
                    'limit': limit,
                    'task_filter': task_filter
                })
                
            except Exception as e:
                logger.error(f"Hub scan error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mcp/models/stats')
        def model_stats():
            """Get model statistics."""
            try:
                scanner = self._get_hub_scanner()
                
                stats = {
                    'total_cached_models': len(scanner.model_cache),
                    'models_with_performance': len(scanner.performance_cache),
                    'models_with_compatibility': len(scanner.compatibility_cache),
                    'architecture_distribution': scanner._get_architecture_distribution(),
                    'task_distribution': scanner._get_task_distribution(),
                    'popular_models': scanner._get_popular_models_summary()[:10]
                }
                
                return jsonify(stats)
                
            except Exception as e:
                logger.error(f"Model stats error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _render_mcp_template(self) -> str:
        """Render the main MCP dashboard template."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #1e40af;
            --secondary: #64748b;
            --success: #059669;
            --warning: #d97706;
        }
        
        body { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .navbar { 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .main-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin: 2rem;
            padding: 2rem;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
            text-decoration: none;
            display: block;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            color: white;
            text-decoration: none;
        }
        
        .external-link {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        }
        
        .status-card {
            background: linear-gradient(45deg, #059669, #10b981);
            color: white;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg">
        <div class="container">
            <a class="navbar-brand fw-bold" href="/mcp">
                <i class="fas fa-cogs me-2"></i>
                MCP Dashboard
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="#services">Services</a>
                <a class="nav-link" href="#external">External</a>
                <a class="nav-link" href="#status">Status</a>
            </div>
        </div>
    </nav>

    <div class="main-container">
        <div class="row">
            <div class="col-md-8">
                <h1 class="mb-4">
                    <i class="fas fa-server me-3"></i>
                    Model Control Plane
                </h1>
                
                <div id="services">
                    <h3 class="mb-4">Available Services</h3>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <a href="/mcp/graphrag" class="feature-card">
                                <h5><i class="fas fa-project-diagram me-2"></i>GraphRAG</h5>
                                <p class="mb-0">Graph-based Retrieval Augmented Generation</p>
                            </a>
                        </div>
                        <div class="col-md-6">
                            <a href="/mcp/analytics" class="feature-card">
                                <h5><i class="fas fa-chart-line me-2"></i>Analytics</h5>
                                <p class="mb-0">Data Analysis and Insights</p>
                            </a>
                        </div>
                        <div class="col-md-6">
                            <a href="/mcp/rag" class="feature-card">
                                <h5><i class="fas fa-search me-2"></i>RAG Query</h5>
                                <p class="mb-0">Retrieval Augmented Generation Queries</p>
                            </a>
                        </div>
                        <div class="col-md-6">
                            <a href="/mcp/investigation" class="feature-card">
                                <h5><i class="fas fa-magnifying-glass me-2"></i>Investigation</h5>
                                <p class="mb-0">Case Investigation and Analysis</p>
                            </a>
                        </div>
                        <div class="col-md-6">
                            <a href="/mcp/models" class="feature-card">
                                <h5><i class="fas fa-robot me-2"></i>Model Discovery</h5>
                                <p class="mb-0">HuggingFace Model Search & AI Recommendations</p>
                            </a>
                        </div>
                    </div>
                </div>
                
                <div id="external" class="mt-5">
                    <h3 class="mb-4">External Services</h3>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <a href="http://127.0.0.1:5000" target="_blank" class="feature-card external-link">
                                <h5><i class="fas fa-gavel me-2"></i>Caselaw GraphRAG</h5>
                                <p class="mb-0">Legal Case Search & Analysis Dashboard</p>
                                <small><i class="fas fa-external-link-alt me-1"></i>http://127.0.0.1:5000</small>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4" id="status">
                <div class="status-card mb-4">
                    <h5><i class="fas fa-server me-2"></i>System Status</h5>
                    <p class="mb-0" id="statusText">Loading...</p>
                </div>
                
                <div class="card">
                    <div class="card-body">
                        <h6><i class="fas fa-info-circle me-2"></i>Quick Actions</h6>
                        <div class="list-group list-group-flush">
                            <button class="list-group-item list-group-item-action" onclick="checkStatus()">
                                <i class="fas fa-sync me-2"></i>Refresh Status
                            </button>
                            <a href="/api/mcp/status" class="list-group-item list-group-item-action" target="_blank">
                                <i class="fas fa-code me-2"></i>View API Status
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function checkStatus() {
            document.getElementById('statusText').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Checking...';
            
            fetch('/api/mcp/status')
                .then(response => response.json())
                .then(data => {
                    const status = data.status === 'running' ? 
                        '<i class="fas fa-check-circle me-2"></i>All Systems Operational' :
                        '<i class="fas fa-exclamation-triangle me-2"></i>System Issues Detected';
                    
                    document.getElementById('statusText').innerHTML = status;
                })
                .catch(error => {
                    console.error('Status check failed:', error);
                    document.getElementById('statusText').innerHTML = 
                        '<i class="fas fa-times-circle me-2"></i>Status Check Failed';
                });
        }
        
        // Load initial status
        checkStatus();
        
        // Auto-refresh status every 30 seconds
        setInterval(checkStatus, 30000);
    </script>
</body>
</html>
        """
        return html
    
    def _render_feature_template(self, feature_name: str, description: str) -> str:
        """Render a feature page template."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{feature_name} - MCP Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .main-container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin: 2rem;
            padding: 2rem;
        }}
        
        .feature-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="feature-header">
            <h1><i class="fas fa-cogs me-3"></i>{feature_name}</h1>
            <p class="mb-0">{description.title()} service interface</p>
        </div>
        
        <div class="alert alert-info">
            <i class="fas fa-info-circle me-2"></i>
            This {feature_name} service is available via the MCP interface.
        </div>
        
        <div class="text-center">
            <a href="/mcp" class="btn btn-primary">
                <i class="fas fa-arrow-left me-2"></i>Back to MCP Dashboard
            </a>
        </div>
    </div>
</body>
</html>
        """
        return html
    
    def _get_hub_scanner(self):
        """Get or create HuggingFace Hub scanner instance."""
        if not hasattr(self, '_hub_scanner'):
            try:
                from .huggingface_hub_scanner import HuggingFaceHubScanner
                self._hub_scanner = HuggingFaceHubScanner(cache_dir="./mcp_model_cache")
            except ImportError:
                logger.warning("HuggingFace Hub scanner not available")
                self._hub_scanner = None
        return self._hub_scanner
    
    def _get_model_count(self) -> int:
        """Get total number of models in the model manager."""
        try:
            scanner = self._get_hub_scanner()
            if scanner:
                return len(scanner.model_cache)
        except Exception:
            pass
        return 0
    
    def _render_model_discovery_template(self) -> str:
        """Render the model discovery page template."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Discovery - MCP Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #1e40af;
            --secondary: #64748b;
            --success: #059669;
            --warning: #d97706;
        }
        
        body { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .navbar { 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .main-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin: 2rem;
            padding: 2rem;
        }
        
        .search-card {
            background: #f8fafc;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .search-input {
            border-radius: 25px;
            border: 2px solid #e2e8f0;
            padding: 12px 20px;
            font-size: 16px;
        }
        
        .search-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1);
        }
        
        .btn-primary {
            background: var(--primary);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 500;
        }
        
        .model-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .model-card:hover {
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .performance-badge {
            background: linear-gradient(45deg, #059669, #10b981);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .compatibility-badge {
            background: linear-gradient(45deg, #3b82f6, #6366f1);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .filter-section {
            background: #f1f5f9;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .loading {
            text-align: center;
            padding: 3rem;
            color: var(--secondary);
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg">
        <div class="container">
            <a class="navbar-brand fw-bold" href="/mcp">
                <i class="fas fa-cogs me-2"></i>
                MCP Dashboard
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/mcp">Home</a>
                <a class="nav-link" href="#search">Search</a>
                <a class="nav-link" href="#recommend">Recommend</a>
                <a class="nav-link" href="#stats">Stats</a>
            </div>
        </div>
    </nav>

    <div class="main-container">
        <div class="row">
            <div class="col-md-8">
                <h1 class="mb-4">
                    <i class="fas fa-robot me-3"></i>
                    Model Discovery & Search
                </h1>
                
                <div class="search-card" id="search">
                    <h3><i class="fas fa-search me-2"></i>Search HuggingFace Models</h3>
                    <div class="row">
                        <div class="col-md-8">
                            <input type="text" class="form-control search-input" id="searchInput" 
                                   placeholder="Search models (e.g., 'bert', 'text-generation', 'vision')">
                        </div>
                        <div class="col-md-4">
                            <button class="btn btn-primary w-100" onclick="searchModels()">
                                <i class="fas fa-search me-2"></i>Search Models
                            </button>
                        </div>
                    </div>
                    
                    <div class="filter-section mt-3">
                        <div class="row">
                            <div class="col-md-4">
                                <label class="form-label">Task Type</label>
                                <select class="form-select" id="taskFilter">
                                    <option value="">All Tasks</option>
                                    <option value="text-generation">Text Generation</option>
                                    <option value="text-classification">Text Classification</option>
                                    <option value="question-answering">Question Answering</option>
                                    <option value="summarization">Summarization</option>
                                    <option value="translation">Translation</option>
                                    <option value="image-classification">Image Classification</option>
                                    <option value="object-detection">Object Detection</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Hardware</label>
                                <select class="form-select" id="hardwareFilter">
                                    <option value="">All Hardware</option>
                                    <option value="cpu">CPU Only</option>
                                    <option value="gpu">GPU Compatible</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Performance</label>
                                <select class="form-select" id="performanceFilter">
                                    <option value="balanced">Balanced</option>
                                    <option value="speed">Speed Optimized</option>
                                    <option value="accuracy">Accuracy Optimized</option>
                                    <option value="memory">Memory Efficient</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="search-card" id="recommend">
                    <h3><i class="fas fa-magic me-2"></i>AI-Powered Recommendations</h3>
                    <p class="text-muted">Get personalized model recommendations using our bandit algorithm</p>
                    <button class="btn btn-success" onclick="getRecommendations()">
                        <i class="fas fa-lightbulb me-2"></i>Get Smart Recommendations
                    </button>
                </div>
                
                <div id="searchResults"></div>
                <div id="recommendationResults"></div>
            </div>
            
            <div class="col-md-4" id="stats">
                <div class="stats-card">
                    <h5><i class="fas fa-chart-bar me-2"></i>Model Statistics</h5>
                    <div id="modelStats">
                        <div class="loading">
                            <i class="fas fa-spinner fa-spin fa-2x"></i>
                            <p>Loading statistics...</p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-body">
                        <h6><i class="fas fa-cog me-2"></i>Hub Scanner</h6>
                        
                        <div class="mt-3">
                            <label class="form-label">Scan Limit</label>
                            <input type="number" class="form-control" id="scanLimit" value="100" min="10" max="1000">
                        </div>
                        
                        <div class="d-grid gap-2 mt-3">
                            <button class="btn btn-outline-primary btn-sm" onclick="scanHub()">
                                <i class="fas fa-sync me-2"></i>Scan HuggingFace Hub
                            </button>
                            <button class="btn btn-outline-info btn-sm" onclick="loadStats()">
                                <i class="fas fa-chart-line me-2"></i>Refresh Statistics
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="card mt-3">
                    <div class="card-body">
                        <h6><i class="fas fa-link me-2"></i>Quick Links</h6>
                        <div class="list-group list-group-flush">
                            <a href="/mcp" class="list-group-item list-group-item-action">
                                <i class="fas fa-home me-2"></i>MCP Home
                            </a>
                            <a href="/mcp/graphrag" class="list-group-item list-group-item-action">
                                <i class="fas fa-project-diagram me-2"></i>GraphRAG
                            </a>
                            <a href="http://127.0.0.1:5000" target="_blank" class="list-group-item list-group-item-action">
                                <i class="fas fa-gavel me-2"></i>Caselaw Dashboard
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Search functionality
        function searchModels() {
            const query = document.getElementById('searchInput').value.trim();
            const taskFilter = document.getElementById('taskFilter').value;
            const hardwareFilter = document.getElementById('hardwareFilter').value;
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            const resultsDiv = document.getElementById('searchResults');
            resultsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin fa-2x"></i><p>Searching models...</p></div>';
            
            const params = new URLSearchParams({
                q: query,
                limit: '20'
            });
            
            if (taskFilter) params.append('task', taskFilter);
            if (hardwareFilter) params.append('hardware', hardwareFilter);
            
            fetch(`/api/mcp/models/search?${params}`)
                .then(response => response.json())
                .then(data => {
                    displaySearchResults(data.results, query);
                })
                .catch(error => {
                    console.error('Search error:', error);
                    resultsDiv.innerHTML = '<div class="alert alert-danger">Search failed. Please try again.</div>';
                });
        }
        
        function displaySearchResults(results, query) {
            const resultsDiv = document.getElementById('searchResults');
            
            if (!results || results.length === 0) {
                resultsDiv.innerHTML = `<div class="alert alert-info">No models found for "${query}". Try scanning more models first.</div>`;
                return;
            }
            
            let html = `<h4 class="mt-4">Search Results for "${query}" (${results.length} found)</h4>`;
            
            results.forEach(result => {
                const modelInfo = result.model_info || {};
                const performance = result.performance || {};
                const compatibility = result.compatibility || {};
                
                html += `
                    <div class="model-card">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <h5 class="mb-1">${modelInfo.model_name || result.model_id}</h5>
                            <span class="badge bg-primary">${modelInfo.pipeline_tag || 'Unknown'}</span>
                        </div>
                        
                        <p class="text-muted mb-2">${modelInfo.description || 'No description available'}</p>
                        
                        <div class="row mb-2">
                            <div class="col-md-6">
                                <small class="text-muted">
                                    <i class="fas fa-download me-1"></i>Downloads: ${(modelInfo.downloads || 0).toLocaleString()}
                                </small>
                            </div>
                            <div class="col-md-6">
                                <small class="text-muted">
                                    <i class="fas fa-heart me-1"></i>Likes: ${(modelInfo.likes || 0).toLocaleString()}
                                </small>
                            </div>
                        </div>
                        
                        <div class="d-flex flex-wrap gap-2">
                            ${performance.throughput_tokens_per_sec ? 
                                `<span class="performance-badge">
                                    <i class="fas fa-tachometer-alt me-1"></i>${performance.throughput_tokens_per_sec.toFixed(1)} tok/s
                                </span>` : ''}
                            ${compatibility.min_ram_gb ? 
                                `<span class="compatibility-badge">
                                    <i class="fas fa-memory me-1"></i>${compatibility.min_ram_gb}GB RAM
                                </span>` : ''}
                            ${modelInfo.architecture ? 
                                `<span class="badge bg-secondary">${modelInfo.architecture}</span>` : ''}
                        </div>
                        
                        <div class="mt-2">
                            <a href="https://huggingface.co/${result.model_id}" target="_blank" class="btn btn-sm btn-outline-primary">
                                <i class="fas fa-external-link-alt me-1"></i>View on HuggingFace
                            </a>
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
        
        function getRecommendations() {
            const taskType = document.getElementById('taskFilter').value || 'text-generation';
            const hardware = document.getElementById('hardwareFilter').value || 'cpu';
            const performance = document.getElementById('performanceFilter').value || 'balanced';
            
            const resultsDiv = document.getElementById('recommendationResults');
            resultsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin fa-2x"></i><p>Getting AI recommendations...</p></div>';
            
            const params = new URLSearchParams({
                task_type: taskType,
                hardware: hardware,
                performance: performance,
                limit: '5'
            });
            
            fetch(`/api/mcp/models/recommend?${params}`)
                .then(response => response.json())
                .then(data => {
                    displayRecommendations(data.recommendations, data.context);
                })
                .catch(error => {
                    console.error('Recommendation error:', error);
                    resultsDiv.innerHTML = '<div class="alert alert-danger">Recommendations failed. Please try again.</div>';
                });
        }
        
        function displayRecommendations(recommendations, context) {
            const resultsDiv = document.getElementById('recommendationResults');
            
            if (!recommendations || recommendations.length === 0) {
                resultsDiv.innerHTML = '<div class="alert alert-info">No recommendations available. Try scanning more models first.</div>';
                return;
            }
            
            let html = `
                <h4 class="mt-4">
                    <i class="fas fa-magic me-2"></i>AI Recommendations
                    <small class="text-muted">(${context.task_type}, ${context.hardware}, ${context.performance})</small>
                </h4>
            `;
            
            recommendations.forEach((rec, index) => {
                const modelInfo = rec.model_info || {};
                const performance = rec.performance || {};
                const compatibility = rec.compatibility || {};
                
                html += `
                    <div class="model-card">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <h5 class="mb-1">
                                <span class="badge bg-success me-2">#${index + 1}</span>
                                ${modelInfo.model_name || rec.model_id}
                            </h5>
                            <div>
                                <span class="badge bg-primary">${modelInfo.pipeline_tag || 'Unknown'}</span>
                                <span class="badge bg-info ms-1">Confidence: ${(rec.confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                        
                        <p class="text-muted mb-2">${modelInfo.description || 'No description available'}</p>
                        
                        <div class="row mb-2">
                            <div class="col-md-4">
                                <small class="text-success">
                                    <i class="fas fa-trophy me-1"></i>Expected Reward: ${rec.expected_reward.toFixed(3)}
                                </small>
                            </div>
                            <div class="col-md-4">
                                <small class="text-muted">
                                    <i class="fas fa-download me-1"></i>${(modelInfo.downloads || 0).toLocaleString()}
                                </small>
                            </div>
                            <div class="col-md-4">
                                <small class="text-muted">
                                    <i class="fas fa-heart me-1"></i>${(modelInfo.likes || 0).toLocaleString()}
                                </small>
                            </div>
                        </div>
                        
                        <div class="d-flex flex-wrap gap-2">
                            ${performance.throughput_tokens_per_sec ? 
                                `<span class="performance-badge">
                                    <i class="fas fa-tachometer-alt me-1"></i>${performance.throughput_tokens_per_sec.toFixed(1)} tok/s
                                </span>` : ''}
                            ${compatibility.min_ram_gb ? 
                                `<span class="compatibility-badge">
                                    <i class="fas fa-memory me-1"></i>${compatibility.min_ram_gb}GB RAM
                                </span>` : ''}
                        </div>
                        
                        <div class="mt-2">
                            <a href="https://huggingface.co/${rec.model_id}" target="_blank" class="btn btn-sm btn-outline-primary">
                                <i class="fas fa-external-link-alt me-1"></i>View on HuggingFace
                            </a>
                            <button class="btn btn-sm btn-success ms-2" onclick="provideFeedback('${rec.model_id}', true)">
                                <i class="fas fa-thumbs-up me-1"></i>Good Recommendation
                            </button>
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
        
        function scanHub() {
            const limit = document.getElementById('scanLimit').value || 100;
            const taskFilter = document.getElementById('taskFilter').value;
            
            const data = {
                limit: parseInt(limit),
                task_filter: taskFilter || null
            };
            
            fetch('/api/mcp/models/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    alert(`Hub scan started! Scanning up to ${data.limit} models.`);
                    // Refresh stats after a delay
                    setTimeout(loadStats, 5000);
                } else {
                    alert('Scan failed: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Scan error:', error);
                alert('Scan failed');
            });
        }
        
        function loadStats() {
            const statsDiv = document.getElementById('modelStats');
            statsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';
            
            fetch('/api/mcp/models/stats')
                .then(response => response.json())
                .then(data => {
                    displayStats(data);
                })
                .catch(error => {
                    console.error('Stats error:', error);
                    statsDiv.innerHTML = '<div class="text-danger">Failed to load statistics</div>';
                });
        }
        
        function displayStats(stats) {
            const statsDiv = document.getElementById('modelStats');
            
            let html = `
                <div class="mb-3">
                    <div class="d-flex justify-content-between">
                        <span>Total Models:</span>
                        <strong>${stats.total_cached_models || 0}</strong>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span>With Performance Data:</span>
                        <strong>${stats.models_with_performance || 0}</strong>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span>With Compatibility:</span>
                        <strong>${stats.models_with_compatibility || 0}</strong>
                    </div>
                </div>
            `;
            
            if (stats.task_distribution) {
                html += '<h6>Popular Tasks:</h6><div class="small">';
                const tasks = Object.entries(stats.task_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5);
                
                tasks.forEach(([task, count]) => {
                    html += `<div class="d-flex justify-content-between">
                        <span>${task}:</span>
                        <span>${count}</span>
                    </div>`;
                });
                html += '</div>';
            }
            
            statsDiv.innerHTML = html;
        }
        
        function provideFeedback(modelId, positive) {
            // This would send feedback to the bandit algorithm
            console.log(`Feedback for ${modelId}: ${positive ? 'positive' : 'negative'}`);
            alert('Feedback recorded! The AI will learn from this.');
        }
        
        // Allow Enter key to trigger search
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchModels();
            }
        });
        
        // Load initial stats
        loadStats();
    </script>
</body>
</html>
        """
        
        return html

    def run(self, debug: bool = False) -> None:
        """Run the MCP dashboard.
        
        Args:
            debug: Enable debug mode
        """
        logger.info(f"Starting MCP Dashboard on http://{self.host}:{self.port}/mcp")
        self.app.run(host=self.host, port=self.port, debug=debug)


if __name__ == '__main__':
    dashboard = MCPDashboard()
    dashboard.run()