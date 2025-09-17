"""
MCP Dashboard

Flask-based MCP (Model Control Plane) dashboard that provides access to various
AI and GraphRAG services, including the Caselaw GraphRAG system.
"""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify
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
                    'Investigation': 'enabled'
                },
                'caselaw': {
                    'available': True,
                    'url': 'http://127.0.0.1:5000'
                }
            })
    
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