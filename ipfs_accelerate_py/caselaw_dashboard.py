"""
Caselaw Dashboard

Flask-based web dashboard for the Caselaw Access Project GraphRAG system.
Provides search interface, case analysis, and temporal logic visualization.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Try to import storage wrapper
try:
    from .common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ipfs_accelerate_py.caselaw_dataset_loader import CaselawDatasetLoader
    from ipfs_accelerate_py.caselaw_graphrag_processor import CaselawGraphRAGProcessor
    from ipfs_accelerate_py.temporal_deontic_logic import TemporalDeonticLogicProcessor
except ImportError as e:
    logging.warning(f"Import error: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaselawDashboard:
    """Web dashboard for caselaw GraphRAG system."""
    
    def __init__(self, port: int = 5000):
        """Initialize the dashboard.
        
        Args:
            port: Port to run the Flask server on
        """
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize storage wrapper for distributed storage
        self._storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None
        
        # Initialize components
        self.dataset_loader = None
        self.graphrag_processor = None
        self.temporal_processor = None
        self.dataset_info = None
        
        try:
            self.dataset_loader = CaselawDatasetLoader()
            self.graphrag_processor = CaselawGraphRAGProcessor()
            self.temporal_processor = TemporalDeonticLogicProcessor()
            
            # Load initial dataset
            cache_dir = os.getenv('CASELAW_CACHE_DIR')
            if cache_dir:
                logger.info(f"Using cache directory: {cache_dir}")
                self.dataset_info = self.dataset_loader.load_external_dataset(cache_dir)
            else:
                self.dataset_info = self.dataset_loader.load_sample_dataset(max_samples=50)
                
            # Process with GraphRAG
            if self.dataset_info:
                self.graph_stats = self.graphrag_processor.process_cases(self.dataset_info)
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self._initialize_mock_data()
        
        self._setup_routes()
        logger.info(f"Caselaw dashboard initialized on port {port}")
    
    def _initialize_mock_data(self):
        """Initialize with mock data when components fail to load."""
        logger.info("Initializing with mock data")
        
        self.dataset_info = {
            'total_cases': 25,
            'courts': ['Supreme Court', 'Court of Appeals'],
            'topics': ['Civil Rights', 'Constitutional Law'],
            'all_cases': [
                {
                    'id': 'mock-1',
                    'title': 'Brown v. Board of Education',
                    'citation': '347 U.S. 483',
                    'court': 'Supreme Court',
                    'year': 1954,
                    'topic': 'Civil Rights',
                    'summary': 'Landmark case declaring segregation unconstitutional.',
                    'relevance': 0.95
                }
            ]
        }
        
        self.graph_stats = {
            'total_nodes': 125,
            'total_edges': 280,
            'entity_types': ['Case', 'Court', 'Judge'],
            'relationship_types': ['CITES', 'OVERRULES']
        }
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return self._render_dashboard_template()
        
        @self.app.route('/api/search')
        def search():
            """Search cases API endpoint."""
            query = request.args.get('q', '')
            limit = int(request.args.get('limit', 10))
            
            try:
                if self.dataset_loader:
                    results = self.dataset_loader.search_cases(query, limit)
                else:
                    results = self._mock_search(query, limit)
                
                # Flatten results for UI compatibility
                flattened_results = []
                for result in results:
                    flattened_results.append({
                        'id': result.get('id', ''),
                        'title': result.get('title', ''),
                        'citation': result.get('citation', ''),
                        'court': result.get('court', ''),
                        'year': result.get('year', ''),
                        'topic': result.get('topic', ''),
                        'summary': result.get('summary', ''),
                        'relevance': result.get('relevance', 0.5)
                    })
                
                return jsonify({
                    'results': flattened_results,
                    'total': len(flattened_results),
                    'query': query
                })
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                return jsonify({'error': str(e), 'results': []}), 500
        
        @self.app.route('/api/legal-doctrines')
        @self.app.route('/api/doctrines')  # Alias for UI compatibility
        def legal_doctrines():
            """Get legal doctrines API endpoint."""
            try:
                if self.dataset_loader:
                    doctrines = self.dataset_loader.get_legal_doctrines()
                else:
                    doctrines = self._mock_doctrines()
                
                return jsonify({
                    'doctrines': doctrines,
                    'total': len(doctrines)
                })
                
            except Exception as e:
                logger.error(f"Doctrines error: {e}")  
                return jsonify({'error': str(e), 'doctrines': []}), 500
        
        @self.app.route('/api/temporal-analysis')
        def temporal_analysis():
            """Get temporal analysis API endpoint."""
            try:
                if self.temporal_processor:
                    # Create sample lineage for demonstration
                    sample_lineage = [
                        {
                            'case_id': 'qualified-immunity-1967',
                            'doctrine': 'qualified immunity',
                            'year': 1967,
                            'holding': 'Police officers have qualified immunity'
                        },
                        {
                            'case_id': 'qualified-immunity-1982',
                            'doctrine': 'qualified immunity', 
                            'year': 1982,
                            'holding': 'Qualified immunity requires clearly established law'
                        }
                    ]
                    
                    analysis = self.temporal_processor.analyze_lineage(sample_lineage)
                    # Add 'analysis' field for frontend compatibility
                    analysis['analysis'] = analysis.get('evolution_steps', [])
                else:
                    analysis = self._mock_temporal_analysis()
                
                return jsonify(analysis)
                
            except Exception as e:
                logger.error(f"Temporal analysis error: {e}")
                return jsonify({'error': str(e), 'analysis': []}), 500
        
        @self.app.route('/api/stats')
        def stats():
            """Get system statistics."""
            try:
                stats = {
                    'total_cases': self.dataset_info.get('total_cases', 0) if self.dataset_info else 0,
                    'total_courts': len(self.dataset_info.get('courts', [])) if self.dataset_info else 0,
                    'total_topics': len(self.dataset_info.get('topics', [])) if self.dataset_info else 0,
                    'graph_nodes': self.graph_stats.get('total_nodes', 0) if hasattr(self, 'graph_stats') else 0,
                    'graph_edges': self.graph_stats.get('total_edges', 0) if hasattr(self, 'graph_stats') else 0
                }
                
                return jsonify(stats)
                
            except Exception as e:
                logger.error(f"Stats error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors."""
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            return jsonify({'error': 'Internal server error'}), 500
    
    def _render_dashboard_template(self) -> str:
        """Render the dashboard HTML template."""
        # Generate HTML template directly since template files may not exist
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Caselaw GraphRAG Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #1e40af;
            --secondary-color: #64748b;
            --success-color: #059669;
            --warning-color: #d97706;
            --danger-color: #dc2626;
            --dark-color: #1f2937;
            --light-color: #f8fafc;
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
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
        }
        
        .search-container {
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
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1);
        }
        
        .btn-primary {
            background: var(--primary-color);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 500;
        }
        
        .result-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .result-card:hover {
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .loading {
            text-align: center;
            padding: 3rem;
            color: var(--secondary-color);
        }
        
        .status-card {
            background: linear-gradient(45deg, #059669, #10b981);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg">
        <div class="container">
            <a class="navbar-brand fw-bold" href="#">
                <i class="fas fa-gavel me-2"></i>
                Caselaw GraphRAG
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="#search">Search</a>
                <a class="nav-link" href="#doctrines">Doctrines</a>
                <a class="nav-link" href="#analysis">Analysis</a>
            </div>
        </div>
    </nav>

    <div class="main-container">
        <div class="row">
            <div class="col-md-8">
                <h1 class="mb-4">
                    <i class="fas fa-balance-scale me-3"></i>
                    Legal Case Search & Analysis
                </h1>
                
                <div class="search-container" id="search">
                    <h3><i class="fas fa-search me-2"></i>Search Cases</h3>
                    <div class="row">
                        <div class="col-md-8">
                            <input type="text" class="form-control search-input" id="searchInput" 
                                   placeholder="Enter keywords (e.g., 'civil rights', 'constitutional law')">
                        </div>
                        <div class="col-md-4">
                            <button class="btn btn-primary w-100" onclick="searchCases()">
                                <i class="fas fa-search me-2"></i>Search
                            </button>
                        </div>
                    </div>
                </div>
                
                <div id="searchResults"></div>
            </div>
            
            <div class="col-md-4">
                <div class="status-card mb-4">
                    <h5><i class="fas fa-server me-2"></i>System Status</h5>
                    <p class="mb-0">GraphRAG Server Online</p>
                </div>
                
                <div class="feature-card">
                    <h5><i class="fas fa-book-open me-2"></i>Legal Doctrines</h5>
                    <p>Explore legal doctrine evolution</p>
                    <button class="btn btn-light btn-sm" onclick="loadDoctrines()">
                        View Doctrines
                    </button>
                </div>
                
                <div class="feature-card">
                    <h5><i class="fas fa-clock me-2"></i>Temporal Analysis</h5>
                    <p>Track doctrine changes over time</p>
                    <button class="btn btn-light btn-sm" onclick="loadTemporal()">
                        View Analysis
                    </button>
                </div>
            </div>
        </div>
        
        <div id="doctrines" class="mt-4" style="display: none;">
            <h3><i class="fas fa-book-open me-2"></i>Legal Doctrines</h3>
            <div id="doctrinesList"></div>
        </div>
        
        <div id="analysis" class="mt-4" style="display: none;">
            <h3><i class="fas fa-clock me-2"></i>Temporal Analysis</h3>
            <div id="analysisResults"></div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Search functionality
        function searchCases() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            const resultsDiv = document.getElementById('searchResults');
            resultsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin fa-2x"></i><p>Searching...</p></div>';
            
            fetch(`/api/search?q=${encodeURIComponent(query)}&limit=10`)
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
                resultsDiv.innerHTML = `<div class="alert alert-info">No results found for "${query}"</div>`;
                return;
            }
            
            let html = `<h4>Search Results for "${query}" (${results.length} found)</h4>`;
            
            results.forEach(result => {
                html += `
                    <div class="result-card">
                        <h5>${result.title || 'Untitled Case'}</h5>
                        <p class="text-muted mb-2">
                            <i class="fas fa-gavel me-2"></i>${result.citation || 'No citation'} | 
                            <i class="fas fa-building me-2"></i>${result.court || 'Unknown court'} | 
                            <i class="fas fa-calendar me-2"></i>${result.year || 'Unknown year'}
                        </p>
                        <p class="mb-2">${result.summary || 'No summary available'}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-primary">${result.topic || 'General'}</span>
                            <small class="text-success">Relevance: ${((result.relevance || 0.5) * 100).toFixed(0)}%</small>
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
        
        function loadDoctrines() {
            const doctrinesDiv = document.getElementById('doctrines');
            const listDiv = document.getElementById('doctrinesList');
            
            doctrinesDiv.style.display = 'block';
            listDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading doctrines...</div>';
            
            fetch('/api/legal-doctrines')
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    if (data.doctrines && data.doctrines.length > 0) {
                        data.doctrines.forEach(doctrine => {
                            html += `
                                <div class="result-card">
                                    <h5>${doctrine.name}</h5>
                                    <p>${doctrine.description}</p>
                                    <small class="text-muted">
                                        Key cases: ${doctrine.key_cases ? doctrine.key_cases.join(', ') : 'None listed'}
                                    </small>
                                </div>
                            `;
                        });
                    } else {
                        html = '<div class="alert alert-info">No doctrines available</div>';
                    }
                    listDiv.innerHTML = html;
                })
                .catch(error => {
                    console.error('Doctrines error:', error);
                    listDiv.innerHTML = '<div class="alert alert-danger">Failed to load doctrines</div>';
                });
        }
        
        function loadTemporal() {
            const analysisDiv = document.getElementById('analysis');
            const resultsDiv = document.getElementById('analysisResults');
            
            analysisDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading analysis...</div>';
            
            fetch('/api/temporal-analysis')
                .then(response => response.json())
                .then(data => {
                    let html = `
                        <div class="result-card">
                            <h5>Temporal Analysis Results</h5>
                            <p>Analysis ID: ${data.lineage_id || 'N/A'}</p>
                            <p>Evolution Steps: ${data.evolution_steps ? data.evolution_steps.length : 0}</p>
                            <p>Theorems Generated: ${data.theorems ? data.theorems.length : 0}</p>
                            <p>Consistency Score: ${data.consistency_score ? (data.consistency_score * 100).toFixed(1) + '%' : 'N/A'}</p>
                        </div>
                    `;
                    resultsDiv.innerHTML = html;
                })
                .catch(error => {
                    console.error('Analysis error:', error);
                    resultsDiv.innerHTML = '<div class="alert alert-danger">Failed to load analysis</div>';
                });
        }
        
        // Allow Enter key to trigger search
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchCases();
            }
        });
        
        // Load initial stats
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                console.log('System stats:', data);
            })
            .catch(error => console.error('Stats error:', error));
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _mock_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Mock search for when components are not available."""
        mock_results = [
            {
                'id': 'mock-search-1',
                'title': f'Mock Result for "{query}"',
                'citation': '123 U.S. 456',
                'court': 'Supreme Court',
                'year': 2020,
                'topic': 'Mock Topic',
                'summary': f'This is a mock search result for the query: {query}',
                'relevance': 0.8
            }
        ]
        return mock_results[:limit]
    
    def _mock_doctrines(self) -> List[Dict[str, Any]]:
        """Mock doctrines for when components are not available."""
        return [
            {
                'name': 'Mock Doctrine',
                'description': 'This is a mock legal doctrine for demonstration.',
                'key_cases': ['Mock Case 1', 'Mock Case 2']
            }
        ]
    
    def _mock_temporal_analysis(self) -> Dict[str, Any]:
        """Mock temporal analysis for when components are not available."""
        return {
            'lineage_id': 'mock_analysis',
            'evolution_steps': [],
            'theorems': [],
            'consistency_score': 0.85,
            'analysis': []  # For frontend compatibility
        }
    
    def run(self, debug: bool = False, host: str = '127.0.0.1') -> None:
        """Run the Flask application.
        
        Args:
            debug: Enable debug mode
            host: Host to bind to
        """
        logger.info(f"Starting Caselaw Dashboard on http://{host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=debug)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Caselaw Dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    
    args = parser.parse_args()
    
    dashboard = CaselawDashboard(port=args.port)
    dashboard.run(debug=args.debug, host=args.host)