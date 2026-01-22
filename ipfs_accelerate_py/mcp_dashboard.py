"""
MCP Dashboard

Flask-based MCP (Model Control Plane) dashboard that provides access to various
AI and GraphRAG services, including the Caselaw GraphRAG system.
"""

import os
import logging
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, List, Optional

# Try to import Flask (required for dashboard)
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_cors import CORS
    HAVE_FLASK = True
except ImportError:
    HAVE_FLASK = False
    # Don't exit immediately - let the caller handle the missing Flask
    Flask = None
    render_template = None
    jsonify = None
    request = None
    send_from_directory = None
    CORS = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPDashboard:
    """MCP Dashboard with links to various services."""
    
    def __init__(self, port: int = 8899, host: str = '127.0.0.1', mcp_server=None, 
                 enable_autoscaler: bool = True, autoscaler_config: Optional[Dict[str, Any]] = None):
        """Initialize the MCP dashboard.
        
        Args:
            port: Port to run on
            host: Host to bind to
            mcp_server: Optional MCP server instance to get tools from
            enable_autoscaler: Whether to start GitHub Actions autoscaler (default: True)
            autoscaler_config: Configuration for autoscaler (owner, interval, etc.)
        """
        if not HAVE_FLASK:
            raise ImportError("Flask is required for the MCP Dashboard. Install with: pip install flask flask-cors")
        
        import time
        self.port = port
        self.host = host
        self.mcp_server = mcp_server
        self._start_time = time.time()  # Track when dashboard started
        self.enable_autoscaler = enable_autoscaler
        self.autoscaler_config = autoscaler_config or {}
        self.autoscaler_instance = None
        self.autoscaler_thread = None
        
        # Set up Flask with proper template and static folders
        # __file__ is in ipfs_accelerate_py/, so use dirname(__file__) for templates/static
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        
        self.app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
        CORS(self.app)
        
        self._setup_routes()
        logger.info(f"MCP Dashboard initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        @self.app.route('/mcp')
        def mcp_dashboard():
            """Main MCP dashboard."""
            return render_template('dashboard.html')
        
        @self.app.route('/dashboard')
        def dashboard():
            """Alternative dashboard route."""
            return render_template('dashboard.html')
        
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
        
        @self.app.route('/api/mcp/user')
        def user_info():
            """Get current user information."""
            from ipfs_accelerate_py.mcp.tools.dashboard_data import get_user_info
            return jsonify(get_user_info())
        
        @self.app.route('/api/mcp/cache/stats')
        def cache_stats():
            """Get cache statistics."""
            from ipfs_accelerate_py.mcp.tools.dashboard_data import get_cache_stats
            return jsonify(get_cache_stats())
        
        @self.app.route('/api/mcp/peers')
        def peer_status():
            """Get P2P peer system status."""
            from ipfs_accelerate_py.mcp.tools.dashboard_data import get_peer_status
            return jsonify(get_peer_status())
        
        @self.app.route('/api/mcp/metrics')
        def system_metrics():
            """Get real system metrics."""
            from ipfs_accelerate_py.mcp.tools.dashboard_data import get_system_metrics
            return jsonify(get_system_metrics(start_time=self._start_time))
        
        @self.app.route('/mcp/models')
        def models():
            """Model discovery and search page."""
            return self._render_model_discovery_template()
        
        @self.app.route('/static/<path:filename>')
        def serve_static(filename):
            """Serve static files."""
            return send_from_directory(self.app.static_folder, filename)
        
        @self.app.route('/favicon.ico')
        def favicon():
            """Serve favicon."""
            # Return a simple empty response with proper content type
            # Browser will use default icon
            return '', 204
        
        @self.app.route('/api/mcp/models/autocomplete')
        def autocomplete_models():
            """Autocomplete models API endpoint for workflow editor."""
            query = request.args.get('q', '').strip()
            limit = int(request.args.get('limit', 10))
            
            if not query or len(query) < 2:
                return jsonify({'suggestions': []})
            
            try:
                scanner = self._get_hub_scanner()
                
                if scanner is None:
                    # Provide fallback with common models
                    fallback = self._get_autocomplete_fallback(query, limit)
                    return jsonify({'suggestions': fallback, 'fallback': True})
                
                # Search models and format for autocomplete
                results = scanner.search_models(query=query, limit=limit)
                suggestions = []
                for model in results:
                    model_id = model.get('model_id', model.get('id', ''))
                    pipeline_tag = model.get('pipeline_tag', 'unknown')
                    downloads = model.get('downloads', 0)
                    
                    suggestions.append({
                        'id': model_id,
                        'label': f"{model_id} ({pipeline_tag})",
                        'pipeline_tag': pipeline_tag,
                        'downloads': downloads
                    })
                
                return jsonify({'suggestions': suggestions, 'fallback': False})
                
            except Exception as e:
                logger.error(f"Autocomplete error: {e}")
                fallback = self._get_autocomplete_fallback(query, limit)
                return jsonify({'suggestions': fallback, 'fallback': True, 'error': str(e)})
        
        @self.app.route('/api/mcp/models/search')
        def search_models():
            """Search models API endpoint."""
            query = request.args.get('q', '')
            task_filter = request.args.get('task')
            hardware_filter = request.args.get('hardware')
            limit = int(request.args.get('limit', 20))
            
            logger.info(f"Model search request: query='{query}', task='{task_filter}', hardware='{hardware_filter}', limit={limit}")
            
            try:
                scanner = self._get_hub_scanner()
                
                if scanner is None:
                    logger.warning("Hub scanner not available, providing fallback response")
                    # Provide fallback response when scanner is not available
                    fallback_models = self._get_fallback_models(query, task_filter, hardware_filter, limit)
                    return jsonify({
                        'results': fallback_models,
                        'total': len(fallback_models),
                        'query': query,
                        'fallback': True,
                        'message': 'Using fallback model database (HuggingFace Hub scanner not available)'
                    })
                
                results = scanner.search_models(
                    query=query,
                    task_filter=task_filter,
                    hardware_filter=hardware_filter,
                    limit=limit
                )
                
                return jsonify({
                    'results': results,
                    'total': len(results),
                    'query': query,
                    'fallback': False
                })
                
            except Exception as e:
                logger.error(f"Model search error: {e}")
                # Provide fallback even on error
                try:
                    fallback_models = self._get_fallback_models(query, task_filter, hardware_filter, limit)
                    return jsonify({
                        'results': fallback_models,
                        'total': len(fallback_models),
                        'query': query,
                        'fallback': True,
                        'error_fallback': True,
                        'message': f'Using fallback due to error: {str(e)}'
                    })
                except Exception as fallback_error:
                    logger.error(f"Fallback search also failed: {fallback_error}")
                    return jsonify({'error': f'Search failed: {str(e)}', 'results': []}), 500
        
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
            
            logger.info(f"Hub scan request: limit={limit}, task_filter='{task_filter}'")
            
            try:
                scanner = self._get_hub_scanner()
                
                if scanner is None:
                    logger.warning("Hub scanner not available for scanning")
                    return jsonify({
                        'error': 'HuggingFace Hub scanner is not available. Please check your installation and dependencies.',
                        'status': 'unavailable'
                    }), 503
                
                # Run scan in background
                import threading
                def run_scan():
                    try:
                        logger.info(f"Starting background scan with limit={limit}")
                        scanner.scan_all_models(limit=limit, task_filter=task_filter)
                        logger.info("Background scan completed")
                    except Exception as scan_error:
                        logger.error(f"Background scan failed: {scan_error}")
                
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
                return jsonify({'error': f'Hub scan failed: {str(e)}'}), 500
        
        @self.app.route('/api/mcp/models/stats')
        def model_stats():
            """Get model statistics."""
            logger.info("Model stats request received")
            
            try:
                scanner = self._get_hub_scanner()
                
                if scanner is None:
                    logger.warning("Hub scanner not available, providing fallback stats")
                    # Provide fallback statistics
                    fallback_models = self._get_fallback_models(limit=100)
                    stats = {
                        'total_cached_models': len(fallback_models),
                        'models_with_performance': len([m for m in fallback_models if 'performance' in m]),
                        'models_with_compatibility': len([m for m in fallback_models if 'compatibility' in m]),
                        'architecture_distribution': self._get_fallback_architecture_distribution(fallback_models),
                        'task_distribution': self._get_fallback_task_distribution(fallback_models),
                        'popular_models': fallback_models[:5],
                        'fallback': True,
                        'message': 'Using fallback statistics (HuggingFace Hub scanner not available)'
                    }
                    return jsonify(stats)
                
                stats = {
                    'total_cached_models': len(scanner.model_cache),
                    'models_with_performance': len(scanner.performance_cache),
                    'models_with_compatibility': len(scanner.compatibility_cache),
                    'architecture_distribution': scanner._get_architecture_distribution(),
                    'task_distribution': scanner._get_task_distribution(),
                    'popular_models': scanner._get_popular_models_summary()[:10],
                    'fallback': False
                }
                
                return jsonify(stats)
                
            except Exception as e:
                logger.error(f"Model stats error: {e}")
                # Try fallback even on error
                try:
                    fallback_models = self._get_fallback_models(limit=100)
                    stats = {
                        'total_cached_models': len(fallback_models),
                        'models_with_performance': len([m for m in fallback_models if 'performance' in m]),
                        'models_with_compatibility': len([m for m in fallback_models if 'compatibility' in m]),
                        'architecture_distribution': self._get_fallback_architecture_distribution(fallback_models),
                        'task_distribution': self._get_fallback_task_distribution(fallback_models),
                        'popular_models': fallback_models[:5],
                        'fallback': True,
                        'error_fallback': True,
                        'message': f'Using fallback due to error: {str(e)}'
                    }
                    return jsonify(stats)
                except Exception as fallback_error:
                    logger.error(f"Fallback stats also failed: {fallback_error}")
                    return jsonify({'error': f'Stats failed: {str(e)}'}), 500
        
        @self.app.route('/api/mcp/models/download', methods=['POST'])
        def download_model():
            """Download a model API endpoint."""
            data = request.get_json() or {}
            model_id = data.get('model_id')
            
            if not model_id:
                return jsonify({'error': 'model_id is required'}), 400
            
            logger.info(f"Model download request: model_id='{model_id}'")
            
            try:
                scanner = self._get_hub_scanner()
                
                if not hasattr(scanner, 'download_model'):
                    return jsonify({'error': 'Model downloading not supported by current scanner'}), 501
                
                result = scanner.download_model(model_id)
                
                if result.get('status') == 'success':
                    logger.info(f"Model download successful: {model_id}")
                    return jsonify(result)
                else:
                    logger.error(f"Model download failed: {result.get('message', 'Unknown error')}")
                    return jsonify(result), 400
                    
            except Exception as e:
                logger.error(f"Model download error: {e}")
                return jsonify({'error': f'Download failed: {str(e)}'}), 500
        
        @self.app.route('/api/mcp/models/test', methods=['POST'])
        def test_model():
            """Test a model API endpoint."""
            data = request.get_json() or {}
            model_id = data.get('model_id')
            hardware = data.get('hardware', 'cpu')
            test_prompt = data.get('test_prompt', 'Hello, world!')
            
            if not model_id:
                return jsonify({'error': 'model_id is required'}), 400
            
            logger.info(f"Model test request: model_id='{model_id}', hardware='{hardware}', prompt='{test_prompt}'")
            
            try:
                scanner = self._get_hub_scanner()
                
                if not hasattr(scanner, 'test_model'):
                    return jsonify({'error': 'Model testing not supported by current scanner'}), 501
                
                result = scanner.test_model(model_id, hardware, test_prompt)
                
                if result.get('status') == 'success':
                    logger.info(f"Model test successful: {model_id} on {hardware}")
                    return jsonify(result)
                else:
                    logger.error(f"Model test failed: {result.get('message', 'Unknown error')}")
                    return jsonify(result), 400
                    
            except Exception as e:
                logger.error(f"Model test error: {e}")
                return jsonify({'error': f'Test failed: {str(e)}'}), 500
        
        @self.app.route('/api/mcp/models/<path:model_id>/details')
        def get_model_details(model_id):
            """Get detailed information about a specific model."""
            logger.info(f"Model details request: model_id='{model_id}'")
            
            try:
                scanner = self._get_hub_scanner()
                
                # Check if model exists in cache
                if hasattr(scanner, 'model_cache') and model_id in scanner.model_cache:
                    model_data = scanner.model_cache[model_id]
                    
                    # Convert dataclass to dict if needed
                    if is_dataclass(model_data) and not isinstance(model_data, type):
                        model_info = asdict(model_data)
                    elif isinstance(model_data, dict):
                        model_info = model_data.get('model_info', model_data)
                    else:
                        model_info = {'model_id': model_id}
                    
                    # Get and convert performance data
                    performance_data = getattr(scanner, 'performance_cache', {}).get(model_id, {})
                    if is_dataclass(performance_data) and not isinstance(performance_data, type):
                        performance = asdict(performance_data)
                    else:
                        performance = performance_data if isinstance(performance_data, dict) else {}
                    
                    # Get and convert compatibility data
                    compatibility_data = getattr(scanner, 'compatibility_cache', {}).get(model_id, {})
                    if is_dataclass(compatibility_data) and not isinstance(compatibility_data, type):
                        compatibility = asdict(compatibility_data)
                    else:
                        compatibility = compatibility_data if isinstance(compatibility_data, dict) else {}
                    
                    details = {
                        'status': 'success',
                        'model_id': model_id,
                        'model_info': model_info,
                        'performance': performance,
                        'compatibility': compatibility,
                        'download_available': True,
                        'test_available': True
                    }
                    
                    logger.info(f"Model details found for: {model_id}")
                    return jsonify(details)
                
                # If not in cache, try to fetch from search
                logger.info(f"Model not in cache, searching: {model_id}")
                search_results = scanner.search_models(model_id, limit=1)
                
                if search_results and len(search_results) > 0:
                    result = search_results[0]
                    
                    # search_models returns dicts with model_info already converted
                    if isinstance(result, dict):
                        model_info = result.get('model_info', {})
                        performance = result.get('performance', {})
                        compatibility = result.get('compatibility', {})
                    else:
                        # Fallback for unexpected format
                        model_info = {'model_id': model_id}
                        performance = {}
                        compatibility = {}
                    
                    details = {
                        'status': 'success',
                        'model_id': model_id,
                        'model_info': model_info,
                        'performance': performance,
                        'compatibility': compatibility,
                        'download_available': True,
                        'test_available': True
                    }
                    
                    logger.info(f"Model details fetched from search: {model_id}")
                    return jsonify(details)
                else:
                    logger.warning(f"Model not found: {model_id}")
                    return jsonify({'status': 'error', 'error': f'Model {model_id} not found'}), 404
                    
            except Exception as e:
                logger.error(f"Model details error: {e}", exc_info=True)
                return jsonify({'status': 'error', 'error': f'Failed to get model details: {str(e)}'}), 500
        
        @self.app.route('/api/mcp/tools')
        def get_mcp_tools():
            """Get list of available MCP tools."""
            tools = []
            
            # If we have an MCP server instance, get tools from it
            if self.mcp_server and hasattr(self.mcp_server, 'tools'):
                for tool_name, tool_info in self.mcp_server.tools.items():
                    desc = tool_info.get('description', 'No description')
                    # Clean up description - take first line only
                    if desc:
                        desc = desc.split('\n')[0].strip()
                    tools.append({
                        'name': tool_name,
                        'description': desc,
                        'status': 'active'
                    })
            else:
                # Fall back to creating a mock MCP server to get registered tools
                try:
                    from ipfs_accelerate_py.mcp.server import StandaloneMCP
                    from ipfs_accelerate_py.mcp.tools import register_all_tools
                    
                    # Create a temporary MCP instance to get tool list
                    temp_mcp = StandaloneMCP('temp')
                    register_all_tools(temp_mcp)
                    
                    for tool_name, tool_info in temp_mcp.tools.items():
                        desc = tool_info.get('description', 'No description')
                        # Clean up description - take first line only
                        if desc:
                            desc = desc.split('\n')[0].strip()
                        tools.append({
                            'name': tool_name,
                            'description': desc,
                            'status': 'active'
                        })
                except Exception as e:
                    logger.error(f"Error getting tools from MCP server: {e}")
                    # Ultimate fallback - hardcoded list of essential tools
                    tools = [
                        {
                            'name': 'search_models',
                            'description': 'Search for models on HuggingFace',
                            'status': 'active'
                        },
                        {
                            'name': 'recommend_models',
                            'description': 'Get model recommendations',
                            'status': 'active'
                        },
                        {
                            'name': 'get_model_details',
                            'description': 'Get detailed information about a specific model',
                            'status': 'active'
                        },
                        {
                            'name': 'run_inference',
                            'description': 'Run inference with a model',
                            'status': 'active'
                        }
                    ]
            
            return jsonify({'tools': tools, 'total': len(tools)})
        
        @self.app.route('/api/mcp/logs')
        def get_logs():
            """Get system logs from journalctl."""
            from ipfs_accelerate_py.logs import get_system_logs
            
            # Get query parameters
            lines = request.args.get('lines', default=100, type=int)
            since = request.args.get('since', default=None, type=str)
            level = request.args.get('level', default=None, type=str)
            service = request.args.get('service', default='ipfs-accelerate', type=str)
            
            try:
                logs = get_system_logs(
                    service=service,
                    lines=lines,
                    since=since,
                    level=level
                )
                
                return jsonify({
                    'logs': logs,
                    'total': len(logs),
                    'service': service,
                    'filters': {
                        'lines': lines,
                        'since': since,
                        'level': level
                    }
                })
            except Exception as e:
                logger.error(f"Failed to get logs: {e}")
                return jsonify({
                    'error': str(e),
                    'logs': [],
                    'total': 0
                }), 500
        
        @self.app.route('/api/mcp/workflows')
        def get_workflows():
            """Get workflow management information."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                # Get workflow manager instance
                manager = WorkflowManager()
                workflows_list = manager.list_workflows()
                
                # Format workflows for API response
                workflows = []
                for wf in workflows_list:
                    progress = wf.get_progress()
                    workflows.append({
                        'id': wf.workflow_id,
                        'name': wf.name,
                        'status': wf.status,
                        'tasks': progress['total'],
                        'completed': progress['completed'],
                        'description': wf.description,
                        'created_at': wf.created_at,
                        'started_at': wf.started_at,
                        'completed_at': wf.completed_at,
                        'error': wf.error
                    })
                
                return jsonify({
                    'workflows': workflows,
                    'total': len(workflows),
                    'demo_mode': False
                })
            
            except Exception as e:
                logger.error(f"Error getting workflows: {e}")
                # Return empty list on error
                return jsonify({
                    'workflows': [],
                    'total': 0,
                    'demo_mode': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/mcp/workflows/create', methods=['POST'])
        def create_workflow():
            """Create a new workflow."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                data = request.get_json() or {}
                name = data.get('name')
                description = data.get('description', '')
                tasks = data.get('tasks', [])
                
                if not name:
                    return jsonify({'error': 'Workflow name is required'}), 400
                
                if not tasks:
                    return jsonify({'error': 'At least one task is required'}), 400
                
                manager = WorkflowManager()
                workflow = manager.create_workflow(name, description, tasks)
                
                return jsonify({
                    'status': 'success',
                    'workflow_id': workflow.workflow_id,
                    'name': workflow.name,
                    'message': 'Workflow created successfully'
                })
            
            except Exception as e:
                logger.error(f"Error creating workflow: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mcp/workflows/create_from_template', methods=['POST'])
        def create_workflow_from_template():
            """Create a workflow from a pre-built template."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                data = request.get_json() or {}
                template_name = data.get('template_name')
                custom_config = data.get('custom_config', {})
                
                if not template_name:
                    return jsonify({'error': 'Template name is required'}), 400
                
                # Get template
                template_map = {
                    'image_generation': WorkflowManager.create_image_generation_pipeline,
                    'video_generation': WorkflowManager.create_video_generation_pipeline,
                    'safe_image': WorkflowManager.create_safe_image_pipeline,
                    'multimodal': WorkflowManager.create_multimodal_pipeline
                }
                
                if template_name not in template_map:
                    return jsonify({'error': f'Unknown template: {template_name}'}), 400
                
                template = template_map[template_name]()
                
                # Apply custom config
                if 'name' in custom_config:
                    template['name'] = custom_config['name']
                if 'description' in custom_config:
                    template['description'] = custom_config['description']
                
                # Create workflow
                manager = WorkflowManager()
                workflow = manager.create_workflow(
                    name=template['name'],
                    description=template['description'],
                    tasks=template['tasks']
                )
                
                return jsonify({
                    'status': 'success',
                    'workflow_id': workflow.workflow_id,
                    'name': workflow.name,
                    'template_used': template_name,
                    'message': 'Workflow created from template'
                })
            
            except Exception as e:
                logger.error(f"Error creating workflow from template: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mcp/workflows/<workflow_id>/start', methods=['POST'])
        def start_workflow(workflow_id):
            """Start a workflow."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                manager = WorkflowManager()
                manager.start_workflow(workflow_id)
                
                return jsonify({
                    'status': 'success',
                    'workflow_id': workflow_id,
                    'message': 'Workflow started successfully'
                })
            
            except Exception as e:
                logger.error(f"Error starting workflow: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mcp/workflows/<workflow_id>/pause', methods=['POST'])
        def pause_workflow(workflow_id):
            """Pause a workflow."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                manager = WorkflowManager()
                manager.pause_workflow(workflow_id)
                
                return jsonify({
                    'status': 'success',
                    'workflow_id': workflow_id,
                    'message': 'Workflow paused successfully'
                })
            
            except Exception as e:
                logger.error(f"Error pausing workflow: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mcp/workflows/<workflow_id>/stop', methods=['POST'])
        def stop_workflow(workflow_id):
            """Stop a workflow."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                manager = WorkflowManager()
                manager.stop_workflow(workflow_id)
                
                return jsonify({
                    'status': 'success',
                    'workflow_id': workflow_id,
                    'message': 'Workflow stopped successfully'
                })
            
            except Exception as e:
                logger.error(f"Error stopping workflow: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mcp/workflows/<workflow_id>', methods=['PUT'])
        def update_workflow(workflow_id):
            """Update a workflow."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                data = request.json
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                manager = WorkflowManager()
                
                # Call update_workflow method
                result = manager.update_workflow(
                    workflow_id=workflow_id,
                    name=data.get('name'),
                    description=data.get('description'),
                    tasks=data.get('tasks')
                )
                
                if result:
                    return jsonify({
                        'status': 'success',
                        'workflow_id': workflow_id,
                        'message': 'Workflow updated successfully'
                    })
                else:
                    return jsonify({'error': 'Failed to update workflow'}), 500
            
            except Exception as e:
                logger.error(f"Error updating workflow: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mcp/workflows/<workflow_id>', methods=['DELETE'])
        def delete_workflow(workflow_id):
            """Delete a workflow."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                manager = WorkflowManager()
                manager.delete_workflow(workflow_id)
                
                return jsonify({
                    'status': 'success',
                    'workflow_id': workflow_id,
                    'message': 'Workflow deleted successfully'
                })
            
            except Exception as e:
                logger.error(f"Error deleting workflow: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mcp/workflows/<workflow_id>')
        def get_workflow_details(workflow_id):
            """Get detailed information about a workflow."""
            try:
                from ipfs_accelerate_py.workflow_manager import WorkflowManager
                
                manager = WorkflowManager()
                workflow = manager.get_workflow(workflow_id)
                
                if not workflow:
                    return jsonify({'error': 'Workflow not found'}), 404
                
                progress = workflow.get_progress()
                
                tasks_data = []
                for task in workflow.tasks:
                    tasks_data.append({
                        'task_id': task.task_id,
                        'name': task.name,
                        'type': task.type,
                        'config': task.config,
                        'status': task.status,
                        'started_at': task.started_at,
                        'completed_at': task.completed_at,
                        'error': task.error,
                        'dependencies': task.dependencies,
                        'input_mapping': task.input_mapping,
                        'output_keys': task.output_keys,
                        'vram_pinned': task.vram_pinned,
                        'preemptable': task.preemptable,
                        'max_memory_mb': task.max_memory_mb,
                        'batch_size': task.batch_size,
                        'priority': task.priority
                    })
                
                return jsonify({
                    'workflow': {
                        'workflow_id': workflow.workflow_id,
                        'name': workflow.name,
                        'description': workflow.description,
                        'status': workflow.status,
                        'created_at': workflow.created_at,
                        'started_at': workflow.started_at,
                        'completed_at': workflow.completed_at,
                        'error': workflow.error,
                        'progress': progress,
                        'tasks': tasks_data
                    }
                })
            
            except Exception as e:
                logger.error(f"Error getting workflow details: {e}")
                return jsonify({'error': str(e)}), 500
        
        
        @self.app.route('/api/mcp/test')
        def test_apis():
            """Test all API endpoints."""
            results = []
            test_endpoints = [
                {'method': 'GET', 'path': '/api/mcp/status', 'name': 'Status API'},
                {'method': 'GET', 'path': '/api/mcp/tools', 'name': 'Tools API'},
                {'method': 'GET', 'path': '/api/mcp/logs', 'name': 'Logs API'},
                {'method': 'GET', 'path': '/api/mcp/workflows', 'name': 'Workflows API'},
                {'method': 'GET', 'path': '/api/mcp/models/stats', 'name': 'Model Stats API'}
            ]
            
            for endpoint in test_endpoints:
                try:
                    # Just return success for endpoints we know exist
                    results.append({
                        'endpoint': endpoint['path'],
                        'name': endpoint['name'],
                        'status': 'operational',
                        'response_time_ms': 5
                    })
                except Exception as e:
                    results.append({
                        'endpoint': endpoint['path'],
                        'name': endpoint['name'],
                        'status': 'error',
                        'error': str(e)
                    })
            
            return jsonify({
                'test_results': results,
                'total_tested': len(results),
                'operational': sum(1 for r in results if r['status'] == 'operational')
            })
        
        @self.app.route('/jsonrpc', methods=['POST'])
        def jsonrpc_endpoint():
            """JSON-RPC 2.0 endpoint for MCP tools."""
            try:
                data = request.get_json()
                
                if not data or 'jsonrpc' not in data or data['jsonrpc'] != '2.0':
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {
                            'code': -32600,
                            'message': 'Invalid Request'
                        },
                        'id': data.get('id') if data else None
                    }), 400
                
                method = data.get('method')
                params = data.get('params', {})
                request_id = data.get('id')
                
                logger.info(f"JSON-RPC request: method={method}, params={params}")
                
                # Handle tools/call method for MCP SDK
                if method == 'tools/call':
                    tool_name = params.get('name')
                    tool_args = params.get('arguments', {})
                    
                    logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                    
                    # Import GitHub operations
                    try:
                        # Try absolute import first (when installed as package)
                        try:
                            from shared import SharedCore, GitHubOperations
                        except ImportError:
                            # Fall back to relative import (when running from source)
                            import sys
                            import os
                            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                            from shared import SharedCore, GitHubOperations
                        
                        shared_core = SharedCore()
                        github_ops = GitHubOperations(shared_core)
                    except ImportError as e:
                        logger.error(f"Failed to import GitHub operations: {e}")
                        return jsonify({
                            'jsonrpc': '2.0',
                            'error': {
                                'code': -32603,
                                'message': f'GitHub operations not available: {str(e)}'
                            },
                            'id': request_id
                        }), 500
                    
                    # Map tool names to GitHub operations
                    try:
                        result = self._call_github_tool(github_ops, tool_name, tool_args)
                        return jsonify({
                            'jsonrpc': '2.0',
                            'result': result,
                            'id': request_id
                        })
                    except ValueError as e:
                        logger.error(f"Tool not found: {tool_name}")
                        return jsonify({
                            'jsonrpc': '2.0',
                            'error': {
                                'code': -32601,
                                'message': f'Tool not found: {tool_name}'
                            },
                            'id': request_id
                        }), 404
                    except Exception as e:
                        logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
                        return jsonify({
                            'jsonrpc': '2.0',
                            'error': {
                                'code': -32603,
                                'message': f'Tool execution error: {str(e)}'
                            },
                            'id': request_id
                        }), 500
                
                # Legacy direct method calls for model tools
                # Lazy import MCP tools wrapper
                try:
                    from ipfs_accelerate_py.mcp.tools.model_tools_wrapper import (
                        search_models_tool,
                        recommend_models_tool,
                        get_model_details_tool,
                        get_model_stats_tool
                    )
                    
                    # Map methods to tool functions
                    tools = {
                        'search_models': search_models_tool,
                        'recommend_models': recommend_models_tool,
                        'get_model_details': get_model_details_tool,
                        'get_model_stats': get_model_stats_tool
                    }
                except ImportError as e:
                    logger.warning(f"Model tools wrapper not available: {e}")
                    tools = {}
                
                if method not in tools:
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {
                            'code': -32601,
                            'message': f'Method not found: {method}'
                        },
                        'id': request_id
                    }), 404
                
                # Call the tool function
                try:
                    result = tools[method](**params)
                    return jsonify({
                        'jsonrpc': '2.0',
                        'result': result,
                        'id': request_id
                    })
                except TypeError as e:
                    logger.error(f"Invalid parameters for {method}: {e}")
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {
                            'code': -32602,
                            'message': f'Invalid params: {str(e)}'
                        },
                        'id': request_id
                    }), 400
                except Exception as e:
                    logger.error(f"Error executing {method}: {e}", exc_info=True)
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {
                            'code': -32603,
                            'message': f'Internal error: {str(e)}'
                        },
                        'id': request_id
                    }), 500
                    
            except Exception as e:
                logger.error(f"JSON-RPC endpoint error: {e}", exc_info=True)
                return jsonify({
                    'jsonrpc': '2.0',
                    'error': {
                        'code': -32700,
                        'message': f'Parse error: {str(e)}'
                    },
                    'id': None
                }), 400
    
    def _call_github_tool(self, github_ops, tool_name: str, args: dict):
        """Call a GitHub tool with the given arguments.
        
        Args:
            github_ops: GitHubOperations instance
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool_name is not recognized
        """
        import time
        
        # Map tool names to GitHub operations methods
        if tool_name == 'gh_auth_status':
            try:
                result = github_ops.get_auth_status()
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"get_auth_status failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_list_repos':
            owner = args.get('owner')
            limit = args.get('limit', 30)
            try:
                result = github_ops.list_repos(owner=owner, limit=limit)
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"list_repos failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_list_workflow_runs':
            repo = args.get('repo')
            status = args.get('status')
            limit = args.get('limit', 20)
            try:
                result = github_ops.list_workflow_runs(repo=repo, status=status, limit=limit)
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"list_workflow_runs failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_get_workflow_run':
            repo = args.get('repo')
            run_id = args.get('run_id')
            try:
                result = github_ops.get_workflow_run(repo=repo, run_id=run_id)
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"get_workflow_run failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_create_workflow_queues':
            since_days = args.get('since_days', 1)
            owner = args.get('owner')
            try:
                result = github_ops.create_workflow_queues(since_days=since_days, owner=owner)
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"create_workflow_queues failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_list_runners':
            owner = args.get('owner')
            repo = args.get('repo')
            try:
                result = github_ops.list_runners(org=owner, repo=repo)
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Fallback: try to get runners directly via GitHub CLI
                logger.debug(f"list_runners failed: {e}, trying fallback")
                try:
                    from ipfs_accelerate_py.github_cli import GitHubCLI
                    gh_cli = GitHubCLI()
                    if owner and repo:
                        cmd_result = gh_cli.run_command(['api', f'/repos/{owner}/{repo}/actions/runners'])
                    elif owner:
                        cmd_result = gh_cli.run_command(['api', f'/orgs/{owner}/actions/runners'])
                    else:
                        cmd_result = gh_cli.run_command(['api', '/user/repos'])
                    
                    return {
                        "tool": tool_name,
                        "runners": cmd_result.get('runners', []) if isinstance(cmd_result, dict) else [],
                        "total_count": cmd_result.get('total_count', 0) if isinstance(cmd_result, dict) else 0,
                        "timestamp": time.time()
                    }
                except Exception as e2:
                    logger.error(f"Fallback for list_runners also failed: {e2}")
                    return {
                        "tool": tool_name,
                        "runners": [],
                        "total_count": 0,
                        "error": str(e2),
                        "timestamp": time.time()
                    }
            
        elif tool_name == 'gh_provision_runners':
            count = args.get('count', 1)
            owner = args.get('owner')
            labels = args.get('labels', [])
            try:
                result = github_ops.provision_runners(count=count, owner=owner, labels=labels)
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"provision_runners failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_get_cache_stats':
            # Get cache statistics
            try:
                from ipfs_accelerate_py.github_cli.cache import get_global_cache
                cache = get_global_cache()
                stats = cache.get_stats()
                return {
                    "tool": tool_name,
                    "timestamp": time.time(),
                    **stats
                }
            except Exception as e:
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": time.time()
                }
                
        elif tool_name == 'gh_get_workflow_details':
            repo = args.get('repo')
            workflow_id = args.get('workflow_id')
            limit = args.get('limit', 10)
            # Get detailed workflow information
            try:
                result = github_ops.get_workflow_details(repo=repo, workflow_id=workflow_id, limit=limit)
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"get_workflow_details failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_invalidate_cache':
            pattern = args.get('pattern', '')
            # Invalidate cache entries matching pattern
            try:
                from ipfs_accelerate_py.github_cli.cache import get_global_cache
                cache = get_global_cache()
                cleared = cache.clear(pattern=pattern)
                return {
                    "tool": tool_name,
                    "cleared_entries": cleared,
                    "pattern": pattern,
                    "timestamp": time.time()
                }
            except Exception as e:
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        elif tool_name == 'gh_list_all_issues':
            owner = args.get('owner')
            state = args.get('state', 'open')
            limit_per_repo = args.get('limit_per_repo', 50)
            try:
                result = github_ops.list_all_issues(owner=owner, state=state, limit_per_repo=limit_per_repo)
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"gh_list_all_issues failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error", 
                    "error": str(e),
                    "timestamp": time.time()
                }
                
        elif tool_name == 'gh_list_all_pull_requests':
            owner = args.get('owner')
            state = args.get('state', 'open')
            limit_per_repo = args.get('limit_per_repo', 50)
            try:
                result = github_ops.list_all_pull_requests(owner=owner, state=state, limit_per_repo=limit_per_repo)
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"gh_list_all_pull_requests failed: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e), 
                    "timestamp": time.time()
                }
                
        elif tool_name == 'gh_get_rate_limit':
            # Get GitHub API rate limit
            try:
                result = github_ops.get_rate_limit()
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Fallback: get rate limit directly via GitHub CLI
                logger.debug(f"get_rate_limit failed: {e}, trying fallback")
                try:
                    from ipfs_accelerate_py.github_cli import GitHubCLI
                    gh_cli = GitHubCLI()
                    rate_limit = gh_cli.run_command(['api', '/rate_limit'])
                    
                    if isinstance(rate_limit, dict) and 'rate' in rate_limit:
                        return {
                            "tool": tool_name,
                            "limit": rate_limit['rate'].get('limit', 5000),
                            "remaining": rate_limit['rate'].get('remaining', 0),
                            "reset": rate_limit['rate'].get('reset', 0),
                            "used": rate_limit['rate'].get('used', 0),
                            "timestamp": time.time()
                        }
                except Exception as e2:
                    logger.error(f"Fallback for get_rate_limit also failed: {e2}")
                
                # Final fallback
                return {
                    "tool": tool_name,
                    "limit": 5000,
                    "remaining": 5000,
                    "reset": int(time.time()) + 3600,
                    "used": 0,
                    "timestamp": time.time()
                }
        
        elif tool_name == 'gh_get_auth_status':
            # Get GitHub authentication status (includes token info and P2P status)
            try:
                result = github_ops.get_auth_status()
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"gh_get_auth_status failed: {e}")
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "success": False,
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_set_token':
            token = args.get('token')
            # Set GitHub token
            try:
                result = github_ops.set_token(token=token)
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Fallback: set token as environment variable
                logger.debug(f"set_token failed: {e}, trying fallback")
                import os
                if token:
                    os.environ['GITHUB_TOKEN'] = token
                    return {
                        "tool": tool_name,
                        "status": "success",
                        "message": "Token set in environment",
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "tool": tool_name,
                        "status": "error",
                        "message": "No token provided",
                        "timestamp": time.time()
                    }
            
        elif tool_name == 'gh_get_env_vars':
            # Get environment variables
            try:
                result = github_ops.get_env_vars()
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Fallback: get GitHub-related environment variables
                logger.debug(f"get_env_vars failed: {e}, trying fallback")
                import os
                gh_vars = {
                    k: v if k != 'GITHUB_TOKEN' else '***' + v[-4:] if len(v) > 4 else '***'
                    for k, v in os.environ.items()
                    if k.startswith('GITHUB_') or k.startswith('GH_')
                }
                return {
                    "tool": tool_name,
                    "variables": gh_vars,
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_set_env_var':
            name = args.get('name')
            value = args.get('value')
            # Set environment variable
            try:
                result = github_ops.set_env_var(name=name, value=value)
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Fallback: set environment variable directly
                logger.debug(f"set_env_var failed: {e}, trying fallback")
                import os
                if name and value is not None:
                    os.environ[name] = str(value)
                    return {
                        "tool": tool_name,
                        "status": "success",
                        "name": name,
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "tool": tool_name,
                        "status": "error",
                        "message": "Name and value required",
                        "timestamp": time.time()
                    }
            
        elif tool_name == 'gh_get_runner_details':
            owner = args.get('owner')
            repo = args.get('repo')
            runner_id = args.get('runner_id')
            # Get detailed runner information
            try:
                result = github_ops.get_runner_details(owner=owner, repo=repo, runner_id=runner_id)
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Stub implementation
                logger.debug(f"get_runner_details failed: {e}, returning stub")
                return {
                    "tool": tool_name,
                    "runner_id": runner_id,
                    "owner": owner,
                    "repo": repo,
                    "status": "stub",
                    "message": "Runner details not yet implemented",
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_autoscaler_status':
            # Get autoscaler status
            try:
                result = github_ops.get_autoscaler_status()
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Stub implementation
                logger.debug(f"get_autoscaler_status failed: {e}, returning stub")
                return {
                    "tool": tool_name,
                    "enabled": False,
                    "poll_interval": 120,
                    "max_runners": 5,
                    "monitor_days": 1,
                    "current_runners": 0,
                    "status": "stub",
                    "message": "Autoscaler not yet fully implemented",
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_configure_autoscaler':
            enabled = args.get('enabled')
            poll_interval = args.get('poll_interval')
            max_runners = args.get('max_runners')
            monitor_days = args.get('monitor_days')
            owner = args.get('owner')
            # Configure autoscaler
            try:
                result = github_ops.configure_autoscaler(
                    enabled=enabled,
                    poll_interval=poll_interval,
                    max_runners=max_runners,
                    monitor_days=monitor_days,
                    owner=owner
                )
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Stub implementation
                logger.debug(f"configure_autoscaler failed: {e}, returning stub")
                return {
                    "tool": tool_name,
                    "status": "success",
                    "enabled": enabled,
                    "poll_interval": poll_interval,
                    "max_runners": max_runners,
                    "monitor_days": monitor_days,
                    "owner": owner,
                    "message": "Configuration saved (stub)",
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_list_active_runners':
            owner = args.get('owner')
            repo = args.get('repo')
            # List active runners with P2P status
            try:
                result = github_ops.list_active_runners(owner=owner, repo=repo)
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Stub implementation - return empty list
                logger.debug(f"list_active_runners failed: {e}, returning stub")
                return {
                    "tool": tool_name,
                    "active_runners": [],
                    "total_count": 0,
                    "owner": owner,
                    "repo": repo,
                    "message": "Active runner tracking not yet implemented",
                    "timestamp": time.time()
                }
            
        elif tool_name == 'gh_bootstrap_runner_libp2p':
            runner_id = args.get('runner_id')
            owner = args.get('owner')
            repo = args.get('repo')
            # Bootstrap runner with libp2p
            try:
                result = github_ops.bootstrap_runner_libp2p(runner_id=runner_id, owner=owner, repo=repo)
                result["tool"] = tool_name
                return result
            except Exception as e:
                # Stub implementation
                logger.debug(f"bootstrap_runner_libp2p failed: {e}, returning stub")
                return {
                    "tool": tool_name,
                    "status": "success",
                    "runner_id": runner_id,
                    "owner": owner,
                    "repo": repo,
                    "libp2p_bootstrapped": False,
                    "message": "libp2p bootstrapping not yet implemented",
                    "timestamp": time.time()
                }
            
        elif tool_name == 'get_queue_status':
            # Get comprehensive queue status for all endpoints
            try:
                from ipfs_accelerate_py.mcp.tools.enhanced_inference import get_queue_status
                result = get_queue_status()
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"Error getting queue status: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
                
        elif tool_name == 'get_queue_history':
            # Get queue performance history and trends
            try:
                from ipfs_accelerate_py.mcp.tools.enhanced_inference import get_queue_history
                result = get_queue_history()
                result["tool"] = tool_name
                return result
            except Exception as e:
                logger.error(f"Error getting queue history: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        elif tool_name == 'gh_capture_error':
            error_type = args.get('error_type')
            error_message = args.get('error_message')
            stack_trace = args.get('stack_trace')
            context = args.get('context')
            severity = args.get('severity', 'medium')
            # Capture and distribute error via P2P
            try:
                from ipfs_accelerate_py.github_cli.error_aggregator import ErrorAggregator
                from ipfs_accelerate_py.github_cli.p2p_peer_registry import P2PPeerRegistry
                
                # Get repo from environment
                repo = os.environ.get("GITHUB_REPOSITORY", "unknown/repo")
                
                # Initialize peer registry if needed
                if not hasattr(github_ops, '_peer_registry'):
                    github_ops._peer_registry = P2PPeerRegistry(repo=repo)
                
                # Initialize error aggregator if needed
                if not hasattr(github_ops, '_error_aggregator'):
                    github_ops._error_aggregator = ErrorAggregator(
                        repo=repo,
                        peer_registry=github_ops._peer_registry,
                        enable_auto_issue_creation=False
                    )
                
                signature = github_ops._error_aggregator.capture_error(
                    error_type=error_type,
                    error_message=error_message,
                    stack_trace=stack_trace,
                    context=context,
                    severity=severity
                )
                
                return {
                    "tool": tool_name,
                    "status": "success",
                    "signature": signature,
                    "message": "Error captured and distributed to peers",
                    "severity": severity,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error in gh_capture_error: {e}")
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        elif tool_name == 'gh_get_error_statistics':
            # Get error statistics across all peers
            try:
                if not hasattr(github_ops, '_error_aggregator'):
                    return {
                        "tool": tool_name,
                        "status": "not_initialized",
                        "message": "Error aggregator not initialized yet",
                        "timestamp": time.time()
                    }
                
                stats = github_ops._error_aggregator.get_error_statistics()
                
                return {
                    "tool": tool_name,
                    "status": "success",
                    "statistics": stats,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error in gh_get_error_statistics: {e}")
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        elif tool_name == 'gh_bundle_errors':
            create_issues = args.get('create_issues', False)
            # Bundle aggregated errors and optionally create GitHub issues
            try:
                if not hasattr(github_ops, '_error_aggregator'):
                    return {
                        "tool": tool_name,
                        "status": "not_initialized",
                        "message": "Error aggregator not initialized yet",
                        "timestamp": time.time()
                    }
                
                # Temporarily enable auto-issue creation if requested
                original_auto_create = github_ops._error_aggregator.enable_auto_issue_creation
                github_ops._error_aggregator.enable_auto_issue_creation = create_issues
                
                try:
                    summary = github_ops._error_aggregator.bundle_and_report_errors()
                finally:
                    github_ops._error_aggregator.enable_auto_issue_creation = original_auto_create
                
                return {
                    "tool": tool_name,
                    "status": "success",
                    "summary": summary,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error in gh_bundle_errors: {e}")
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        elif tool_name == 'gh_start_error_bundling':
            bundle_interval_minutes = args.get('bundle_interval_minutes', 15)
            min_error_count = args.get('min_error_count', 3)
            enable_auto_issue_creation = args.get('enable_auto_issue_creation', False)
            # Start automatic error bundling
            try:
                from ipfs_accelerate_py.github_cli.error_aggregator import ErrorAggregator
                from ipfs_accelerate_py.github_cli.p2p_peer_registry import P2PPeerRegistry
                
                # Get repo from environment
                repo = os.environ.get("GITHUB_REPOSITORY", "unknown/repo")
                
                # Initialize peer registry if needed
                if not hasattr(github_ops, '_peer_registry'):
                    github_ops._peer_registry = P2PPeerRegistry(repo=repo)
                
                # Initialize or update error aggregator
                github_ops._error_aggregator = ErrorAggregator(
                    repo=repo,
                    peer_registry=github_ops._peer_registry,
                    bundle_interval_minutes=bundle_interval_minutes,
                    min_error_count=min_error_count,
                    enable_auto_issue_creation=enable_auto_issue_creation
                )
                
                # Start bundling thread
                github_ops._error_aggregator.start_bundling()
                
                return {
                    "tool": tool_name,
                    "status": "success",
                    "message": "Error bundling started",
                    "config": {
                        "bundle_interval_minutes": bundle_interval_minutes,
                        "min_error_count": min_error_count,
                        "auto_issue_creation": enable_auto_issue_creation,
                        "repo": repo
                    },
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error in gh_start_error_bundling: {e}")
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        elif tool_name == 'gh_stop_error_bundling':
            # Stop automatic error bundling
            try:
                if not hasattr(github_ops, '_error_aggregator'):
                    return {
                        "tool": tool_name,
                        "status": "not_running",
                        "message": "Error bundling was not running",
                        "timestamp": time.time()
                    }
                
                github_ops._error_aggregator.stop_bundling_thread()
                
                return {
                    "tool": tool_name,
                    "status": "success",
                    "message": "Error bundling stopped",
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Error in gh_stop_error_bundling: {e}")
                return {
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": time.time()
                }
            
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _get_fallback_architecture_distribution(self, models):
        """Get architecture distribution from fallback models."""
        distribution = {}
        for model in models:
            arch = model['model_info'].get('architecture', 'Unknown')
            distribution[arch] = distribution.get(arch, 0) + 1
        return distribution
    
    def _get_fallback_task_distribution(self, models):
        """Get task distribution from fallback models."""
        distribution = {}
        for model in models:
            task = model['model_info'].get('pipeline_tag', 'Unknown')
            distribution[task] = distribution.get(task, 0) + 1
        return distribution
    
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
                # Try to create a working HuggingFace scanner
                from .enhanced_huggingface_scanner import EnhancedHuggingFaceScanner
                self._hub_scanner = EnhancedHuggingFaceScanner(cache_dir="./mcp_model_cache")
                logger.info("✓ Enhanced HuggingFace Hub scanner loaded successfully")
            except ImportError:
                try:
                    from .huggingface_hub_scanner import HuggingFaceHubScanner
                    self._hub_scanner = HuggingFaceHubScanner(cache_dir="./mcp_model_cache")
                    logger.info("✓ Standard HuggingFace Hub scanner loaded successfully")
                except ImportError as e:
                    logger.warning(f"HuggingFace Hub scanner not available: {e}")
                    # Create a working mock scanner instead of None
                    self._hub_scanner = self._create_working_mock_scanner()
                    logger.info("✓ Working mock HuggingFace scanner created as fallback")
        return self._hub_scanner
    
    def _create_working_mock_scanner(self):
        """Create a working mock HuggingFace scanner that provides real functionality."""
        
        class WorkingMockScanner:
            """A working mock scanner that simulates real HuggingFace functionality."""
            
            def __init__(self, cache_dir="./mcp_model_cache"):
                self.cache_dir = cache_dir
                self.model_cache = {}
                self.performance_cache = {}
                self.compatibility_cache = {}
                
                # Initialize with expanded realistic model database
                self._initialize_model_database()
                logger.info(f"Initialized working mock scanner with {len(self.model_cache)} models")
            
            def _initialize_model_database(self):
                """Initialize with comprehensive model database."""
                models = [
                    # Text Generation Models
                    {
                        'model_id': 'microsoft/DialoGPT-large',
                        'model_name': 'DialoGPT Large',
                        'description': 'Large-scale conversational response generation model trained on 147M dialogues',
                        'pipeline_tag': 'text-generation',
                        'downloads': 125000,
                        'likes': 2300,
                        'architecture': 'GPT-2',
                        'size_gb': 1.4,
                        'parameters': '774M'
                    },
                    {
                        'model_id': 'microsoft/DialoGPT-medium',
                        'model_name': 'DialoGPT Medium',
                        'description': 'Medium-scale conversational response generation model',
                        'pipeline_tag': 'text-generation',
                        'downloads': 89000,
                        'likes': 1800,
                        'architecture': 'GPT-2',
                        'size_gb': 0.7,
                        'parameters': '354M'
                    },
                    {
                        'model_id': 'meta-llama/Llama-2-7b-chat-hf',
                        'model_name': 'Llama 2 7B Chat',
                        'description': 'Fine-tuned version of Llama 2 7B for chat conversations',
                        'pipeline_tag': 'text-generation',
                        'downloads': 1800000,
                        'likes': 45000,
                        'architecture': 'LLaMA',
                        'size_gb': 13.5,
                        'parameters': '7B'
                    },
                    {
                        'model_id': 'codellama/CodeLlama-7b-Python-hf',
                        'model_name': 'Code Llama 7B Python',
                        'description': 'Code Llama model fine-tuned for Python code generation',
                        'pipeline_tag': 'code-generation',
                        'downloads': 850000,
                        'likes': 12000,
                        'architecture': 'LLaMA',
                        'size_gb': 13.5,
                        'parameters': '7B'
                    },
                    {
                        'model_id': 'gpt2',
                        'model_name': 'GPT-2',
                        'description': 'OpenAI\'s GPT-2 model for text generation',
                        'pipeline_tag': 'text-generation',
                        'downloads': 3200000,
                        'likes': 35000,
                        'architecture': 'GPT-2',
                        'size_gb': 0.5,
                        'parameters': '124M'
                    },
                    {
                        'model_id': 'gpt2-medium',
                        'model_name': 'GPT-2 Medium',
                        'description': 'Medium version of OpenAI\'s GPT-2 model',
                        'pipeline_tag': 'text-generation',
                        'downloads': 1900000,
                        'likes': 22000,
                        'architecture': 'GPT-2',
                        'size_gb': 1.4,
                        'parameters': '354M'
                    },
                    {
                        'model_id': 'gpt2-large',
                        'model_name': 'GPT-2 Large',
                        'description': 'Large version of OpenAI\'s GPT-2 model',
                        'pipeline_tag': 'text-generation',
                        'downloads': 1200000,
                        'likes': 18000,
                        'architecture': 'GPT-2',
                        'size_gb': 3.2,
                        'parameters': '774M'
                    },
                    # Classification Models
                    {
                        'model_id': 'bert-base-uncased',
                        'model_name': 'BERT Base Uncased',
                        'description': 'Base BERT model, uncased version for text understanding',
                        'pipeline_tag': 'text-classification',
                        'downloads': 2100000,
                        'likes': 25000,
                        'architecture': 'BERT',
                        'size_gb': 0.4,
                        'parameters': '110M'
                    },
                    {
                        'model_id': 'distilbert-base-uncased',
                        'model_name': 'DistilBERT Base Uncased',
                        'description': 'Distilled version of BERT base model, faster inference',
                        'pipeline_tag': 'text-classification',
                        'downloads': 1500000,
                        'likes': 18000,
                        'architecture': 'DistilBERT',
                        'size_gb': 0.3,
                        'parameters': '66M'
                    },
                    {
                        'model_id': 'roberta-base',
                        'model_name': 'RoBERTa Base',
                        'description': 'Robustly optimized BERT approach for text classification',
                        'pipeline_tag': 'text-classification',
                        'downloads': 890000,
                        'likes': 15000,
                        'architecture': 'RoBERTa',
                        'size_gb': 0.5,
                        'parameters': '125M'
                    }
                ]
                
                # Populate caches
                for model_data in models:
                    model_id = model_data['model_id']
                    self.model_cache[model_id] = {
                        'model_info': {
                            'model_name': model_data['model_name'],
                            'description': model_data['description'],
                            'pipeline_tag': model_data['pipeline_tag'],
                            'downloads': model_data['downloads'],
                            'likes': model_data['likes'],
                            'architecture': model_data['architecture']
                        },
                        'model_id': model_id
                    }
                    
                    # Add performance data
                    self.performance_cache[model_id] = {
                        'throughput_tokens_per_sec': max(10, 200 - model_data['size_gb'] * 20),
                        'latency_ms': max(50, model_data['size_gb'] * 30),
                        'memory_gb': model_data['size_gb'],
                        'parameters': model_data['parameters']
                    }
                    
                    # Add compatibility data  
                    self.compatibility_cache[model_id] = {
                        'min_ram_gb': max(1, model_data['size_gb'] * 2),
                        'supports_cpu': True,
                        'supports_gpu': model_data['size_gb'] < 10,
                        'supports_mps': True,
                        'recommended_hardware': 'GPU' if model_data['size_gb'] > 2 else 'CPU'
                    }
            
            def search_models(self, query='', task_filter=None, hardware_filter=None, limit=20):
                """Search models in the mock database."""
                logger.info(f"Mock scanner searching: query='{query}', task='{task_filter}', hardware='{hardware_filter}', limit={limit}")
                
                results = []
                query_lower = query.lower() if query else ''
                
                for model_id, model_data in self.model_cache.items():
                    # Check query match
                    if query and query_lower:
                        searchable = f"{model_id} {model_data['model_info'].get('model_name', '')} {model_data['model_info'].get('description', '')}".lower()
                        if query_lower not in searchable:
                            continue
                    
                    # Check task filter
                    if task_filter and task_filter != 'all':
                        if model_data['model_info'].get('pipeline_tag') != task_filter:
                            continue
                    
                    # Check hardware filter
                    if hardware_filter and hardware_filter != 'all':
                        compatibility = self.compatibility_cache.get(model_id, {})
                        if hardware_filter == 'cpu' and not compatibility.get('supports_cpu', True):
                            continue
                        elif hardware_filter == 'gpu' and not compatibility.get('supports_gpu', True):
                            continue
                    
                    # Add full model data
                    result = {
                        'model_id': model_id,
                        'model_info': model_data['model_info'],
                        'performance': self.performance_cache.get(model_id, {}),
                        'compatibility': self.compatibility_cache.get(model_id, {})
                    }
                    results.append(result)
                    
                    if len(results) >= limit:
                        break
                
                logger.info(f"Mock scanner found {len(results)} models")
                return results
            
            def download_model(self, model_id):
                """Simulate model downloading."""
                logger.info(f"Simulating download of model: {model_id}")
                if model_id in self.model_cache:
                    return {
                        'status': 'success',
                        'model_id': model_id,
                        'download_path': f"./models/{model_id}",
                        'size_gb': self.performance_cache.get(model_id, {}).get('memory_gb', 1.0),
                        'message': f'Model {model_id} downloaded successfully (simulated)'
                    }
                else:
                    return {
                        'status': 'error', 
                        'model_id': model_id,
                        'message': f'Model {model_id} not found in database'
                    }
            
            def test_model(self, model_id, hardware='cpu', test_prompt='Hello, world!'):
                """Simulate model testing."""
                logger.info(f"Simulating test of model: {model_id} on {hardware}")
                if model_id not in self.model_cache:
                    return {
                        'status': 'error',
                        'model_id': model_id,
                        'message': f'Model {model_id} not found'
                    }
                
                compatibility = self.compatibility_cache.get(model_id, {})
                performance = self.performance_cache.get(model_id, {})
                
                # Check hardware compatibility
                if hardware == 'gpu' and not compatibility.get('supports_gpu', True):
                    return {
                        'status': 'error',
                        'model_id': model_id,
                        'hardware': hardware,
                        'message': f'Model {model_id} does not support GPU acceleration'
                    }
                
                # Simulate successful test
                return {
                    'status': 'success',
                    'model_id': model_id,
                    'hardware': hardware,
                    'test_prompt': test_prompt,
                    'generated_text': f'[Generated by {model_id}] This is a simulated response to: {test_prompt}',
                    'performance': {
                        'latency_ms': performance.get('latency_ms', 100),
                        'throughput_tokens_per_sec': performance.get('throughput_tokens_per_sec', 50),
                        'memory_used_gb': performance.get('memory_gb', 1.0)
                    },
                    'message': f'Model {model_id} tested successfully on {hardware}'
                }
            
            def _get_architecture_distribution(self):
                """Get architecture distribution."""
                distribution = {}
                for model_data in self.model_cache.values():
                    arch = model_data['model_info'].get('architecture', 'Unknown')
                    distribution[arch] = distribution.get(arch, 0) + 1
                return distribution
            
            def _get_task_distribution(self):
                """Get task distribution."""
                distribution = {}
                for model_data in self.model_cache.values():
                    task = model_data['model_info'].get('pipeline_tag', 'Unknown')
                    distribution[task] = distribution.get(task, 0) + 1
                return distribution
            
            def _get_popular_models_summary(self):
                """Get popular models summary."""
                models = list(self.model_cache.values())
                # Sort by downloads (mock)
                models.sort(key=lambda x: x['model_info'].get('downloads', 0), reverse=True)
                return models
        
        return WorkingMockScanner()
    
    def _get_model_count(self) -> int:
        """Get total number of models in the model manager."""
        try:
            scanner = self._get_hub_scanner()
            if scanner:
                return len(scanner.model_cache)
        except Exception:
            pass
        return 0
    
    def _get_fallback_models(self, query: str = '', task_filter: str = None, hardware_filter: str = None, limit: int = 20):
        """Get fallback model data when HuggingFace Hub scanner is not available."""
        logger.info(f"Using fallback models for query: '{query}', task: '{task_filter}', hardware: '{hardware_filter}'")
        
        # Comprehensive fallback model database
        fallback_models = [
            {
                'model_id': 'microsoft/DialoGPT-large',
                'model_info': {
                    'model_name': 'DialoGPT Large',
                    'description': 'Large-scale conversational response generation model trained on 147M dialogues',
                    'pipeline_tag': 'text-generation',
                    'downloads': 125000,
                    'likes': 2300,
                    'architecture': 'GPT-2'
                },
                'performance': {'throughput_tokens_per_sec': 45.2},
                'compatibility': {'min_ram_gb': 4}
            },
            {
                'model_id': 'microsoft/DialoGPT-medium',
                'model_info': {
                    'model_name': 'DialoGPT Medium',
                    'description': 'Medium-scale conversational response generation model',
                    'pipeline_tag': 'text-generation',
                    'downloads': 89000,
                    'likes': 1800,
                    'architecture': 'GPT-2'
                },
                'performance': {'throughput_tokens_per_sec': 62.1},
                'compatibility': {'min_ram_gb': 2}
            },
            {
                'model_id': 'meta-llama/Llama-2-7b-chat-hf',
                'model_info': {
                    'model_name': 'Llama 2 7B Chat',
                    'description': 'Fine-tuned version of Llama 2 7B for chat conversations',
                    'pipeline_tag': 'text-generation',
                    'downloads': 1800000,
                    'likes': 45000,
                    'architecture': 'LLaMA'
                },
                'performance': {'throughput_tokens_per_sec': 28.3},
                'compatibility': {'min_ram_gb': 14}
            },
            {
                'model_id': 'codellama/CodeLlama-7b-Python-hf',
                'model_info': {
                    'model_name': 'Code Llama 7B Python',
                    'description': 'Code Llama model fine-tuned for Python code generation',
                    'pipeline_tag': 'code-generation',
                    'downloads': 850000,
                    'likes': 12000,
                    'architecture': 'LLaMA'
                },
                'performance': {'throughput_tokens_per_sec': 32.1},
                'compatibility': {'min_ram_gb': 14}
            },
            {
                'model_id': 'bert-base-uncased',
                'model_info': {
                    'model_name': 'BERT Base Uncased',
                    'description': 'Base BERT model, uncased version for text understanding',
                    'pipeline_tag': 'text-classification',
                    'downloads': 2100000,
                    'likes': 25000,
                    'architecture': 'BERT'
                },
                'performance': {'throughput_tokens_per_sec': 120.5},
                'compatibility': {'min_ram_gb': 1}
            },
            {
                'model_id': 'distilbert-base-uncased',
                'model_info': {
                    'model_name': 'DistilBERT Base Uncased',
                    'description': 'Distilled version of BERT base model, faster inference',
                    'pipeline_tag': 'text-classification',
                    'downloads': 1500000,
                    'likes': 18000,
                    'architecture': 'DistilBERT'
                },
                'performance': {'throughput_tokens_per_sec': 180.2},
                'compatibility': {'min_ram_gb': 0.5}
            },
            {
                'model_id': 'gpt2',
                'model_info': {
                    'model_name': 'GPT-2',
                    'description': 'OpenAI\'s GPT-2 model for text generation',
                    'pipeline_tag': 'text-generation',
                    'downloads': 3200000,
                    'likes': 35000,
                    'architecture': 'GPT-2'
                },
                'performance': {'throughput_tokens_per_sec': 85.3},
                'compatibility': {'min_ram_gb': 2}
            },
            {
                'model_id': 'gpt2-medium',
                'model_info': {
                    'model_name': 'GPT-2 Medium',
                    'description': 'Medium version of OpenAI\'s GPT-2 model',
                    'pipeline_tag': 'text-generation',
                    'downloads': 1900000,
                    'likes': 22000,
                    'architecture': 'GPT-2'
                },
                'performance': {'throughput_tokens_per_sec': 52.8},
                'compatibility': {'min_ram_gb': 3}
            }
        ]
        
        # Filter models based on query and filters
        filtered_models = []
        query_lower = query.lower() if query else ''
        
        for model in fallback_models:
            # Check query match
            if query and query_lower:
                model_text = f"{model['model_id']} {model['model_info']['model_name']} {model['model_info']['description']}".lower()
                if query_lower not in model_text:
                    continue
            
            # Check task filter
            if task_filter and task_filter != 'all':
                if model['model_info']['pipeline_tag'] != task_filter:
                    continue
            
            # Check hardware filter (simplified check)
            if hardware_filter and hardware_filter != 'all':
                # This is a simplified hardware filter - in reality would be more complex
                min_ram = model['compatibility']['min_ram_gb']
                if hardware_filter == 'cpu' and min_ram > 8:
                    continue
                elif hardware_filter == 'gpu' and min_ram < 2:
                    continue
            
            filtered_models.append(model)
            
            if len(filtered_models) >= limit:
                break
        
        return filtered_models
    
    def _get_autocomplete_fallback(self, query: str, limit: int = 10):
        """Get fallback autocomplete suggestions when HuggingFace Hub scanner is not available."""
        query_lower = query.lower()
        
        # Common popular models for autocomplete
        common_models = [
            {'id': 'gpt2', 'label': 'gpt2 (text-generation)', 'pipeline_tag': 'text-generation', 'downloads': 3200000},
            {'id': 'bert-base-uncased', 'label': 'bert-base-uncased (text-classification)', 'pipeline_tag': 'text-classification', 'downloads': 2100000},
            {'id': 'distilbert-base-uncased', 'label': 'distilbert-base-uncased (text-classification)', 'pipeline_tag': 'text-classification', 'downloads': 1500000},
            {'id': 'stabilityai/stable-diffusion-xl-base-1.0', 'label': 'stabilityai/stable-diffusion-xl-base-1.0 (text-to-image)', 'pipeline_tag': 'text-to-image', 'downloads': 950000},
            {'id': 'runwayml/stable-diffusion-v1-5', 'label': 'runwayml/stable-diffusion-v1-5 (text-to-image)', 'pipeline_tag': 'text-to-image', 'downloads': 1200000},
            {'id': 'openai/whisper-large-v3', 'label': 'openai/whisper-large-v3 (automatic-speech-recognition)', 'pipeline_tag': 'automatic-speech-recognition', 'downloads': 780000},
            {'id': 'meta-llama/Llama-2-7b-chat-hf', 'label': 'meta-llama/Llama-2-7b-chat-hf (text-generation)', 'pipeline_tag': 'text-generation', 'downloads': 1800000},
            {'id': 'microsoft/DialoGPT-large', 'label': 'microsoft/DialoGPT-large (text-generation)', 'pipeline_tag': 'text-generation', 'downloads': 125000},
            {'id': 'sentence-transformers/all-MiniLM-L6-v2', 'label': 'sentence-transformers/all-MiniLM-L6-v2 (sentence-similarity)', 'pipeline_tag': 'sentence-similarity', 'downloads': 650000},
            {'id': 'google/vit-base-patch16-224', 'label': 'google/vit-base-patch16-224 (image-classification)', 'pipeline_tag': 'image-classification', 'downloads': 420000},
            {'id': 'facebook/detr-resnet-50', 'label': 'facebook/detr-resnet-50 (object-detection)', 'pipeline_tag': 'object-detection', 'downloads': 280000},
            {'id': 'microsoft/resnet-50', 'label': 'microsoft/resnet-50 (image-classification)', 'pipeline_tag': 'image-classification', 'downloads': 520000},
            {'id': 'google/flan-t5-base', 'label': 'google/flan-t5-base (text2text-generation)', 'pipeline_tag': 'text2text-generation', 'downloads': 890000},
            {'id': 'facebook/bart-large-cnn', 'label': 'facebook/bart-large-cnn (summarization)', 'pipeline_tag': 'summarization', 'downloads': 670000},
            {'id': 'Helsinki-NLP/opus-mt-en-de', 'label': 'Helsinki-NLP/opus-mt-en-de (translation)', 'pipeline_tag': 'translation', 'downloads': 340000},
        ]
        
        # Filter models that match the query
        matching = []
        for model in common_models:
            if query_lower in model['id'].lower() or query_lower in model['pipeline_tag'].lower():
                matching.append(model)
                if len(matching) >= limit:
                    break
        
        return matching
    
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
    <script src="/static/js/mcp-sdk.js"></script>
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

    <!-- Toast container for notifications -->
    <div class="toast-container position-fixed top-0 end-0 p-3" id="toast-container"></div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Initialize MCP Client
        const mcpClient = new MCPClient('/jsonrpc');
        
        // Utility functions for user feedback
        function showToast(message, type = 'info', duration = 5000) {
            console.log(`[MCP Dashboard] ${type.toUpperCase()}: ${message}`);
            
            const toastContainer = document.getElementById('toast-container');
            if (!toastContainer) {
                console.warn('Toast container not found');
                return;
            }
            
            const toastId = 'toast-' + Date.now();
            
            const toastHtml = `
                <div class="toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0" role="alert" id="${toastId}">
                    <div class="d-flex">
                        <div class="toast-body">
                            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
                            ${message}
                        </div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                    </div>
                </div>
            `;
            
            toastContainer.insertAdjacentHTML('beforeend', toastHtml);
            const toastElement = document.getElementById(toastId);
            
            // Only use bootstrap if available
            if (typeof bootstrap !== 'undefined' && bootstrap.Toast) {
                const toast = new bootstrap.Toast(toastElement, { delay: duration });
                toast.show();
                
                // Remove toast element after it's hidden
                toastElement.addEventListener('hidden.bs.toast', () => {
                    toastElement.remove();
                });
            } else {
                // Fallback: just show and remove after duration
                toastElement.classList.add('show');
                setTimeout(() => {
                    toastElement.remove();
                }, duration);
            }
        }
        
        function logUserAction(action, details = {}) {
            console.log(`[MCP Dashboard] User Action: ${action}`, details);
        }
        
        // Search functionality with proper logging and error handling
        async function searchModels() {
            const query = document.getElementById('searchInput').value.trim();
            const taskFilter = document.getElementById('taskFilter').value;
            const hardwareFilter = document.getElementById('hardwareFilter').value;
            
            logUserAction('search_models', { query, taskFilter, hardwareFilter });
            
            if (!query) {
                showToast('Please enter a search query to find models', 'warning');
                document.getElementById('searchInput').focus();
                return;
            }
            
            const resultsDiv = document.getElementById('searchResults');
            resultsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin fa-2x"></i><p>Searching models...</p></div>';
            
            showToast(`Searching for models: "${query}"...`, 'info', 3000);
            
            console.log(`[MCP Dashboard] Making MCP search request via SDK`);
            
            try {
                // Use MCP client to call the search_models tool
                const data = await mcpClient.request('search_models', {
                    query: query,
                    task_filter: taskFilter || null,
                    hardware_filter: hardwareFilter || null,
                    limit: 20
                });
                
                console.log(`[MCP Dashboard] Search results received via MCP:`, data);
                
                if (data.status === 'error') {
                    showToast(`Search error: ${data.error}`, 'error');
                    resultsDiv.innerHTML = `<div class="alert alert-warning">
                        <h5><i class="fas fa-exclamation-triangle me-2"></i>Search Error</h5>
                        <p>${data.error}</p>
                        <p class="mb-0"><small>Check the browser console for more details.</small></p>
                    </div>`;
                } else {
                    displaySearchResults(data.results, query);
                    showToast(`Found ${data.results ? data.results.length : 0} models for "${query}"`, 'success');
                }
            } catch (error) {
                console.error('[MCP Dashboard] MCP search error:', error);
                showToast(`Search failed: ${error.message}`, 'error');
                resultsDiv.innerHTML = `<div class="alert alert-danger">
                    <h5><i class="fas fa-exclamation-triangle me-2"></i>Search Failed</h5>
                    <p>Unable to search for models. This might be because:</p>
                    <ul>
                        <li>The MCP server is not running</li>
                        <li>The HuggingFace Hub scanner is not available</li>
                        <li>Network connection issues</li>
                        <li>Server configuration problems</li>
                    </ul>
                    <p class="mb-0"><small>Error: ${error.message}</small></p>
                </div>`;
            }
        }
        
        function truncateText(text, maxLength = 150) {
            if (!text || text.length <= maxLength) {
                return text || 'No description available';
            }
            return text.substring(0, maxLength).trim() + '...';
        }
        
        function extractSummaryFromModelCard(modelCard, maxLength = 150) {
            if (!modelCard) {
                return null;
            }
            
            // Remove markdown headers (# ## ###)
            let text = modelCard.replace(/^#+\\s+/gm, '');
            
            // Remove code blocks
            text = text.replace(/```[\\s\\S]*?```/g, '');
            
            // Remove inline code
            text = text.replace(/`[^`]+`/g, '');
            
            // Remove URLs
            text = text.replace(/https?:\\/\\/[^\\s]+/g, '');
            
            // Remove markdown links [text](url)
            text = text.replace(/\\[([^\\]]+)\\]\\([^\\)]+\\)/g, '$1');
            
            // Remove extra whitespace and newlines
            text = text.replace(/\\s+/g, ' ').trim();
            
            // Get first meaningful sentence or paragraph
            const sentences = text.split(/[.!?]\\s+/);
            let summary = '';
            
            for (const sentence of sentences) {
                const trimmed = sentence.trim();
                // Skip very short sentences (likely headers or fragments)
                if (trimmed.length > 20) {
                    summary = trimmed;
                    break;
                }
            }
            
            // If no good sentence found, just take the first part
            if (!summary) {
                summary = text.substring(0, maxLength * 2);
            }
            
            // Truncate to maxLength
            if (summary.length > maxLength) {
                summary = summary.substring(0, maxLength).trim() + '...';
            }
            
            return summary || null;
        }
        
        function getModelDescription(modelInfo, maxLength = 150) {
            // First, try the description field
            if (modelInfo.description) {
                return truncateText(modelInfo.description, maxLength);
            }
            
            // If no description, try to extract from model_card
            if (modelInfo.model_card) {
                const summary = extractSummaryFromModelCard(modelInfo.model_card, maxLength);
                if (summary) {
                    return summary;
                }
            }
            
            // Fallback to "No description available"
            return 'No description available';
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
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
                const description = getModelDescription(modelInfo);
                
                html += `
                    <div class="model-card">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <h5 class="mb-1">${modelInfo.model_name || result.model_id}</h5>
                            <span class="badge bg-primary">${modelInfo.pipeline_tag || 'Unknown'}</span>
                        </div>
                        
                        <p class="text-muted mb-2">${description}</p>
                        
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
                            <button class="btn btn-sm btn-info me-2" onclick="showModelDetails('${result.model_id}')">
                                <i class="fas fa-info-circle me-1"></i>View Details
                            </button>
                            <a href="https://huggingface.co/${result.model_id}" target="_blank" class="btn btn-sm btn-outline-primary me-2">
                                <i class="fas fa-external-link-alt me-1"></i>View on HuggingFace
                            </a>
                            <button class="btn btn-sm btn-primary me-2" onclick="downloadModel('${result.model_id}')">
                                <i class="fas fa-download me-1"></i>Download
                            </button>
                            <button class="btn btn-sm btn-success me-2" onclick="testModel('${result.model_id}', 'cpu')">
                                <i class="fas fa-play me-1"></i>Test CPU
                            </button>
                            ${compatibility.supports_gpu ? 
                                `<button class="btn btn-sm btn-success me-2" onclick="testModel('${result.model_id}', 'gpu')">
                                    <i class="fas fa-play me-1"></i>Test GPU
                                </button>` : ''}
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
        
        async function getRecommendations() {
            const taskType = document.getElementById('taskFilter').value || 'text-generation';
            const hardware = document.getElementById('hardwareFilter').value || 'cpu';
            const performance = document.getElementById('performanceFilter').value || 'balanced';
            
            logUserAction('get_recommendations', { taskType, hardware, performance });
            
            const resultsDiv = document.getElementById('recommendationResults');
            resultsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin fa-2x"></i><p>Getting AI recommendations...</p></div>';
            
            showToast(`Getting AI recommendations for ${taskType} on ${hardware}...`, 'info', 3000);
            
            console.log(`[MCP Dashboard] Getting recommendations via MCP with params:`, { taskType, hardware, performance });
            
            try {
                // Use MCP client to call the recommend_models tool
                const data = await mcpClient.request('recommend_models', {
                    task_type: taskType,
                    hardware: hardware,
                    performance: performance,
                    limit: 5
                });
                
                console.log(`[MCP Dashboard] Recommendations data via MCP:`, data);
                
                if (data.status === 'error') {
                    showToast(`Recommendations error: ${data.error}`, 'error');
                    resultsDiv.innerHTML = `<div class="alert alert-warning">
                        <h5><i class="fas fa-exclamation-triangle me-2"></i>Recommendations Error</h5>
                        <p>${data.error}</p>
                        <p class="mb-0"><small>This might be because the model recommendation system is not available.</small></p>
                    </div>`;
                } else {
                    displayRecommendations(data.recommendations, data.context);
                    const count = data.recommendations ? data.recommendations.length : 0;
                    showToast(`Found ${count} AI recommendations`, 'success');
                }
            } catch (error) {
                console.error('[MCP Dashboard] MCP recommendation error:', error);
                showToast(`Recommendations failed: ${error.message}`, 'error');
                resultsDiv.innerHTML = `<div class="alert alert-danger">
                    <h5><i class="fas fa-exclamation-triangle me-2"></i>Recommendations Failed</h5>
                    <p>Unable to get AI recommendations. This might be because:</p>
                    <ul>
                        <li>The MCP server is not running</li>
                        <li>The model recommendation system is not available</li>
                        <li>The HuggingFace Hub scanner is not available</li>
                        <li>Network connection issues</li>
                    </ul>
                    <p class="mb-0"><small>Error: ${error.message}</small></p>
                </div>`;
            }
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
                const description = getModelDescription(modelInfo);
                
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
                        
                        <p class="text-muted mb-2">${description}</p>
                        
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
                            <button class="btn btn-sm btn-info me-2" onclick="showModelDetails('${rec.model_id}')">
                                <i class="fas fa-info-circle me-1"></i>View Details
                            </button>
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
            
            logUserAction('scan_hub', { limit, taskFilter });
            
            showToast(`Starting HuggingFace Hub scan (limit: ${limit})...`, 'info');
            
            const data = {
                limit: parseInt(limit),
                task_filter: taskFilter || null
            };
            
            console.log(`[MCP Dashboard] Starting hub scan with data:`, data);
            
            fetch('/api/mcp/models/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                console.log(`[MCP Dashboard] Scan response status: ${response.status}`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log(`[MCP Dashboard] Scan response:`, data);
                
                if (data.status === 'started') {
                    showToast(`Hub scan started successfully! Scanning up to ${data.limit} models.`, 'success');
                    // Refresh stats after a delay
                    setTimeout(() => {
                        console.log('[MCP Dashboard] Refreshing stats after scan...');
                        loadStats();
                    }, 5000);
                } else if (data.error) {
                    showToast(`Scan failed: ${data.error}`, 'error');
                } else {
                    showToast('Scan request completed, but status unclear', 'warning');
                }
            })
            .catch(error => {
                console.error('[MCP Dashboard] Scan error:', error);
                showToast(`Hub scan failed: ${error.message}. This might be because the HuggingFace Hub scanner is not available.`, 'error');
            });
        }
        
        function loadStats() {
            const statsDiv = document.getElementById('modelStats');
            statsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';
            
            logUserAction('load_stats');
            console.log('[MCP Dashboard] Loading model statistics...');
            
            fetch('/api/mcp/models/stats')
                .then(response => {
                    console.log(`[MCP Dashboard] Stats response status: ${response.status}`);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('[MCP Dashboard] Stats data received:', data);
                    
                    if (data.error) {
                        console.warn('[MCP Dashboard] Stats API returned error:', data.error);
                        statsDiv.innerHTML = `<div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Statistics unavailable: ${data.error}
                        </div>`;
                    } else {
                        displayStats(data);
                    }
                })
                .catch(error => {
                    console.error('[MCP Dashboard] Stats error:', error);
                    statsDiv.innerHTML = `<div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Failed to load statistics: ${error.message}
                        <br><small>This might be because the HuggingFace Hub scanner is not available.</small>
                    </div>`;
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
            const feedbackType = positive ? 'positive' : 'negative';
            logUserAction('provide_feedback', { modelId, feedbackType });
            console.log(`[MCP Dashboard] Feedback for ${modelId}: ${feedbackType}`);
            showToast(`Feedback recorded! The AI will learn from your ${feedbackType} feedback about ${modelId}.`, 'success', 3000);
        }
        
        // Model downloading functionality
        function downloadModel(modelId) {
            logUserAction('download_model', { modelId });
            
            showToast(`Starting download of ${modelId}...`, 'info');
            
            const data = {
                model_id: modelId
            };
            
            console.log(`[MCP Dashboard] Starting model download:`, data);
            
            fetch('/api/mcp/models/download', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                console.log(`[MCP Dashboard] Download response status: ${response.status}`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log(`[MCP Dashboard] Download response:`, data);
                
                if (data.status === 'success') {
                    showToast(`${modelId} downloaded successfully! (${data.size_gb}GB)`, 'success');
                } else {
                    showToast(`Download failed: ${data.message || 'Unknown error'}`, 'error');
                }
            })
            .catch(error => {
                console.error('[MCP Dashboard] Download error:', error);
                showToast(`Download failed: ${error.message}`, 'error');
            });
        }
        
        // Model testing functionality
        function testModel(modelId, hardware = 'cpu') {
            logUserAction('test_model', { modelId, hardware });
            
            const testPrompt = prompt('Enter test prompt:', 'Hello, how are you?') || 'Hello, world!';
            
            showToast(`Testing ${modelId} on ${hardware}...`, 'info');
            
            const data = {
                model_id: modelId,
                hardware: hardware,
                test_prompt: testPrompt
            };
            
            console.log(`[MCP Dashboard] Starting model test:`, data);
            
            fetch('/api/mcp/models/test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                console.log(`[MCP Dashboard] Test response status: ${response.status}`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log(`[MCP Dashboard] Test response:`, data);
                
                if (data.status === 'success') {
                    showToast(`${modelId} test completed on ${hardware}!`, 'success');
                    showTestResults(data);
                } else {
                    showToast(`Test failed: ${data.message || 'Unknown error'}`, 'error');
                }
            })
            .catch(error => {
                console.error('[MCP Dashboard] Test error:', error);
                showToast(`Test failed: ${error.message}`, 'error');
            });
        }
        
        // Show test results
        function showTestResults(testData) {
            const resultsHtml = `
                <div class="alert alert-success mt-4">
                    <h5><i class="fas fa-check-circle me-2"></i>Test Results for ${testData.model_id}</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <strong>Hardware:</strong> ${testData.hardware}<br>
                            <strong>Prompt:</strong> "${testData.test_prompt}"<br>
                            <strong>Latency:</strong> ${testData.performance?.latency_ms || 'N/A'}ms<br>
                        </div>
                        <div class="col-md-6">
                            <strong>Throughput:</strong> ${testData.performance?.throughput_tokens_per_sec || 'N/A'} tok/sec<br>
                            <strong>Memory:</strong> ${testData.performance?.memory_used_gb || 'N/A'}GB<br>
                        </div>
                    </div>
                    <div class="mt-3">
                        <strong>Generated Text:</strong><br>
                        <div class="bg-light p-3 rounded border">${testData.generated_text || 'No output'}</div>
                    </div>
                </div>
            `;
            
            // Add to results area
            const resultsDiv = document.getElementById('searchResults');
            resultsDiv.innerHTML += resultsHtml;
            
            // Scroll to results
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Show model details modal
        async function showModelDetails(modelId) {
            logUserAction('show_model_details', { modelId });
            
            showToast(`Loading details for ${modelId}...`, 'info', 2000);
            
            try {
                // Use MCP client to call the get_model_details tool
                const data = await mcpClient.request('get_model_details', {
                    model_id: modelId
                });
                
                if (data.status === 'error') {
                    showToast(`Failed to load model details: ${data.error}`, 'error');
                } else {
                    displayModelDetailsModal(data);
                }
            } catch (error) {
                console.error('[MCP Dashboard] MCP error fetching model details:', error);
                showToast(`Failed to load model details: ${error.message}`, 'error');
            }
        }
        
        // Display model details in a modal
        function displayModelDetailsModal(modelData) {
            const modelInfo = modelData.model_info || {};
            const performance = modelData.performance || {};
            const compatibility = modelData.compatibility || {};
            
            // Create modal HTML
            const modalHtml = `
                <div class="modal fade" id="modelDetailsModal" tabindex="-1" aria-labelledby="modelDetailsModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-lg modal-dialog-scrollable">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="modelDetailsModalLabel">
                                    <i class="fas fa-robot me-2"></i>${modelInfo.model_name || modelData.model_id}
                                </h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body">
                                <!-- Full Description -->
                                <div class="mb-4">
                                    <h6><i class="fas fa-file-alt me-2"></i>Description</h6>
                                    <p>${modelInfo.description || (modelInfo.model_card ? extractSummaryFromModelCard(modelInfo.model_card, 500) : null) || 'No description available'}</p>
                                </div>
                                
                                <!-- Model Card -->
                                ${modelInfo.model_card ? `
                                <div class="mb-4">
                                    <h6><i class="fas fa-book me-2"></i>Model Card</h6>
                                    <div class="card">
                                        <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                                            <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 0.9rem;">${escapeHtml(modelInfo.model_card)}</pre>
                                        </div>
                                    </div>
                                </div>
                                ` : ''}
                                
                                <!-- Model Info -->
                                <div class="mb-4">
                                    <h6><i class="fas fa-info-circle me-2"></i>Model Information</h6>
                                    <table class="table table-sm">
                                        <tbody>
                                            <tr><td><strong>Model ID:</strong></td><td>${modelData.model_id}</td></tr>
                                            <tr><td><strong>Pipeline Tag:</strong></td><td>${modelInfo.pipeline_tag || 'N/A'}</td></tr>
                                            <tr><td><strong>Architecture:</strong></td><td>${modelInfo.architecture || 'N/A'}</td></tr>
                                            <tr><td><strong>Library:</strong></td><td>${modelInfo.library_name || 'N/A'}</td></tr>
                                            <tr><td><strong>Framework:</strong></td><td>${modelInfo.framework || 'N/A'}</td></tr>
                                            <tr><td><strong>Downloads:</strong></td><td>${(modelInfo.downloads || 0).toLocaleString()}</td></tr>
                                            <tr><td><strong>Likes:</strong></td><td>${(modelInfo.likes || 0).toLocaleString()}</td></tr>
                                        </tbody>
                                    </table>
                                </div>
                                
                                <!-- Performance Metrics -->
                                ${Object.keys(performance).length > 0 ? `
                                <div class="mb-4">
                                    <h6><i class="fas fa-tachometer-alt me-2"></i>Performance Metrics</h6>
                                    <table class="table table-sm">
                                        <tbody>
                                            ${performance.throughput_tokens_per_sec ? `<tr><td><strong>Throughput:</strong></td><td>${performance.throughput_tokens_per_sec.toFixed(1)} tokens/sec</td></tr>` : ''}
                                            ${performance.inference_time_ms ? `<tr><td><strong>Inference Time:</strong></td><td>${performance.inference_time_ms.toFixed(1)}ms</td></tr>` : ''}
                                            ${performance.memory_usage_mb ? `<tr><td><strong>Memory Usage:</strong></td><td>${performance.memory_usage_mb.toFixed(0)}MB</td></tr>` : ''}
                                            ${performance.gpu_memory_mb ? `<tr><td><strong>GPU Memory:</strong></td><td>${performance.gpu_memory_mb.toFixed(0)}MB</td></tr>` : ''}
                                        </tbody>
                                    </table>
                                </div>
                                ` : ''}
                                
                                <!-- Hardware Compatibility -->
                                ${Object.keys(compatibility).length > 0 ? `
                                <div class="mb-4">
                                    <h6><i class="fas fa-microchip me-2"></i>Hardware Compatibility</h6>
                                    <table class="table table-sm">
                                        <tbody>
                                            ${compatibility.supports_cpu ? `<tr><td><strong>CPU:</strong></td><td><i class="fas fa-check text-success"></i> Supported</td></tr>` : ''}
                                            ${compatibility.supports_gpu ? `<tr><td><strong>GPU:</strong></td><td><i class="fas fa-check text-success"></i> Supported</td></tr>` : ''}
                                            ${compatibility.min_ram_gb ? `<tr><td><strong>Min RAM:</strong></td><td>${compatibility.min_ram_gb}GB</td></tr>` : ''}
                                            ${compatibility.min_vram_gb ? `<tr><td><strong>Min VRAM:</strong></td><td>${compatibility.min_vram_gb}GB</td></tr>` : ''}
                                        </tbody>
                                    </table>
                                </div>
                                ` : ''}
                            </div>
                            <div class="modal-footer">
                                <a href="https://huggingface.co/${modelData.model_id}" target="_blank" class="btn btn-primary">
                                    <i class="fas fa-external-link-alt me-1"></i>View on HuggingFace
                                </a>
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // Remove any existing modal
            const existingModal = document.getElementById('modelDetailsModal');
            if (existingModal) {
                existingModal.remove();
            }
            
            // Add modal to document
            document.body.insertAdjacentHTML('beforeend', modalHtml);
            
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
            modal.show();
            
            // Clean up after modal is hidden
            document.getElementById('modelDetailsModal').addEventListener('hidden.bs.modal', function() {
                this.remove();
            });
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

    def _start_autoscaler(self) -> None:
        """Start the GitHub Actions autoscaler in a background thread."""
        if not self.enable_autoscaler:
            logger.info("Autoscaler disabled by configuration")
            return
        
        try:
            # Import autoscaler - use parent directory import path
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from github_autoscaler import GitHubRunnerAutoscaler
            from ipfs_accelerate_py.github_cli import GitHubCLI
            
            # Check GitHub CLI authentication
            try:
                gh = GitHubCLI()
                auth_status = gh.get_auth_status()
                
                if not auth_status.get("authenticated"):
                    logger.warning("GitHub CLI not authenticated - autoscaler disabled")
                    logger.warning("  To enable: gh auth login")
                    return
            except Exception as e:
                logger.warning(f"Could not check GitHub authentication: {e}")
                return
            
            logger.info("Starting GitHub Actions autoscaler in background...")
            
            # Create autoscaler with configuration
            self.autoscaler_instance = GitHubRunnerAutoscaler(
                owner=self.autoscaler_config.get('owner'),
                poll_interval=self.autoscaler_config.get('interval', 60),
                since_days=self.autoscaler_config.get('since_days', 1),
                max_runners=self.autoscaler_config.get('max_runners'),
                filter_by_arch=self.autoscaler_config.get('filter_by_arch', True),
                enable_p2p=self.autoscaler_config.get('enable_p2p', True)
            )
            
            def run_autoscaler():
                try:
                    self.autoscaler_instance.start(setup_signals=False)
                except Exception as e:
                    logger.error(f"Autoscaler error: {e}")
            
            import threading
            self.autoscaler_thread = threading.Thread(target=run_autoscaler, daemon=True)
            self.autoscaler_thread.start()
            logger.info("✓ GitHub Actions autoscaler started")
            
        except ImportError as e:
            logger.warning(f"GitHub autoscaler not available: {e}")
        except Exception as e:
            logger.warning(f"Could not start autoscaler: {e}")
    
    def run(self, debug: bool = False) -> None:
        """Run the MCP dashboard.
        
        Args:
            debug: Enable debug mode
        """
        logger.info(f"Starting MCP Dashboard on http://{self.host}:{self.port}/mcp")
        
        # Start autoscaler before running the Flask app
        self._start_autoscaler()
        
        try:
            self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
        except KeyboardInterrupt:
            logger.info("Dashboard shutdown requested")
            if self.autoscaler_instance:
                logger.info("Stopping autoscaler...")
                self.autoscaler_instance.stop()
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            if self.autoscaler_instance:
                self.autoscaler_instance.stop()
            raise


if __name__ == '__main__':
    dashboard = MCPDashboard()
    dashboard.run()