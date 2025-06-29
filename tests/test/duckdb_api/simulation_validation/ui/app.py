#!/usr/bin/env python3
"""
Simulation Validation Framework - Web Application

This module implements a Flask web application that provides a user interface for the
Simulation Accuracy and Validation Framework.
"""

import os
import sys
import json
import logging
import datetime
import tempfile
import functools
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_validation_web")

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file, g, abort
    from flask_wtf import FlaskForm
    from flask_wtf.csrf import CSRFProtect
    from wtforms import StringField, SelectField, IntegerField, FloatField, BooleanField, SubmitField, TextAreaField, PasswordField
    from wtforms.validators import DataRequired, Optional, NumberRange, Email, Length, EqualTo
    from werkzeug.security import generate_password_hash, check_password_hash
    import apscheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    
    FLASK_AVAILABLE = True
except ImportError:
    logger.error("Flask or related packages not available. Install required packages with: pip install flask flask-wtf apscheduler")
    FLASK_AVAILABLE = False

# Import the Simulation Validation Framework
try:
    from duckdb_api.simulation_validation.simulation_validation_framework import SimulationValidationFramework
    from duckdb_api.simulation_validation.core.base import SimulationResult, HardwareResult
    from duckdb_api.simulation_validation.db_integration import SimulationDBIntegration
    
    FRAMEWORK_AVAILABLE = True
except ImportError:
    logger.error("Simulation Validation Framework not available or could not be imported")
    FRAMEWORK_AVAILABLE = False

# User role definitions
class UserRole:
    VIEWER = "viewer"  # Can only view results
    ANALYST = "analyst"  # Can run validations and analysis
    ADMIN = "admin"  # Full access including user management
    
    @staticmethod
    def get_all_roles():
        return [UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN]

# Define user authentication forms
class LoginForm(FlaskForm):
    """Form for user login."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    """Form for user registration."""
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', 
                                     validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[(r, r.title()) for r in UserRole.get_all_roles()])
    submit = SubmitField('Register')

class ResetPasswordForm(FlaskForm):
    """Form for resetting password."""
    old_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password', 
                                    validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Reset Password')

# Function to require login
def login_required(f):
    """Decorator to require login for a route."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Function to require specific role
def role_required(role):
    """Decorator to require specific role for a route."""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('login', next=request.url))
            
            if 'user_role' not in session:
                flash('Invalid user session. Please log in again.', 'danger')
                return redirect(url_for('logout'))
            
            user_role = session['user_role']
            
            # Admin can access everything
            if user_role == UserRole.ADMIN:
                return f(*args, **kwargs)
            
            # Analysts can access analyst and viewer content
            if user_role == UserRole.ANALYST and role == UserRole.VIEWER:
                return f(*args, **kwargs)
            
            # Must match exactly for other roles
            if user_role != role:
                flash(f'You need {role} permissions to access this page.', 'danger')
                return redirect(url_for('index'))
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Notification Manager
class NotificationManager:
    """Manager for user notifications."""
    
    NOTIFICATION_TYPES = {
        'info': 'Information',
        'success': 'Success',
        'warning': 'Warning',
        'danger': 'Alert'
    }
    
    def __init__(self, max_notifications: int = 50, db_integration=None):
        """Initialize notification manager."""
        self.max_notifications = max_notifications
        self.db_integration = db_integration
    
    def add_notification(self, user_id: str, message: str, 
                         notification_type: str = 'info', 
                         related_entity: Optional[str] = None,
                         entity_id: Optional[str] = None,
                         store_in_db: bool = True) -> Dict[str, Any]:
        """
        Add a notification for a user.
        
        Args:
            user_id: ID of the user
            message: Notification message
            notification_type: Type of notification (info, success, warning, danger)
            related_entity: Type of entity the notification is related to
            entity_id: ID of the related entity
            store_in_db: Whether to store the notification in the database
            
        Returns:
            The notification object
        """
        if notification_type not in self.NOTIFICATION_TYPES:
            notification_type = 'info'
        
        notification = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'message': message,
            'type': notification_type,
            'related_entity': related_entity,
            'entity_id': entity_id,
            'created_at': datetime.datetime.now().isoformat(),
            'read': False
        }
        
        # Store in session
        if 'notifications' not in session:
            session['notifications'] = []
        
        # Add to beginning of list
        session['notifications'].insert(0, notification)
        
        # Trim to max size
        if len(session['notifications']) > self.max_notifications:
            session['notifications'] = session['notifications'][:self.max_notifications]
        
        # Store notification in database if available
        if store_in_db and self.db_integration:
            try:
                self.db_integration.store_notification(notification)
            except Exception as e:
                logger.error(f"Error storing notification in database: {e}")
        
        # Make sure session is saved
        session.modified = True
        
        return notification
    
    def get_notifications(self, user_id: str, limit: int = 10, 
                         include_read: bool = False, 
                         notification_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get notifications for a user.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of notifications to return
            include_read: Whether to include read notifications
            notification_type: Type of notification to filter by
            
        Returns:
            List of notification objects
        """
        if 'notifications' not in session:
            session['notifications'] = []
        
        notifications = session['notifications']
        
        # Filter notifications
        filtered = [n for n in notifications 
                   if n['user_id'] == user_id and 
                   (include_read or not n['read']) and
                   (notification_type is None or n['type'] == notification_type)]
        
        # Return limited number
        return filtered[:limit]
    
    def mark_as_read(self, notification_id: str, user_id: str) -> bool:
        """
        Mark a notification as read.
        
        Args:
            notification_id: ID of the notification
            user_id: ID of the user
            
        Returns:
            Whether the notification was found and marked as read
        """
        if 'notifications' not in session:
            return False
        
        for i, notification in enumerate(session['notifications']):
            if notification['id'] == notification_id and notification['user_id'] == user_id:
                session['notifications'][i]['read'] = True
                session.modified = True
                
                # Update in database if available
                if self.db_integration:
                    try:
                        self.db_integration.update_notification(notification_id, {'read': True})
                    except Exception as e:
                        logger.error(f"Error updating notification in database: {e}")
                
                return True
        
        return False
    
    def clear_notifications(self, user_id: str) -> int:
        """
        Clear all notifications for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Number of notifications cleared
        """
        if 'notifications' not in session:
            return 0
        
        original_count = len(session['notifications'])
        session['notifications'] = [n for n in session['notifications'] if n['user_id'] != user_id]
        session.modified = True
        
        cleared_count = original_count - len(session['notifications'])
        
        # Update in database if available
        if cleared_count > 0 and self.db_integration:
            try:
                self.db_integration.clear_notifications(user_id)
            except Exception as e:
                logger.error(f"Error clearing notifications in database: {e}")
        
        return cleared_count

# Job Scheduler
class JobScheduler:
    """Scheduler for long-running operations."""
    
    JOB_TYPES = {
        'validation': 'Run Validation',
        'calibration': 'Run Calibration',
        'drift_detection': 'Run Drift Detection',
        'parameter_discovery': 'Run Parameter Discovery',
        'report_generation': 'Generate Report',
        'dashboard_generation': 'Generate Dashboard'
    }
    
    JOB_STATUSES = {
        'pending': 'Pending',
        'running': 'Running',
        'completed': 'Completed',
        'failed': 'Failed',
        'cancelled': 'Cancelled'
    }
    
    def __init__(self, app=None, max_jobs: int = 5, poll_interval: int = 10, 
                db_path: Optional[str] = None):
        """
        Initialize job scheduler.
        
        Args:
            app: Flask application
            max_jobs: Maximum number of concurrent jobs
            poll_interval: Interval in seconds to poll for job status
            db_path: Path to job database
        """
        self.max_jobs = max_jobs
        self.poll_interval = poll_interval
        
        # Configure job database
        job_db_path = db_path or 'sqlite:///jobs.db'
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler(
            jobstores={
                'default': SQLAlchemyJobStore(url=job_db_path)
            }
        )
        
        # Initialize jobs dictionary
        self.jobs = {}
        
        # Start scheduler
        self.scheduler.start()
        
        # Register with Flask if provided
        if app is not None:
            self._init_app(app)
    
    def _init_app(self, app):
        """Register with Flask application."""
        # Register shutdown function
        @app.teardown_appcontext
        def shutdown_scheduler(exception=None):
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
    
    def add_job(self, job_type: str, job_name: str, user_id: str, 
               func: Callable, args=None, kwargs=None, 
               scheduled_time: Optional[datetime.datetime] = None) -> str:
        """
        Add a job to the scheduler.
        
        Args:
            job_type: Type of job
            job_name: Name of the job
            user_id: ID of the user
            func: Function to execute
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            scheduled_time: Time to schedule the job for
            
        Returns:
            ID of the scheduled job
        """
        if job_type not in self.JOB_TYPES:
            raise ValueError(f"Invalid job type: {job_type}")
        
        # Check if we can add more jobs
        active_jobs = [j for j in self.jobs.values() 
                      if j['status'] in ['pending', 'running']]
        
        if len(active_jobs) >= self.max_jobs:
            raise ValueError(f"Maximum number of concurrent jobs ({self.max_jobs}) reached")
        
        # Create job details
        job_id = str(uuid.uuid4())
        job_details = {
            'id': job_id,
            'type': job_type,
            'name': job_name,
            'user_id': user_id,
            'status': 'pending',
            'created_at': datetime.datetime.now().isoformat(),
            'scheduled_time': scheduled_time.isoformat() if scheduled_time else None,
            'started_at': None,
            'completed_at': None,
            'result': None,
            'error': None
        }
        
        # Store job details
        self.jobs[job_id] = job_details
        
        # Create wrapper function to update job status
        def job_wrapper():
            # Update job status to running
            self.jobs[job_id]['status'] = 'running'
            self.jobs[job_id]['started_at'] = datetime.datetime.now().isoformat()
            
            try:
                # Execute function
                args_list = args or []
                kwargs_dict = kwargs or {}
                result = func(*args_list, **kwargs_dict)
                
                # Update job status to completed
                self.jobs[job_id]['status'] = 'completed'
                self.jobs[job_id]['completed_at'] = datetime.datetime.now().isoformat()
                self.jobs[job_id]['result'] = result
                
                return result
            except Exception as e:
                # Update job status to failed
                self.jobs[job_id]['status'] = 'failed'
                self.jobs[job_id]['completed_at'] = datetime.datetime.now().isoformat()
                self.jobs[job_id]['error'] = str(e)
                
                logger.error(f"Job {job_id} failed: {e}")
                raise
        
        # Schedule job
        if scheduled_time:
            scheduler_job = self.scheduler.add_job(
                job_wrapper,
                'date',
                run_date=scheduled_time,
                id=job_id
            )
        else:
            scheduler_job = self.scheduler.add_job(
                job_wrapper,
                'date',
                run_date=datetime.datetime.now(),
                id=job_id
            )
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job details.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job details or None if not found
        """
        return self.jobs.get(job_id)
    
    def get_jobs(self, user_id: Optional[str] = None, 
                job_type: Optional[str] = None,
                status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get jobs filtered by criteria.
        
        Args:
            user_id: ID of the user
            job_type: Type of job
            status: Status of the job
            
        Returns:
            List of job details
        """
        filtered_jobs = []
        
        for job in self.jobs.values():
            if user_id and job['user_id'] != user_id:
                continue
            
            if job_type and job['type'] != job_type:
                continue
            
            if status and job['status'] != status:
                continue
            
            filtered_jobs.append(job)
        
        # Sort by created_at descending
        filtered_jobs.sort(key=lambda j: j['created_at'], reverse=True)
        
        return filtered_jobs
    
    def cancel_job(self, job_id: str, user_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: ID of the job
            user_id: ID of the user
            
        Returns:
            Whether the job was cancelled
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # Check user
        if job['user_id'] != user_id:
            return False
        
        # Check status
        if job['status'] not in ['pending', 'running']:
            return False
        
        # Cancel job
        try:
            self.scheduler.remove_job(job_id)
        except:
            pass
        
        # Update status
        job['status'] = 'cancelled'
        job['completed_at'] = datetime.datetime.now().isoformat()
        
        return True

class WebApp:
    """
    Web Application for the Simulation Validation Framework.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the web application.
        
        Args:
            config_path: Path to configuration file
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the web application")
        
        if not FRAMEWORK_AVAILABLE:
            raise ImportError("Simulation Validation Framework is required for the web application")
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize the framework
        self.framework = SimulationValidationFramework(config_path)
        
        # Initialize database integration if available
        try:
            self.db_integration = SimulationDBIntegration(
                db_path=self.config["database"].get("db_path", "benchmark_db.duckdb")
            )
            self.db_available = True
        except:
            logger.warning("Database integration not available")
            self.db_integration = None
            self.db_available = False
        
        # Initialize Flask application
        self.app = Flask(__name__,
                         template_folder=os.path.join(os.path.dirname(__file__), "templates"),
                         static_folder=os.path.join(os.path.dirname(__file__), "static"))
        
        # Configure Flask application
        self.app.config['SECRET_KEY'] = self.config.get('secret_key', os.urandom(24).hex())
        self.app.config['SESSION_TYPE'] = 'filesystem'
        self.app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'simulation_validation_uploads')
        
        # Ensure upload directory exists
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Initialize CSRF protection
        self.csrf = CSRFProtect(self.app)
        
        # Initialize notification manager
        self.notification_manager = NotificationManager(
            max_notifications=self.config["notifications"].get("max_notifications", 50),
            db_integration=self.db_integration if self.db_available else None
        )
        
        # Initialize job scheduler
        self.job_scheduler = JobScheduler(
            app=self.app,
            max_jobs=self.config["jobs"].get("max_jobs", 5),
            poll_interval=self.config["jobs"].get("poll_interval", 10),
            db_path=self.config.get("job_db_path")
        )
        
        # Set up routes
        self._setup_routes()
        
        logger.info("Web application initialized")
    
    def run(self, host: str = "localhost", port: int = 5000, debug: bool = False):
        """
        Run the web application.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting web application on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Default configuration
        default_config = {
            "title": "Simulation Validation Framework",
            "theme": "light",
            "pagination_size": 20,
            "database": {
                "enabled": True,
                "db_path": "benchmark_db.duckdb"
            },
            "authentication": {
                "enabled": False,
                "users": {
                    "admin": "password"  # Not secure, just for testing
                }
            },
            "jobs": {
                "max_jobs": 5,
                "poll_interval": 10  # seconds
            },
            "notifications": {
                "enabled": True,
                "max_notifications": 50
            }
        }
        
        # Load configuration from file if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                logger.info("Using default configuration")
        
        # Apply default values for missing keys
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                # Merge nested dictionaries
                for nested_key, nested_value in value.items():
                    if nested_key not in config[key]:
                        config[key][nested_key] = nested_value
        
        return config
    
    def _setup_routes(self):
        """Set up routes for the web application."""
        
        # Add authentication and user management routes
        self._setup_auth_routes()
        
        # Add job management routes
        self._setup_job_routes()
        
        # Add notification routes
        self._setup_notification_routes()
        
        # Add user preferences routes
        self._setup_preferences_routes()
        
        # Add integration routes
        self._setup_integration_routes()
        
        # Set up global context processor
        @self.app.context_processor
        def inject_globals():
            """Inject global variables into templates."""
            context = {
                'site_title': self.config["title"],
                'theme': self.config["theme"],
                'now': datetime.datetime.now(),
                'is_authenticated': 'user_id' in session,
                'user_role': session.get('user_role'),
                'unread_notifications': 0
            }
            
            # Add unread notification count
            if 'user_id' in session:
                try:
                    notifications = self.notification_manager.get_notifications(
                        user_id=session['user_id'],
                        include_read=False
                    )
                    context['unread_notifications'] = len(notifications)
                except:
                    pass
            
            return context

        # Home page
        @self.app.route('/')
        def index():
            """Home page displaying overview and quick links."""
            # Count validation results if database is available
            validation_count = 0
            hardware_count = 0
            model_count = 0
            
            if self.db_available:
                try:
                    validation_count = self.db_integration.count_validation_results()
                    hardware_types = self.db_integration.get_hardware_types()
                    model_types = self.db_integration.get_model_types()
                    hardware_count = len(hardware_types)
                    model_count = len(model_types)
                except Exception as e:
                    logger.error(f"Error fetching database statistics: {e}")
            
            # Get recent validation results
            recent_validations = []
            if self.db_available:
                try:
                    recent_validations = self.db_integration.get_recent_validation_results(limit=5)
                except Exception as e:
                    logger.error(f"Error fetching recent validation results: {e}")
            
            # Get recent jobs if authenticated
            recent_jobs = []
            if 'user_id' in session:
                try:
                    recent_jobs = self.job_scheduler.get_jobs(
                        user_id=session['user_id'],
                        limit=5
                    )
                except Exception as e:
                    logger.error(f"Error fetching recent jobs: {e}")
            
            return render_template(
                'index.html',
                title=self.config["title"],
                theme=self.config["theme"],
                validation_count=validation_count,
                hardware_count=hardware_count,
                model_count=model_count,
                recent_validations=recent_validations,
                recent_jobs=recent_jobs,
                is_authenticated='user_id' in session,
                user_role=session.get('user_role')
            )
        
        # Validation results page
        @self.app.route('/validation')
        def validation_results():
            """Page displaying validation results."""
            # Get query parameters
            page = request.args.get('page', 1, type=int)
            hardware_id = request.args.get('hardware_id', '')
            model_id = request.args.get('model_id', '')
            limit = self.config["pagination_size"]
            
            # Get validation results
            results = []
            total_count = 0
            hardware_types = []
            model_types = []
            
            if self.db_available:
                try:
                    # Get filters
                    filters = {}
                    if hardware_id:
                        filters['hardware_id'] = hardware_id
                    if model_id:
                        filters['model_id'] = model_id
                    
                    # Get results
                    offset = (page - 1) * limit
                    results = self.db_integration.get_validation_results(
                        limit=limit, offset=offset, filters=filters
                    )
                    
                    # Get total count for pagination
                    total_count = self.db_integration.count_validation_results(filters=filters)
                    
                    # Get hardware and model types for filtering
                    hardware_types = self.db_integration.get_hardware_types()
                    model_types = self.db_integration.get_model_types()
                    
                except Exception as e:
                    logger.error(f"Error fetching validation results: {e}")
                    flash(f"Error fetching validation results: {str(e)}", "danger")
            
            # Calculate pagination
            total_pages = (total_count + limit - 1) // limit if total_count > 0 else 1
            
            return render_template(
                'validation_results.html',
                title="Validation Results",
                theme=self.config["theme"],
                results=results,
                total_count=total_count,
                page=page,
                total_pages=total_pages,
                limit=limit,
                hardware_id=hardware_id,
                model_id=model_id,
                hardware_types=hardware_types,
                model_types=model_types
            )
        
        # Validation details page
        @self.app.route('/validation/<validation_id>')
        def validation_details(validation_id):
            """Page displaying details of a validation result."""
            # Get validation result
            validation_result = None
            
            if self.db_available:
                try:
                    validation_result = self.db_integration.get_validation_result(validation_id)
                except Exception as e:
                    logger.error(f"Error fetching validation result {validation_id}: {e}")
                    flash(f"Error fetching validation result: {str(e)}", "danger")
            
            if not validation_result:
                flash("Validation result not found", "danger")
                return redirect(url_for('validation_results'))
            
            return render_template(
                'validation_details.html',
                title="Validation Details",
                theme=self.config["theme"],
                validation=validation_result
            )
        
        # Generate report page
        @self.app.route('/report/generate', methods=['GET', 'POST'])
        def generate_report():
            """Page for generating reports."""
            if request.method == 'POST':
                # Get form data
                hardware_id = request.form.get('hardware_id', '')
                model_id = request.form.get('model_id', '')
                report_format = request.form.get('format', 'html')
                include_visualizations = request.form.get('include_visualizations', 'on') == 'on'
                
                # Generate temporary filename for the report
                report_filename = f"validation_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                if report_format == 'markdown':
                    report_filename += '.md'
                elif report_format == 'json':
                    report_filename += '.json'
                elif report_format == 'csv':
                    report_filename += '.csv'
                elif report_format == 'pdf':
                    report_filename += '.pdf'
                else:
                    report_filename += '.html'
                
                report_path = os.path.join(self.app.config['UPLOAD_FOLDER'], report_filename)
                
                # Get validation results
                validation_results = []
                if self.db_available:
                    try:
                        filters = {}
                        if hardware_id:
                            filters['hardware_id'] = hardware_id
                        if model_id:
                            filters['model_id'] = model_id
                        
                        # Get results
                        db_results = self.db_integration.get_validation_results(filters=filters)
                        
                        # Convert to validation result objects
                        validation_results = self.db_integration.convert_to_validation_results(db_results)
                    except Exception as e:
                        logger.error(f"Error fetching validation results: {e}")
                        flash(f"Error fetching validation results: {str(e)}", "danger")
                
                if not validation_results:
                    flash("No validation results found for the selected criteria", "warning")
                    return redirect(url_for('generate_report'))
                
                # Generate report
                try:
                    # Generate and save the report
                    report_path = self.framework.generate_report(
                        validation_results,
                        format=report_format,
                        include_visualizations=include_visualizations,
                        output_path=report_path,
                        hardware_id=hardware_id if hardware_id else None,
                        model_id=model_id if model_id else None
                    )
                    
                    # Store report path in session for download
                    session['report_path'] = report_path
                    session['report_filename'] = report_filename
                    
                    flash("Report generated successfully", "success")
                    return redirect(url_for('download_report'))
                    
                except Exception as e:
                    logger.error(f"Error generating report: {e}")
                    flash(f"Error generating report: {str(e)}", "danger")
                    return redirect(url_for('generate_report'))
            
            # GET request - show form
            hardware_types = []
            model_types = []
            
            if self.db_available:
                try:
                    hardware_types = self.db_integration.get_hardware_types()
                    model_types = self.db_integration.get_model_types()
                except Exception as e:
                    logger.error(f"Error fetching hardware and model types: {e}")
            
            return render_template(
                'generate_report.html',
                title="Generate Report",
                theme=self.config["theme"],
                hardware_types=hardware_types,
                model_types=model_types
            )
        
        # Download report
        @self.app.route('/report/download')
        def download_report():
            """Download the generated report."""
            report_path = session.get('report_path')
            report_filename = session.get('report_filename')
            
            if not report_path or not os.path.exists(report_path):
                flash("Report not found or has expired", "danger")
                return redirect(url_for('generate_report'))
            
            return send_file(report_path, as_attachment=True, download_name=report_filename)
        
        # Visualization dashboard
        @self.app.route('/dashboard')
        def dashboard():
            """Page displaying the visualization dashboard."""
            # Get query parameters
            hardware_id = request.args.get('hardware_id', '')
            model_id = request.args.get('model_id', '')
            
            # Generate dashboard path
            dashboard_filename = f"dashboard_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
            dashboard_path = os.path.join(self.app.config['UPLOAD_FOLDER'], dashboard_filename)
            
            # Get validation results
            validation_results = []
            if self.db_available:
                try:
                    filters = {}
                    if hardware_id:
                        filters['hardware_id'] = hardware_id
                    if model_id:
                        filters['model_id'] = model_id
                    
                    # Get results
                    db_results = self.db_integration.get_validation_results(filters=filters)
                    
                    # Convert to validation result objects
                    validation_results = self.db_integration.convert_to_validation_results(db_results)
                except Exception as e:
                    logger.error(f"Error fetching validation results: {e}")
                    flash(f"Error fetching validation results: {str(e)}", "danger")
            
            if not validation_results:
                flash("No validation results found for the selected criteria", "warning")
                return redirect(url_for('index'))
            
            # Generate dashboard
            try:
                dashboard_path = self.framework.create_comprehensive_dashboard(
                    validation_results,
                    hardware_id=hardware_id if hardware_id else None,
                    model_id=model_id if model_id else None,
                    output_path=dashboard_path,
                    title="Simulation Validation Dashboard"
                )
                
                # Store dashboard path in session for iframe
                session['dashboard_path'] = dashboard_path
                session['dashboard_filename'] = dashboard_filename
                
                # Determine URL for the dashboard
                dashboard_url = url_for('static_dashboard', filename=dashboard_filename)
                
                return render_template(
                    'dashboard.html',
                    title="Visualization Dashboard",
                    theme=self.config["theme"],
                    dashboard_url=dashboard_url
                )
                
            except Exception as e:
                logger.error(f"Error generating dashboard: {e}")
                flash(f"Error generating dashboard: {str(e)}", "danger")
                return redirect(url_for('index'))
        
        # Serve static dashboard files
        @self.app.route('/dashboards/<filename>')
        def static_dashboard(filename):
            """Serve static dashboard files."""
            return send_file(os.path.join(self.app.config['UPLOAD_FOLDER'], filename))
        
        # Validate page (form for running validation)
        @self.app.route('/validate', methods=['GET', 'POST'])
        def validate():
            """Page for running validation."""
            if request.method == 'POST':
                # Get form data
                hardware_id = request.form.get('hardware_id', '')
                model_id = request.form.get('model_id', '')
                batch_size = request.form.get('batch_size', 1, type=int)
                precision = request.form.get('precision', 'float32')
                protocol = request.form.get('protocol', 'standard')
                
                # Create simulation result
                simulation_metrics = {
                    'throughput_items_per_second': request.form.get('sim_throughput', 0, type=float),
                    'average_latency_ms': request.form.get('sim_latency', 0, type=float),
                    'memory_peak_mb': request.form.get('sim_memory', 0, type=float),
                    'power_consumption_w': request.form.get('sim_power', 0, type=float)
                }
                
                simulation_result = SimulationResult(
                    model_id=model_id,
                    hardware_id=hardware_id,
                    metrics=simulation_metrics,
                    batch_size=batch_size,
                    precision=precision,
                    timestamp=datetime.datetime.now().isoformat(),
                    simulation_version="1.0",
                    additional_metadata={}
                )
                
                # Create hardware result
                hardware_metrics = {
                    'throughput_items_per_second': request.form.get('hw_throughput', 0, type=float),
                    'average_latency_ms': request.form.get('hw_latency', 0, type=float),
                    'memory_peak_mb': request.form.get('hw_memory', 0, type=float),
                    'power_consumption_w': request.form.get('hw_power', 0, type=float)
                }
                
                hardware_result = HardwareResult(
                    model_id=model_id,
                    hardware_id=hardware_id,
                    metrics=hardware_metrics,
                    batch_size=batch_size,
                    precision=precision,
                    timestamp=datetime.datetime.now().isoformat(),
                    hardware_details={},
                    test_environment={},
                    additional_metadata={}
                )
                
                # Run validation
                try:
                    validation_results = self.framework.validate(
                        [simulation_result],
                        [hardware_result],
                        protocol=protocol
                    )
                    
                    if not validation_results:
                        flash("Validation failed to produce results", "danger")
                        return redirect(url_for('validate'))
                    
                    # Store validation results in database if available
                    if self.db_available:
                        try:
                            for result in validation_results:
                                self.db_integration.store_validation_result(result)
                            flash("Validation results stored in database", "success")
                        except Exception as e:
                            logger.error(f"Error storing validation results: {e}")
                            flash(f"Error storing validation results: {str(e)}", "warning")
                    
                    # Store validation result ID in session for redirect
                    session['validation_id'] = validation_results[0].id if hasattr(validation_results[0], 'id') else None
                    
                    flash("Validation completed successfully", "success")
                    
                    # Redirect to validation details if we have an ID, otherwise to results page
                    if session.get('validation_id'):
                        return redirect(url_for('validation_details', validation_id=session['validation_id']))
                    else:
                        return redirect(url_for('validation_results'))
                    
                except Exception as e:
                    logger.error(f"Error running validation: {e}")
                    flash(f"Error running validation: {str(e)}", "danger")
                    return redirect(url_for('validate'))
            
            # GET request - show form
            hardware_types = []
            model_types = []
            
            if self.db_available:
                try:
                    hardware_types = self.db_integration.get_hardware_types()
                    model_types = self.db_integration.get_model_types()
                except Exception as e:
                    logger.error(f"Error fetching hardware and model types: {e}")
            
            return render_template(
                'validate.html',
                title="Run Validation",
                theme=self.config["theme"],
                hardware_types=hardware_types,
                model_types=model_types
            )
        
        # Calibration page
        @self.app.route('/calibrate', methods=['GET', 'POST'])
        def calibrate():
            """Page for running calibration."""
            if request.method == 'POST':
                # Get form data
                hardware_id = request.form.get('hardware_id', '')
                model_id = request.form.get('model_id', '')
                
                # Get validation results
                validation_results = []
                if self.db_available:
                    try:
                        filters = {
                            'hardware_id': hardware_id,
                            'model_id': model_id
                        }
                        
                        # Get results
                        db_results = self.db_integration.get_validation_results(filters=filters)
                        
                        # Convert to validation result objects
                        validation_results = self.db_integration.convert_to_validation_results(db_results)
                    except Exception as e:
                        logger.error(f"Error fetching validation results: {e}")
                        flash(f"Error fetching validation results: {str(e)}", "danger")
                
                if not validation_results:
                    flash("No validation results found for the selected criteria", "warning")
                    return redirect(url_for('calibrate'))
                
                # Get simulation parameters from form
                simulation_parameters = {}
                for key in request.form.keys():
                    if key.startswith('param_'):
                        param_name = key[6:]  # Remove 'param_' prefix
                        param_value = request.form.get(key)
                        
                        # Try to convert to appropriate type
                        try:
                            if '.' in param_value:
                                param_value = float(param_value)
                            else:
                                param_value = int(param_value)
                        except ValueError:
                            # Keep as string if conversion fails
                            pass
                        
                        simulation_parameters[param_name] = param_value
                
                # Run calibration
                try:
                    updated_parameters = self.framework.calibrate(
                        validation_results,
                        simulation_parameters
                    )
                    
                    # Store updated parameters in session for display
                    session['previous_parameters'] = simulation_parameters
                    session['updated_parameters'] = updated_parameters
                    
                    flash("Calibration completed successfully", "success")
                    return redirect(url_for('calibration_results'))
                    
                except Exception as e:
                    logger.error(f"Error running calibration: {e}")
                    flash(f"Error running calibration: {str(e)}", "danger")
                    return redirect(url_for('calibrate'))
            
            # GET request - show form
            hardware_types = []
            model_types = []
            
            if self.db_available:
                try:
                    hardware_types = self.db_integration.get_hardware_types()
                    model_types = self.db_integration.get_model_types()
                except Exception as e:
                    logger.error(f"Error fetching hardware and model types: {e}")
            
            return render_template(
                'calibrate.html',
                title="Run Calibration",
                theme=self.config["theme"],
                hardware_types=hardware_types,
                model_types=model_types
            )
        
        # Calibration results page
        @self.app.route('/calibration/results')
        def calibration_results():
            """Page displaying calibration results."""
            previous_parameters = session.get('previous_parameters', {})
            updated_parameters = session.get('updated_parameters', {})
            
            if not updated_parameters:
                flash("No calibration results available", "warning")
                return redirect(url_for('calibrate'))
            
            return render_template(
                'calibration_results.html',
                title="Calibration Results",
                theme=self.config["theme"],
                previous_parameters=previous_parameters,
                updated_parameters=updated_parameters
            )
        
        # Drift detection page
        @self.app.route('/drift', methods=['GET', 'POST'])
        def drift_detection():
            """Page for running drift detection."""
            if request.method == 'POST':
                # Get form data
                hardware_id = request.form.get('hardware_id', '')
                model_id = request.form.get('model_id', '')
                historical_days = request.form.get('historical_days', 30, type=int)
                
                # Get validation results
                all_validation_results = []
                if self.db_available:
                    try:
                        filters = {
                            'hardware_id': hardware_id,
                            'model_id': model_id
                        }
                        
                        # Get all results
                        db_results = self.db_integration.get_validation_results(filters=filters)
                        
                        # Convert to validation result objects
                        all_validation_results = self.db_integration.convert_to_validation_results(db_results)
                    except Exception as e:
                        logger.error(f"Error fetching validation results: {e}")
                        flash(f"Error fetching validation results: {str(e)}", "danger")
                
                if not all_validation_results:
                    flash("No validation results found for the selected criteria", "warning")
                    return redirect(url_for('drift_detection'))
                
                # Split results into historical and new
                current_time = datetime.datetime.now()
                cutoff_time = current_time - datetime.timedelta(days=historical_days)
                
                historical_results = []
                new_results = []
                
                for result in all_validation_results:
                    result_time = datetime.datetime.fromisoformat(result.validation_timestamp)
                    if result_time < cutoff_time:
                        historical_results.append(result)
                    else:
                        new_results.append(result)
                
                if not historical_results:
                    flash(f"No historical validation results found (older than {historical_days} days)", "warning")
                    return redirect(url_for('drift_detection'))
                
                if not new_results:
                    flash("No recent validation results found", "warning")
                    return redirect(url_for('drift_detection'))
                
                # Run drift detection
                try:
                    drift_results = self.framework.detect_drift(
                        historical_results,
                        new_results
                    )
                    
                    # Store drift results in session for display
                    session['drift_results'] = drift_results
                    
                    flash("Drift detection completed successfully", "success")
                    return redirect(url_for('drift_results'))
                    
                except Exception as e:
                    logger.error(f"Error running drift detection: {e}")
                    flash(f"Error running drift detection: {str(e)}", "danger")
                    return redirect(url_for('drift_detection'))
            
            # GET request - show form
            hardware_types = []
            model_types = []
            
            if self.db_available:
                try:
                    hardware_types = self.db_integration.get_hardware_types()
                    model_types = self.db_integration.get_model_types()
                except Exception as e:
                    logger.error(f"Error fetching hardware and model types: {e}")
            
            return render_template(
                'drift_detection.html',
                title="Run Drift Detection",
                theme=self.config["theme"],
                hardware_types=hardware_types,
                model_types=model_types
            )
        
        # Drift detection results page
        @self.app.route('/drift/results')
        def drift_results():
            """Page displaying drift detection results."""
            drift_results = session.get('drift_results', {})
            
            if not drift_results:
                flash("No drift detection results available", "warning")
                return redirect(url_for('drift_detection'))
            
            # Generate drift visualization
            visualization_path = None
            try:
                # Generate filename
                viz_filename = f"drift_viz_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
                viz_path = os.path.join(self.app.config['UPLOAD_FOLDER'], viz_filename)
                
                # Generate visualization
                visualization_path = self.framework.visualize_drift_detection(
                    drift_results,
                    interactive=True,
                    output_path=viz_path
                )
                
                # Get URL
                if visualization_path:
                    visualization_url = url_for('static_dashboard', filename=os.path.basename(visualization_path))
                else:
                    visualization_url = None
                
            except Exception as e:
                logger.error(f"Error generating drift visualization: {e}")
                visualization_url = None
            
            return render_template(
                'drift_results.html',
                title="Drift Detection Results",
                theme=self.config["theme"],
                drift_results=drift_results,
                visualization_url=visualization_url
            )
        
        # Parameter discovery page
        @self.app.route('/parameters', methods=['GET', 'POST'])
        def parameter_discovery():
            """Page for running parameter discovery."""
            if request.method == 'POST':
                # Get form data
                hardware_id = request.form.get('hardware_id', '')
                model_id = request.form.get('model_id', '')
                
                # Get validation results
                validation_results = []
                if self.db_available:
                    try:
                        filters = {
                            'hardware_id': hardware_id,
                            'model_id': model_id
                        }
                        
                        # Get results
                        db_results = self.db_integration.get_validation_results(filters=filters)
                        
                        # Convert to validation result objects
                        validation_results = self.db_integration.convert_to_validation_results(db_results)
                    except Exception as e:
                        logger.error(f"Error fetching validation results: {e}")
                        flash(f"Error fetching validation results: {str(e)}", "danger")
                
                if not validation_results:
                    flash("No validation results found for the selected criteria", "warning")
                    return redirect(url_for('parameter_discovery'))
                
                # Run parameter discovery
                try:
                    parameter_recommendations = self.framework.discover_parameters(
                        validation_results
                    )
                    
                    # Store parameter recommendations in session for display
                    session['parameter_recommendations'] = parameter_recommendations
                    
                    flash("Parameter discovery completed successfully", "success")
                    return redirect(url_for('parameter_results'))
                    
                except Exception as e:
                    logger.error(f"Error running parameter discovery: {e}")
                    flash(f"Error running parameter discovery: {str(e)}", "danger")
                    return redirect(url_for('parameter_discovery'))
            
            # GET request - show form
            hardware_types = []
            model_types = []
            
            if self.db_available:
                try:
                    hardware_types = self.db_integration.get_hardware_types()
                    model_types = self.db_integration.get_model_types()
                except Exception as e:
                    logger.error(f"Error fetching hardware and model types: {e}")
            
            return render_template(
                'parameter_discovery.html',
                title="Parameter Discovery",
                theme=self.config["theme"],
                hardware_types=hardware_types,
                model_types=model_types
            )
        
        # Parameter discovery results page
        @self.app.route('/parameters/results')
        def parameter_results():
            """Page displaying parameter discovery results."""
            parameter_recommendations = session.get('parameter_recommendations', {})
            
            if not parameter_recommendations:
                flash("No parameter discovery results available", "warning")
                return redirect(url_for('parameter_discovery'))
            
            return render_template(
                'parameter_results.html',
                title="Parameter Discovery Results",
                theme=self.config["theme"],
                parameter_recommendations=parameter_recommendations
            )
        
        # Settings page
        @self.app.route('/settings', methods=['GET', 'POST'])
        def settings():
            """Page for application settings."""
            if request.method == 'POST':
                # Get form data
                theme = request.form.get('theme', 'light')
                page_size = request.form.get('page_size', 20, type=int)
                
                # Update configuration
                self.config["theme"] = theme
                self.config["pagination_size"] = page_size
                
                flash("Settings updated successfully", "success")
                return redirect(url_for('settings'))
            
            # Get framework version if available
            framework_version = None
            try:
                if hasattr(self.framework, 'version'):
                    framework_version = self.framework.version
                else:
                    framework_version = "1.0.0"
            except:
                framework_version = "Unknown"
            
            return render_template(
                'settings.html',
                title="Settings",
                theme=self.config["theme"],
                page_size=self.config["pagination_size"],
                db_path=self.config["database"].get("db_path", "benchmark_db.duckdb"),
                db_enabled=self.config["database"].get("enabled", True),
                db_available=self.db_available,
                framework_available=FRAMEWORK_AVAILABLE,
                framework_version=framework_version,
                environment=os.environ.get("ENVIRONMENT", "Production")
            )
        
        # API endpoints
        @self.app.route('/api/status')
        def api_status():
            """API endpoint for system status."""
            status = {
                "status": "ok",
                "timestamp": datetime.datetime.now().isoformat(),
                "framework_available": FRAMEWORK_AVAILABLE,
                "database_available": self.db_available
            }
            
            # Add database statistics if available
            if self.db_available:
                try:
                    status["validation_count"] = self.db_integration.count_validation_results()
                    status["hardware_types"] = self.db_integration.get_hardware_types()
                    status["model_types"] = self.db_integration.get_model_types()
                except Exception as e:
                    logger.error(f"Error fetching database statistics: {e}")
                    status["database_error"] = str(e)
            
            return jsonify(status)
        
        # End of route setup
        logger.info("Routes have been set up")
    
    def _setup_auth_routes(self):
        """Set up authentication and user management routes."""
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """User login page."""
            # Redirect if already logged in
            if 'user_id' in session:
                return redirect(url_for('index'))
            
            form = LoginForm()
            
            if form.validate_on_submit():
                username = form.username.data
                password = form.password.data
                
                # Verify user credentials
                user = self._verify_user(username, password)
                
                if user:
                    # Store user data in session
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    session['user_role'] = user['role']
                    
                    # Add login notification
                    self.notification_manager.add_notification(
                        user_id=user['id'],
                        message=f"Welcome back, {username}! Last login: {user.get('last_login', 'first login')}",
                        notification_type='info'
                    )
                    
                    # Update last login time
                    user['last_login'] = datetime.datetime.now().isoformat()
                    self._update_user(user['id'], user)
                    
                    # Redirect to requested page or default
                    next_page = request.args.get('next', url_for('index'))
                    return redirect(next_page)
                else:
                    flash('Invalid username or password', 'danger')
            
            return render_template(
                'login.html',
                title="Login",
                theme=self.config["theme"],
                form=form
            )
        
        @self.app.route('/logout')
        def logout():
            """User logout."""
            # Add logout notification before clearing session
            if 'user_id' in session:
                self.notification_manager.add_notification(
                    user_id=session['user_id'],
                    message=f"You have been logged out",
                    notification_type='info',
                    store_in_db=False  # Don't store logout notifications
                )
            
            # Clear session
            session.pop('user_id', None)
            session.pop('username', None)
            session.pop('user_role', None)
            
            flash('You have been logged out', 'info')
            return redirect(url_for('login'))
        
        @self.app.route('/register', methods=['GET', 'POST'])
        @role_required(UserRole.ADMIN)  # Only admins can register new users
        def register():
            """User registration page."""
            form = RegisterForm()
            
            if form.validate_on_submit():
                username = form.username.data
                
                # Check if username already exists
                if self._get_user_by_username(username):
                    flash('Username already exists', 'danger')
                    return render_template(
                        'register.html',
                        title="Register",
                        theme=self.config["theme"],
                        form=form
                    )
                
                # Create new user
                user = {
                    'id': str(uuid.uuid4()),
                    'username': username,
                    'email': form.email.data,
                    'password_hash': generate_password_hash(form.password.data),
                    'role': form.role.data,
                    'created_at': datetime.datetime.now().isoformat(),
                    'last_login': None,
                    'preferences': {
                        'theme': self.config["theme"],
                        'pagination_size': self.config["pagination_size"]
                    }
                }
                
                # Store user
                self._add_user(user)
                
                # Add notification for admin
                self.notification_manager.add_notification(
                    user_id=session['user_id'],
                    message=f"User {username} registered successfully",
                    notification_type='success'
                )
                
                flash(f'User {username} registered successfully', 'success')
                return redirect(url_for('manage_users'))
            
            return render_template(
                'register.html',
                title="Register",
                theme=self.config["theme"],
                form=form
            )
        
        @self.app.route('/profile', methods=['GET'])
        @login_required
        def profile():
            """User profile page."""
            # Get user data
            user = self._get_user(session['user_id'])
            
            if not user:
                flash('User not found', 'danger')
                return redirect(url_for('logout'))
            
            # Get user jobs
            jobs = self.job_scheduler.get_jobs(
                user_id=session['user_id'],
                limit=10
            )
            
            # Get user notifications
            notifications = self.notification_manager.get_notifications(
                user_id=session['user_id'],
                limit=10,
                include_read=True
            )
            
            return render_template(
                'profile.html',
                title="User Profile",
                theme=self.config["theme"],
                user=user,
                jobs=jobs,
                notifications=notifications
            )
        
        @self.app.route('/profile/reset-password', methods=['GET', 'POST'])
        @login_required
        def reset_password():
            """Reset password page."""
            form = ResetPasswordForm()
            
            if form.validate_on_submit():
                # Get user data
                user = self._get_user(session['user_id'])
                
                if not user:
                    flash('User not found', 'danger')
                    return redirect(url_for('logout'))
                
                # Verify old password
                if not check_password_hash(user['password_hash'], form.old_password.data):
                    flash('Current password is incorrect', 'danger')
                    return render_template(
                        'reset_password.html',
                        title="Reset Password",
                        theme=self.config["theme"],
                        form=form
                    )
                
                # Update password
                user['password_hash'] = generate_password_hash(form.new_password.data)
                self._update_user(user['id'], user)
                
                # Add notification
                self.notification_manager.add_notification(
                    user_id=session['user_id'],
                    message=f"Your password has been reset",
                    notification_type='success'
                )
                
                flash('Password reset successfully', 'success')
                return redirect(url_for('profile'))
            
            return render_template(
                'reset_password.html',
                title="Reset Password",
                theme=self.config["theme"],
                form=form
            )
        
        @self.app.route('/manage/users', methods=['GET'])
        @role_required(UserRole.ADMIN)  # Only admins can manage users
        def manage_users():
            """User management page."""
            # Get all users
            users = self._get_all_users()
            
            return render_template(
                'manage_users.html',
                title="Manage Users",
                theme=self.config["theme"],
                users=users
            )
        
        @self.app.route('/manage/users/<user_id>', methods=['GET', 'POST'])
        @role_required(UserRole.ADMIN)  # Only admins can manage users
        def edit_user(user_id):
            """Edit user page."""
            # Get user data
            user = self._get_user(user_id)
            
            if not user:
                flash('User not found', 'danger')
                return redirect(url_for('manage_users'))
            
            # Create form
            form = RegisterForm(obj=user)
            
            if form.validate_on_submit():
                # Update user
                user['username'] = form.username.data
                user['email'] = form.email.data
                user['role'] = form.role.data
                
                # Update password if provided
                if form.password.data:
                    user['password_hash'] = generate_password_hash(form.password.data)
                
                # Update user
                self._update_user(user['id'], user)
                
                # Add notification
                self.notification_manager.add_notification(
                    user_id=session['user_id'],
                    message=f"User {user['username']} updated successfully",
                    notification_type='success'
                )
                
                flash(f'User {user["username"]} updated successfully', 'success')
                return redirect(url_for('manage_users'))
            
            return render_template(
                'edit_user.html',
                title="Edit User",
                theme=self.config["theme"],
                form=form,
                user=user
            )
        
        @self.app.route('/manage/users/<user_id>/delete', methods=['POST'])
        @role_required(UserRole.ADMIN)  # Only admins can manage users
        def delete_user(user_id):
            """Delete user."""
            # Cannot delete self
            if user_id == session['user_id']:
                flash('You cannot delete yourself', 'danger')
                return redirect(url_for('manage_users'))
            
            # Get user data
            user = self._get_user(user_id)
            
            if not user:
                flash('User not found', 'danger')
                return redirect(url_for('manage_users'))
            
            # Delete user
            self._delete_user(user_id)
            
            # Add notification
            self.notification_manager.add_notification(
                user_id=session['user_id'],
                message=f"User {user['username']} deleted successfully",
                notification_type='success'
            )
            
            flash(f'User {user["username"]} deleted successfully', 'success')
            return redirect(url_for('manage_users'))
    
    def _setup_job_routes(self):
        """Set up job management routes."""
        
        @self.app.route('/jobs')
        @login_required
        def jobs():
            """Job management page."""
            # Get filter parameters
            job_type = request.args.get('type')
            status = request.args.get('status')
            
            # Get jobs for current user (admins can see all jobs)
            if session.get('user_role') == UserRole.ADMIN and request.args.get('all') == '1':
                user_id = None  # Get all jobs
            else:
                user_id = session['user_id']
            
            jobs = self.job_scheduler.get_jobs(
                user_id=user_id,
                job_type=job_type,
                status=status
            )
            
            return render_template(
                'jobs.html',
                title="Jobs",
                theme=self.config["theme"],
                jobs=jobs,
                job_types=self.job_scheduler.JOB_TYPES,
                job_statuses=self.job_scheduler.JOB_STATUSES,
                selected_type=job_type,
                selected_status=status
            )
        
        @self.app.route('/jobs/<job_id>')
        @login_required
        def job_details(job_id):
            """Job details page."""
            # Get job
            job = self.job_scheduler.get_job(job_id)
            
            if not job:
                flash('Job not found', 'danger')
                return redirect(url_for('jobs'))
            
            # Check permission (admins can see all jobs)
            if job['user_id'] != session['user_id'] and session.get('user_role') != UserRole.ADMIN:
                flash('You do not have permission to view this job', 'danger')
                return redirect(url_for('jobs'))
            
            return render_template(
                'job_details.html',
                title="Job Details",
                theme=self.config["theme"],
                job=job,
                job_types=self.job_scheduler.JOB_TYPES,
                job_statuses=self.job_scheduler.JOB_STATUSES
            )
        
        @self.app.route('/jobs/<job_id>/cancel', methods=['POST'])
        @login_required
        def cancel_job(job_id):
            """Cancel a job."""
            # Cancel job
            result = self.job_scheduler.cancel_job(job_id, session['user_id'])
            
            if result:
                # Add notification
                self.notification_manager.add_notification(
                    user_id=session['user_id'],
                    message=f"Job cancelled successfully",
                    notification_type='success',
                    related_entity='job',
                    entity_id=job_id
                )
                
                flash('Job cancelled successfully', 'success')
            else:
                flash('Failed to cancel job', 'danger')
            
            return redirect(url_for('job_details', job_id=job_id))
    
    def _setup_notification_routes(self):
        """Set up notification routes."""
        
        @self.app.route('/notifications')
        @login_required
        def notifications():
            """Notifications page."""
            # Get all notifications for current user
            user_notifications = self.notification_manager.get_notifications(
                user_id=session['user_id'],
                limit=100,  # Get more for this page
                include_read=True
            )
            
            return render_template(
                'notifications.html',
                title="Notifications",
                theme=self.config["theme"],
                notifications=user_notifications,
                notification_types=self.notification_manager.NOTIFICATION_TYPES
            )
        
        @self.app.route('/notifications/mark-read/<notification_id>', methods=['POST'])
        @login_required
        def mark_notification_read(notification_id):
            """Mark a notification as read."""
            # Mark as read
            result = self.notification_manager.mark_as_read(notification_id, session['user_id'])
            
            if not result:
                flash('Notification not found', 'danger')
            
            # Return to previous page or notifications page
            next_page = request.args.get('next', url_for('notifications'))
            return redirect(next_page)
        
        @self.app.route('/notifications/clear', methods=['POST'])
        @login_required
        def clear_notifications():
            """Clear all notifications."""
            # Clear notifications
            count = self.notification_manager.clear_notifications(session['user_id'])
            
            flash(f'{count} notifications cleared', 'success')
            return redirect(url_for('notifications'))
        
        @self.app.route('/api/notifications', methods=['GET'])
        @login_required
        def api_notifications():
            """API endpoint for notifications."""
            # Get unread notifications
            notifications = self.notification_manager.get_notifications(
                user_id=session['user_id'],
                include_read=False
            )
            
            return jsonify({
                'count': len(notifications),
                'notifications': notifications
            })
    
    def _setup_preferences_routes(self):
        """Set up user preferences routes."""
        
        @self.app.route('/preferences', methods=['GET', 'POST'])
        @login_required
        def preferences():
            """User preferences page."""
            # Get user
            user = self._get_user(session['user_id'])
            
            if not user:
                flash('User not found', 'danger')
                return redirect(url_for('logout'))
            
            if request.method == 'POST':
                # Update preferences
                user_preferences = user.get('preferences', {})
                
                # Update theme
                if 'theme' in request.form:
                    user_preferences['theme'] = request.form['theme']
                    # Also update session theme for immediate effect
                    self.config['theme'] = request.form['theme']
                
                # Update pagination size
                if 'pagination_size' in request.form:
                    try:
                        pagination_size = int(request.form['pagination_size'])
                        user_preferences['pagination_size'] = pagination_size
                        # Also update session pagination size for immediate effect
                        self.config['pagination_size'] = pagination_size
                    except:
                        flash('Invalid pagination size', 'danger')
                
                # Update notifications preferences
                if 'notifications_enabled' in request.form:
                    user_preferences['notifications_enabled'] = request.form['notifications_enabled'] == 'on'
                
                # Update other preferences as needed
                
                # Save preferences
                user['preferences'] = user_preferences
                self._update_user(user['id'], user)
                
                flash('Preferences updated successfully', 'success')
                return redirect(url_for('preferences'))
            
            # Get user preferences or defaults
            user_preferences = user.get('preferences', {})
            
            return render_template(
                'preferences.html',
                title="Preferences",
                theme=self.config["theme"],
                user=user,
                preferences=user_preferences
            )
    
    def _setup_integration_routes(self):
        """Set up integration routes for CI/CD and webhooks."""
        
        @self.app.route('/api/ci-status', methods=['GET'])
        def ci_status():
            """API endpoint for CI/CD integration status."""
            # Get the last CI/CD run from the database
            ci_status = {
                "status": "ok",
                "timestamp": datetime.datetime.now().isoformat(),
                "last_ci_run": None,
                "validation_results": []
            }
            
            # Add database statistics if available
            if self.db_available:
                try:
                    # Get last CI run
                    last_ci_run = self.db_integration.get_last_ci_run()
                    if last_ci_run:
                        ci_status["last_ci_run"] = {
                            "run_id": last_ci_run.get("run_id"),
                            "timestamp": last_ci_run.get("timestamp"),
                            "status": last_ci_run.get("status"),
                            "commit_id": last_ci_run.get("commit_id")
                        }
                    
                    # Get validation results from the last CI run
                    if last_ci_run and "run_id" in last_ci_run:
                        run_id = last_ci_run["run_id"]
                        validation_results = self.db_integration.get_validation_results_by_ci_run(run_id)
                        ci_status["validation_results"] = validation_results
                except Exception as e:
                    logger.error(f"Error fetching CI/CD status: {e}")
                    ci_status["error"] = str(e)
            
            return jsonify(ci_status)
        
        @self.app.route('/api/ci-trigger', methods=['POST'])
        def ci_trigger():
            """API endpoint for CI/CD trigger."""
            # Validate request
            if not request.json:
                return jsonify({"status": "error", "message": "Invalid request"}), 400
            
            # Validate API key
            api_key = request.headers.get('X-API-Key')
            if not api_key or not self._validate_ci_api_key(api_key):
                return jsonify({"status": "error", "message": "Invalid API key"}), 401
            
            # Extract parameters
            run_id = request.json.get('run_id')
            commit_id = request.json.get('commit_id')
            timestamp = request.json.get('timestamp', datetime.datetime.now().isoformat())
            validation_results = request.json.get('validation_results', [])
            
            if not run_id or not validation_results:
                return jsonify({"status": "error", "message": "Missing required parameters"}), 400
            
            # Store CI run and validation results in database
            if self.db_available:
                try:
                    # Store CI run
                    ci_run = {
                        "run_id": run_id,
                        "commit_id": commit_id,
                        "timestamp": timestamp,
                        "status": "completed",
                        "results": validation_results
                    }
                    self.db_integration.store_ci_run(ci_run)
                    
                    # Store validation results
                    for result in validation_results:
                        result["ci_run_id"] = run_id
                        self.db_integration.store_validation_result(result)
                    
                    # Notify admins
                    for user in self._get_users_by_role(UserRole.ADMIN):
                        self.notification_manager.add_notification(
                            user_id=user['id'],
                            message=f"New CI run completed: {run_id}",
                            notification_type='info',
                            related_entity='ci_run',
                            entity_id=run_id
                        )
                    
                    return jsonify({"status": "success", "message": "CI run stored successfully"})
                except Exception as e:
                    logger.error(f"Error storing CI run: {e}")
                    return jsonify({"status": "error", "message": str(e)}), 500
            else:
                return jsonify({"status": "error", "message": "Database not available"}), 500
        
        @self.app.route('/api/webhooks/validation', methods=['POST'])
        def webhook_validation():
            """Webhook for receiving validation results."""
            # Validate request
            if not request.json:
                return jsonify({"status": "error", "message": "Invalid request"}), 400
            
            # Validate API key
            api_key = request.headers.get('X-API-Key')
            if not api_key or not self._validate_webhook_api_key(api_key):
                return jsonify({"status": "error", "message": "Invalid API key"}), 401
            
            # Extract parameters
            validation_result = request.json
            
            if not validation_result:
                return jsonify({"status": "error", "message": "Missing validation result"}), 400
            
            # Store validation result in database
            if self.db_available:
                try:
                    self.db_integration.store_validation_result(validation_result)
                    
                    # Notify admins
                    for user in self._get_users_by_role(UserRole.ADMIN):
                        self.notification_manager.add_notification(
                            user_id=user['id'],
                            message=f"New validation result received via webhook",
                            notification_type='info',
                            related_entity='validation',
                            entity_id=validation_result.get('id')
                        )
                    
                    return jsonify({"status": "success", "message": "Validation result stored successfully"})
                except Exception as e:
                    logger.error(f"Error storing validation result: {e}")
                    return jsonify({"status": "error", "message": str(e)}), 500
            else:
                return jsonify({"status": "error", "message": "Database not available"}), 500
        
        @self.app.route('/integrations', methods=['GET'])
        @role_required(UserRole.ADMIN)  # Only admins can view integrations
        def integrations():
            """Integrations configuration page."""
            # Get CI/CD integrations
            ci_integrations = self._get_ci_integrations()
            
            # Get webhook integrations
            webhook_integrations = self._get_webhook_integrations()
            
            return render_template(
                'integrations.html',
                title="Integrations",
                theme=self.config["theme"],
                ci_integrations=ci_integrations,
                webhook_integrations=webhook_integrations
            )
        
        @self.app.route('/integrations/ci/generate-key', methods=['POST'])
        @role_required(UserRole.ADMIN)  # Only admins can manage integrations
        def generate_ci_api_key():
            """Generate a new CI/CD API key."""
            # Generate key
            api_key = str(uuid.uuid4())
            
            # Store key
            if not self._add_ci_api_key(api_key):
                flash('Failed to generate API key', 'danger')
                return redirect(url_for('integrations'))
            
            # Add notification
            self.notification_manager.add_notification(
                user_id=session['user_id'],
                message=f"New CI/CD API key generated",
                notification_type='success'
            )
            
            flash('New CI/CD API key generated successfully', 'success')
            session['temp_ci_api_key'] = api_key  # Store for display
            return redirect(url_for('integrations'))
        
        @self.app.route('/integrations/webhook/generate-key', methods=['POST'])
        @role_required(UserRole.ADMIN)  # Only admins can manage integrations
        def generate_webhook_api_key():
            """Generate a new webhook API key."""
            # Generate key
            api_key = str(uuid.uuid4())
            
            # Store key
            if not self._add_webhook_api_key(api_key):
                flash('Failed to generate API key', 'danger')
                return redirect(url_for('integrations'))
            
            # Add notification
            self.notification_manager.add_notification(
                user_id=session['user_id'],
                message=f"New webhook API key generated",
                notification_type='success'
            )
            
            flash('New webhook API key generated successfully', 'success')
            session['temp_webhook_api_key'] = api_key  # Store for display
            return redirect(url_for('integrations'))
    
    # User management helper methods
    def _get_user(self, user_id):
        """Get user by ID."""
        if 'users' not in self.config:
            self.config['users'] = {}
        
        return self.config['users'].get(user_id)
    
    def _get_user_by_username(self, username):
        """Get user by username."""
        if 'users' not in self.config:
            self.config['users'] = {}
        
        for user_id, user in self.config['users'].items():
            if user['username'] == username:
                return user
        
        return None
    
    def _verify_user(self, username, password):
        """Verify user credentials."""
        user = self._get_user_by_username(username)
        
        if user and check_password_hash(user['password_hash'], password):
            return user
        
        return None
    
    def _add_user(self, user):
        """Add a new user."""
        if 'users' not in self.config:
            self.config['users'] = {}
        
        self.config['users'][user['id']] = user
        return True
    
    def _update_user(self, user_id, user):
        """Update an existing user."""
        if 'users' not in self.config:
            self.config['users'] = {}
        
        if user_id not in self.config['users']:
            return False
        
        self.config['users'][user_id] = user
        return True
    
    def _delete_user(self, user_id):
        """Delete a user."""
        if 'users' not in self.config:
            return False
        
        if user_id not in self.config['users']:
            return False
        
        del self.config['users'][user_id]
        return True
    
    def _get_all_users(self):
        """Get all users."""
        if 'users' not in self.config:
            self.config['users'] = {}
        
        return list(self.config['users'].values())
    
    def _get_users_by_role(self, role):
        """Get users by role."""
        if 'users' not in self.config:
            self.config['users'] = {}
        
        return [user for user in self.config['users'].values() if user.get('role') == role]
    
    # Integration helper methods
    def _validate_ci_api_key(self, api_key):
        """Validate CI/CD API key."""
        if 'ci_api_keys' not in self.config:
            self.config['ci_api_keys'] = []
        
        return api_key in self.config['ci_api_keys']
    
    def _validate_webhook_api_key(self, api_key):
        """Validate webhook API key."""
        if 'webhook_api_keys' not in self.config:
            self.config['webhook_api_keys'] = []
        
        return api_key in self.config['webhook_api_keys']
    
    def _add_ci_api_key(self, api_key):
        """Add CI/CD API key."""
        if 'ci_api_keys' not in self.config:
            self.config['ci_api_keys'] = []
        
        self.config['ci_api_keys'].append(api_key)
        return True
    
    def _add_webhook_api_key(self, api_key):
        """Add webhook API key."""
        if 'webhook_api_keys' not in self.config:
            self.config['webhook_api_keys'] = []
        
        self.config['webhook_api_keys'].append(api_key)
        return True
    
    def _get_ci_integrations(self):
        """Get CI/CD integrations."""
        if 'ci_api_keys' not in self.config:
            self.config['ci_api_keys'] = []
        
        return [
            {
                'name': f"CI Integration {i+1}",
                'key': key[:8] + '...',  # Show only part of the key for security
                'created_at': datetime.datetime.now().isoformat()  # In a real app, store creation time
            }
            for i, key in enumerate(self.config['ci_api_keys'])
        ]
    
    def _get_webhook_integrations(self):
        """Get webhook integrations."""
        if 'webhook_api_keys' not in self.config:
            self.config['webhook_api_keys'] = []
        
        return [
            {
                'name': f"Webhook Integration {i+1}",
                'key': key[:8] + '...',  # Show only part of the key for security
                'created_at': datetime.datetime.now().isoformat()  # In a real app, store creation time
            }
            for i, key in enumerate(self.config['webhook_api_keys'])
        ]


def main():
    """Command-line entry point for running the web application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulation Validation Framework Web UI")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the server to")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize the web application
        app = WebApp(config_path=args.config)
        
        # Run the application
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    except Exception as e:
        logger.error(f"Error starting web application: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()