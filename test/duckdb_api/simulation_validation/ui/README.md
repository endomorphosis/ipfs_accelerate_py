# Simulation Validation Web UI

This directory contains the web-based user interface for the Simulation Accuracy and Validation Framework. The web UI provides an interactive interface for managing validation tasks, viewing reports, and monitoring simulation accuracy.

## Key Features

- **User Authentication and Authorization**: Secure login system with role-based access control (Viewer, Analyst, Admin)
- **Job Management**: Schedule and monitor long-running validation, calibration, and reporting tasks
- **Notification System**: Real-time notifications for important system events
- **User Preferences**: Customizable interface settings for each user
- **Reporting Interface**: Interactive report generation and visualization
- **Dashboard Access**: Access to comprehensive monitoring dashboards
- **CI/CD Integration**: Integration with CI/CD systems via API and webhooks

## Components

The web UI is built using Flask and Bootstrap, with a modular architecture:

- **app.py**: Main web application with route definitions and initialization
- **templates/**: HTML templates for the web UI
  - **base.html**: Base template with common layout
  - **index.html**: Home page
  - **login.html**: Authentication page
  - **profile.html**: User profile page
  - **preferences.html**: User preferences page
  - **notifications.html**: Notification management
  - **jobs.html**: Job management
  - **job_details.html**: Detailed job information
  - **validation_*.html**: Validation results and details
  - **integrations.html**: CI/CD and webhook integration management
- **static/**: Static assets (CSS, JavaScript, images)

## Usage

### Running the Web UI

```bash
# Start the web UI on localhost port 5000
python -m duckdb_api.simulation_validation.ui.app --host localhost --port 5000

# Start with a custom configuration file
python -m duckdb_api.simulation_validation.ui.app --config config.json --port 8080

# Start in debug mode
python -m duckdb_api.simulation_validation.ui.app --debug
```

### Configuration

The web UI can be configured using a JSON configuration file:

```json
{
  "title": "Simulation Validation Framework",
  "theme": "light",
  "pagination_size": 20,
  "database": {
    "enabled": true,
    "db_path": "benchmark_db.duckdb"
  },
  "authentication": {
    "enabled": true,
    "users": {
      "admin": "password"
    }
  },
  "jobs": {
    "max_jobs": 5,
    "poll_interval": 10
  },
  "notifications": {
    "enabled": true,
    "max_notifications": 50
  }
}
```

### User Roles and Permissions

The web UI supports three user roles:

1. **Viewer**: Can view results and reports, but cannot run validations or modify settings
2. **Analyst**: Can run validations, calibrations, and generate reports, but cannot manage users
3. **Admin**: Full access to all features, including user management and CI/CD integration

### API Integration

The web UI provides several API endpoints for integration with external systems:

- `/api/status`: Get system status and statistics
- `/api/notifications`: Get user notifications
- `/api/ci-status`: Get CI/CD integration status
- `/api/ci-trigger`: Trigger CI/CD workflow
- `/api/webhooks/validation`: Webhook for receiving validation results

### Dependencies

- **Flask**: Web framework
- **Bootstrap**: UI framework
- **Flask-WTF**: Form handling
- **Werkzeug**: Authentication utilities
- **APScheduler**: Job scheduling

## Development

### Adding New Features

To add new features to the web UI:

1. Add new routes in `app.py`
2. Create/update templates in the `templates/` directory
3. Add static assets in the `static/` directory
4. Update the configuration structure as needed

### Testing

Run the web UI tests with:

```bash
# Run web UI tests
python -m duckdb_api.simulation_validation.test.test_web_ui

# Test specific components
python -m duckdb_api.simulation_validation.test.test_web_ui --test-auth
python -m duckdb_api.simulation_validation.test.test_web_ui --test-jobs
python -m duckdb_api.simulation_validation.test.test_web_ui --test-notifications
```

## Security Considerations

- All form submissions are protected by CSRF tokens
- Passwords are hashed using Werkzeug's password hashing
- API endpoints are protected by API keys
- User sessions expire after inactivity

## Future Enhancements

Planned enhancements for the web UI include:

- **OAuth Integration**: Support for OAuth-based authentication
- **Real-time Updates**: WebSocket-based real-time updates for job status
- **Advanced Visualization**: More interactive visualization options
- **Multi-language Support**: Internationalization support
- **Mobile Optimization**: Enhanced mobile UI