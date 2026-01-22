/**
 * Automated Error Reporting System for JavaScript
 * 
 * This module provides functionality to automatically convert runtime errors
 * into GitHub issues for tracking and resolution.
 */

class ErrorReporter {
    constructor(options = {}) {
        this.githubToken = options.githubToken || this._getEnvVar('GITHUB_TOKEN');
        this.githubRepo = options.githubRepo || this._getEnvVar('GITHUB_REPO');
        this.enabled = options.enabled !== false && this.githubToken && this.githubRepo;
        this.includeSystemInfo = options.includeSystemInfo !== false;
        this.autoLabel = options.autoLabel !== false;
        this.apiEndpoint = options.apiEndpoint || '/api/report-error';
        
        // Track reported errors to avoid duplicates
        this.reportedErrors = new Set();
        this._loadReportedErrors();
        
        if (this.enabled) {
            console.log(`Error reporter initialized for ${this.githubRepo}`);
        } else {
            console.log('Error reporter disabled (missing configuration)');
        }
    }
    
    _getEnvVar(name) {
        // Try to get from various sources
        if (typeof process !== 'undefined' && process.env) {
            return process.env[name];
        }
        if (typeof window !== 'undefined' && window.ENV) {
            return window.ENV[name];
        }
        return null;
    }
    
    _loadReportedErrors() {
        try {
            const stored = localStorage.getItem('ipfs_accelerate_reported_errors');
            if (stored) {
                const data = JSON.parse(stored);
                this.reportedErrors = new Set(data.error_hashes || []);
            }
        } catch (error) {
            console.warn('Failed to load reported errors cache:', error);
        }
    }
    
    _saveReportedErrors() {
        try {
            const data = {
                error_hashes: Array.from(this.reportedErrors),
                last_updated: new Date().toISOString()
            };
            localStorage.setItem('ipfs_accelerate_reported_errors', JSON.stringify(data));
        } catch (error) {
            console.warn('Failed to save reported errors cache:', error);
        }
    }
    
    _computeErrorHash(errorInfo) {
        /**
         * Compute a hash for the error to detect duplicates.
         */
        const signatureParts = [
            errorInfo.error_type || '',
            errorInfo.error_message || '',
            errorInfo.source_component || ''
        ];
        
        // Add first few stack frames for uniqueness
        const stackLines = (errorInfo.stack || '').split('\n');
        for (let i = 0; i < Math.min(6, stackLines.length); i++) {
            if (stackLines[i].trim()) {
                signatureParts.push(stackLines[i].trim());
            }
        }
        
        const signature = signatureParts.join('|');
        return this._simpleHash(signature);
    }
    
    _simpleHash(str) {
        /**
         * Simple hash function for JavaScript.
         */
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash).toString(16);
    }
    
    _gatherSystemInfo() {
        /**
         * Gather system information for the error report.
         */
        const info = {
            user_agent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            timestamp: new Date().toISOString(),
            url: window.location.href,
            screen_resolution: `${window.screen.width}x${window.screen.height}`,
            viewport_size: `${window.innerWidth}x${window.innerHeight}`
        };
        
        // Add performance info if available
        if (window.performance && window.performance.memory) {
            info.memory = {
                used: window.performance.memory.usedJSHeapSize,
                total: window.performance.memory.totalJSHeapSize,
                limit: window.performance.memory.jsHeapSizeLimit
            };
        }
        
        return info;
    }
    
    _createIssueBody(errorInfo) {
        /**
         * Create the issue body content.
         */
        const parts = [
            '## Automated Error Report (JavaScript)',
            '',
            `**Error Type:** \`${errorInfo.error_type || 'Unknown'}\``,
            `**Component:** \`${errorInfo.source_component || 'Unknown'}\``,
            `**Timestamp:** ${errorInfo.timestamp || 'N/A'}`,
            '',
            '### Error Message',
            '```',
            errorInfo.error_message || 'No message available',
            '```',
            '',
            '### Stack Trace',
            '```',
            errorInfo.stack || 'No stack trace available',
            '```'
        ];
        
        // Add context if available
        if (errorInfo.context) {
            parts.push(
                '',
                '### Additional Context',
                '```json',
                JSON.stringify(errorInfo.context, null, 2),
                '```'
            );
        }
        
        // Add system info if enabled
        if (this.includeSystemInfo && errorInfo.system_info) {
            parts.push(
                '',
                '### System Information',
                '```json',
                JSON.stringify(errorInfo.system_info, null, 2),
                '```'
            );
        }
        
        parts.push(
            '',
            '---',
            '_This issue was automatically generated by the IPFS Accelerate error reporting system (JavaScript)._'
        );
        
        return parts.join('\n');
    }
    
    _determineLabels(errorInfo) {
        /**
         * Determine appropriate labels for the issue.
         */
        const labels = ['bug', 'automated-report', 'javascript'];
        
        // Add component-specific labels
        const component = (errorInfo.source_component || '').toLowerCase();
        if (component.includes('dashboard')) {
            labels.push('dashboard');
        }
        
        // Add priority labels based on error type
        const errorType = (errorInfo.error_type || '').toLowerCase();
        const criticalErrors = ['syntaxerror', 'referenceerror', 'typeerror'];
        if (criticalErrors.some(critical => errorType === critical)) {
            labels.push('priority:high');
        }
        
        return labels;
    }
    
    async reportError(options = {}) {
        /**
         * Report an error by creating a GitHub issue.
         * 
         * @param {Error} options.error - The error object
         * @param {string} options.errorType - Type of error
         * @param {string} options.errorMessage - Error message
         * @param {string} options.stack - Stack trace
         * @param {string} options.sourceComponent - Component where error occurred
         * @param {object} options.context - Additional context
         * @returns {Promise<string|null>} URL of created issue, or null if not created
         */
        if (!this.enabled) {
            console.debug('Error reporting is disabled');
            return null;
        }
        
        try {
            // Build error information
            const errorInfo = {
                timestamp: new Date().toISOString(),
                source_component: options.sourceComponent || 'dashboard',
                context: options.context || {}
            };
            
            // Extract error details
            if (options.error) {
                errorInfo.error_type = options.error.name || 'Error';
                errorInfo.error_message = options.error.message || String(options.error);
                errorInfo.stack = options.error.stack || '';
            } else {
                errorInfo.error_type = options.errorType || 'UnknownError';
                errorInfo.error_message = options.errorMessage || 'No message provided';
                errorInfo.stack = options.stack || 'No stack trace available';
            }
            
            // Add system info
            if (this.includeSystemInfo) {
                errorInfo.system_info = this._gatherSystemInfo();
            }
            
            // Check if we've already reported this error
            const errorHash = this._computeErrorHash(errorInfo);
            if (this.reportedErrors.has(errorHash)) {
                console.log(`Error ${errorHash} already reported, skipping`);
                return null;
            }
            
            // Create GitHub issue via backend API
            const issueUrl = await this._createGitHubIssue(errorInfo);
            
            if (issueUrl) {
                // Mark error as reported
                this.reportedErrors.add(errorHash);
                this._saveReportedErrors();
                console.log(`Error reported successfully: ${issueUrl}`);
            }
            
            return issueUrl;
            
        } catch (error) {
            console.error('Failed to report error:', error);
            return null;
        }
    }
    
    async _createGitHubIssue(errorInfo) {
        /**
         * Create a GitHub issue for the error via backend API.
         */
        try {
            const title = `[Auto] ${errorInfo.error_type}: ${errorInfo.source_component}`;
            const body = this._createIssueBody(errorInfo);
            const labels = this._determineLabels(errorInfo);
            
            // Send to backend API endpoint
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title,
                    body,
                    labels,
                    error_info: errorInfo
                })
            });
            
            if (!response.ok) {
                throw new Error(`API returned ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            return result.issue_url || null;
            
        } catch (error) {
            console.error('Failed to create GitHub issue:', error);
            return null;
        }
    }
}

// Global error reporter instance
let globalReporter = null;

function getErrorReporter(options = {}) {
    /**
     * Get or create the global error reporter instance.
     */
    if (!globalReporter) {
        globalReporter = new ErrorReporter(options);
    }
    return globalReporter;
}

function reportError(options) {
    /**
     * Convenience function to report an error using the global reporter.
     */
    return getErrorReporter().reportError(options);
}

function installGlobalErrorHandler(sourceComponent = 'dashboard') {
    /**
     * Install global error handlers for window and unhandled rejections.
     */
    const reporter = getErrorReporter();
    
    // Handle uncaught errors
    window.addEventListener('error', (event) => {
        reporter.reportError({
            error: event.error,
            sourceComponent: sourceComponent,
            context: {
                type: 'uncaught_error',
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno
            }
        });
    });
    
    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
        reporter.reportError({
            error: event.reason,
            sourceComponent: sourceComponent,
            context: {
                type: 'unhandled_rejection',
                promise: 'Promise rejected'
            }
        });
    });
    
    console.log(`Installed global error handlers for ${sourceComponent}`);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ErrorReporter,
        getErrorReporter,
        reportError,
        installGlobalErrorHandler
    };
}

// Export for browser
if (typeof window !== 'undefined') {
    window.ErrorReporter = ErrorReporter;
    window.getErrorReporter = getErrorReporter;
    window.reportError = reportError;
    window.installGlobalErrorHandler = installGlobalErrorHandler;
}
