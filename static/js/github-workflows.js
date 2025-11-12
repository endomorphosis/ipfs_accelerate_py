// GitHub Workflows Integration for IPFS Accelerate Dashboard
// Provides workflow monitoring and runner management features
// Uses MCP Server JavaScript SDK for communication

class GitHubWorkflowsManager {
    constructor(mcpClient) {
        this.workflows = {};
        this.runners = [];
        this.cacheStats = null;
        this.rateLimit = null;
        this.updateInterval = null;
        // Use MCP client if provided, otherwise create new one
        this.mcp = mcpClient || (typeof MCPClient !== 'undefined' ? new MCPClient() : null);
        
        if (!this.mcp) {
            console.warn('[GitHub Workflows] MCP Client not available, falling back to direct API');
        }
        
        console.log('[GitHub Workflows] Manager initialized with MCP SDK');
    }

    // Safe HTML escaping - use global escapeHtml if available, otherwise use fallback
    _escapeHtml(text) {
        if (typeof escapeHtml !== 'undefined') {
            return escapeHtml(text);
        }
        // Fallback if escapeHtml not loaded yet
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, function(m) { return map[m]; });
    }

    // Initialize the workflows manager
    async initialize() {
        console.log('[GitHub Workflows] Initializing with MCP SDK...');
        await Promise.all([
            this.fetchWorkflows(),
            this.fetchRunners(),
            this.fetchIssues(),
            this.fetchPullRequests(),
            this.fetchCacheStats(),
            this.fetchRateLimit()
        ]);
        this.startAutoRefresh();
        console.log('[GitHub Workflows] Initialization complete');
    }

    // Fetch workflows from the server using MCP tools
    async fetchWorkflows() {
        try {
            if (this.mcp) {
                // Use MCP SDK to call gh_create_workflow_queues tool
                console.log('[GitHub Workflows] Calling gh_create_workflow_queues via MCP SDK...');
                const result = await this.mcp.request('tools/call', {
                    name: 'gh_create_workflow_queues',
                    arguments: {
                        since_days: 1
                    }
                });
                
                console.log('[GitHub Workflows] Received workflow data:', result);
                
                if (result && result.queues) {
                    this.workflows = result.queues;
                    this.workflowsError = null;
                    this.renderWorkflows();
                    console.log(`[GitHub Workflows] Loaded ${Object.keys(this.workflows).length} repositories`);
                } else if (result && result.error) {
                    console.error('[GitHub Workflows] MCP tool returned error:', result.error);
                    this.workflowsError = result.error;
                    this.renderWorkflows();
                } else {
                    console.error('[GitHub Workflows] Invalid response from MCP tool');
                    this.workflowsError = 'Invalid response from server';
                    this.renderWorkflows();
                }
            } else {
                // Fallback to direct API
                console.warn('[GitHub Workflows] Falling back to direct API call');
                const response = await fetch('/api/github/workflows');
                if (response.ok) {
                    this.workflows = await response.json();
                    this.workflowsError = null;
                    this.renderWorkflows();
                } else {
                    console.error('[GitHub Workflows] Failed to fetch workflows:', response.statusText);
                    this.workflowsError = `Failed to fetch: ${response.statusText}`;
                    this.renderWorkflows();
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching workflows:', error);
            this.workflowsError = error.message || 'Unknown error occurred';
            this.renderWorkflows();
        }
    }

    // Fetch runners from the server using MCP tools
    async fetchRunners() {
        try {
            if (this.mcp) {
                // Use MCP SDK to call gh_list_runners tool
                console.log('[GitHub Workflows] Calling gh_list_runners via MCP SDK...');
                const result = await this.mcp.request('tools/call', {
                    name: 'gh_list_runners',
                    arguments: {}
                });
                
                console.log('[GitHub Workflows] Received runner data:', result);
                
                if (result && result.runners) {
                    this.runners = result.runners;
                    this.runnersError = null;
                    this.renderRunners();
                    console.log(`[GitHub Workflows] Loaded ${this.runners.length} runners`);
                } else if (result && result.error) {
                    console.error('[GitHub Workflows] MCP tool returned error:', result.error);
                    this.runnersError = result.error;
                    this.renderRunners();
                } else {
                    console.error('[GitHub Workflows] Invalid response from MCP tool');
                    this.runnersError = 'Invalid response from server';
                    this.renderRunners();
                }
            } else {
                // Fallback to direct API
                console.warn('[GitHub Workflows] Falling back to direct API call');
                const response = await fetch('/api/github/runners');
                if (response.ok) {
                    this.runners = await response.json();
                    this.runnersError = null;
                    this.renderRunners();
                } else {
                    console.error('[GitHub Workflows] Failed to fetch runners:', response.statusText);
                    this.runnersError = `Failed to fetch: ${response.statusText}`;
                    this.renderRunners();
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching runners:', error);
            this.runnersError = error.message || 'Unknown error occurred';
            this.renderRunners();
        }
    }

    // Fetch issues from all repositories using MCP tools
    async fetchIssues() {
        try {
            if (this.mcp) {
                // Use MCP SDK to call gh_list_all_issues tool
                console.log('[GitHub Workflows] Calling gh_list_all_issues via MCP SDK...');
                const result = await this.mcp.request('tools/call', {
                    name: 'gh_list_all_issues',
                    arguments: {
                        state: 'open',
                        limit_per_repo: 20
                    }
                });
                
                console.log('[GitHub Workflows] Received issues data:', result);
                
                if (result && result.issues) {
                    this.issues = result.issues;
                    this.renderIssues();
                    console.log(`[GitHub Workflows] Loaded ${result.total_issues} issues from ${result.repo_count} repos`);
                } else if (result && result.error) {
                    console.error('[GitHub Workflows] MCP tool returned error:', result.error);
                    this.issuesError = result.error;
                    this.renderIssues();
                } else {
                    console.error('[GitHub Workflows] Invalid response from MCP tool');
                    this.issuesError = 'Invalid response from server';
                    this.renderIssues();
                }
            } else {
                // Fallback to direct API
                console.warn('[GitHub Workflows] Falling back to direct API call');
                const response = await fetch('/api/github/issues');
                if (response.ok) {
                    this.issues = await response.json();
                    this.renderIssues();
                } else {
                    console.error('[GitHub Workflows] Failed to fetch issues:', response.statusText);
                    this.issuesError = `Failed to fetch: ${response.statusText}`;
                    this.renderIssues();
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching issues:', error);
            this.issuesError = error.message || 'Unknown error occurred';
            this.renderIssues();
        }
    }

    // Fetch pull requests from all repositories using MCP tools
    async fetchPullRequests() {
        try {
            if (this.mcp) {
                // Use MCP SDK to call gh_list_all_pull_requests tool
                console.log('[GitHub Workflows] Calling gh_list_all_pull_requests via MCP SDK...');
                const result = await this.mcp.request('tools/call', {
                    name: 'gh_list_all_pull_requests',
                    arguments: {
                        state: 'open',
                        limit_per_repo: 20
                    }
                });
                
                console.log('[GitHub Workflows] Received pull requests data:', result);
                
                if (result && result.pull_requests) {
                    this.pullRequests = result.pull_requests;
                    this.renderPullRequests();
                    console.log(`[GitHub Workflows] Loaded ${result.total_prs} PRs from ${result.repo_count} repos`);
                } else if (result && result.error) {
                    console.error('[GitHub Workflows] MCP tool returned error:', result.error);
                    this.pullRequestsError = result.error;
                    this.renderPullRequests();
                } else {
                    console.error('[GitHub Workflows] Invalid response from MCP tool');
                    this.pullRequestsError = 'Invalid response from server';
                    this.renderPullRequests();
                }
            } else {
                // Fallback to direct API
                console.warn('[GitHub Workflows] Falling back to direct API call');
                const response = await fetch('/api/github/pull-requests');
                if (response.ok) {
                    this.pullRequests = await response.json();
                    this.renderPullRequests();
                } else {
                    console.error('[GitHub Workflows] Failed to fetch pull requests:', response.statusText);
                    this.pullRequestsError = `Failed to fetch: ${response.statusText}`;
                    this.renderPullRequests();
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching pull requests:', error);
            this.pullRequestsError = error.message || 'Unknown error occurred';
            this.renderPullRequests();
        }
    }

    // Fetch cache statistics using MCP SDK
    async fetchCacheStats() {
        try {
            if (this.mcp) {
                console.log('[GitHub Workflows] Calling gh_get_cache_stats via MCP SDK...');
                const result = await this.mcp.request('tools/call', {
                    name: 'gh_get_cache_stats',
                    arguments: {}
                });
                
                console.log('[GitHub Workflows] Received cache stats:', result);
                
                if (result && !result.error) {
                    this.cacheStats = result;
                    this.renderCacheStats();
                } else if (result && result.error) {
                    console.error('[GitHub Workflows] MCP tool returned error:', result.error);
                } else {
                    console.error('[GitHub Workflows] Invalid response from MCP tool');
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching cache stats:', error);
        }
    }

    // Fetch GitHub API rate limit using MCP SDK
    async fetchRateLimit() {
        try {
            if (this.mcp) {
                console.log('[GitHub Workflows] Calling gh_get_rate_limit via MCP SDK...');
                const result = await this.mcp.request('tools/call', {
                    name: 'gh_get_rate_limit',
                    arguments: {}
                });
                
                console.log('[GitHub Workflows] Received rate limit data:', result);
                
                if (result && result.rate_limit) {
                    this.rateLimit = result.rate_limit;
                    this.renderRateLimit();
                } else if (result && result.error) {
                    console.error('[GitHub Workflows] MCP tool returned error:', result.error);
                } else {
                    console.error('[GitHub Workflows] Invalid response from MCP tool');
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching rate limit:', error);
        }
    }

    // Render workflows in the dashboard
    renderWorkflows() {
        const container = document.getElementById('github-workflows-container');
        if (!container) return;

        // Check if there's an error to display
        if (this.workflowsError) {
            container.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <p style="color: #ef4444; margin-bottom: 10px;">⚠️ Error loading workflows</p>
                    <p style="color: #6b7280; font-size: 14px;">${this._escapeHtml(this.workflowsError)}</p>
                    <p style="color: #6b7280; font-size: 12px; margin-top: 10px;">Please check GitHub authentication and permissions.</p>
                </div>
            `;
            return;
        }

        let html = '<div class="workflows-grid">';

        if (!this.workflows || Object.keys(this.workflows).length === 0) {
            html += '<p class="empty-state">No active workflows found</p>';
        } else {
            for (const [repo, workflows] of Object.entries(this.workflows)) {
                const running = workflows.filter(w => w.status === 'in_progress').length;
                const failed = workflows.filter(w => w.conclusion === 'failure').length;
                const success = workflows.filter(w => w.conclusion === 'success').length;

                html += `
                    <div class="workflow-card">
                        <div class="workflow-header">
                            <h4>${repo}</h4>
                            <span class="workflow-count">${workflows.length} workflows</span>
                        </div>
                        <div class="workflow-stats">
                            <div class="stat">
                                <span class="stat-icon">⚡</span>
                                <span class="stat-value">${running}</span>
                                <span class="stat-label">Running</span>
                            </div>
                            <div class="stat">
                                <span class="stat-icon">❌</span>
                                <span class="stat-value">${failed}</span>
                                <span class="stat-label">Failed</span>
                            </div>
                            <div class="stat">
                                <span class="stat-icon">✅</span>
                                <span class="stat-value">${success}</span>
                                <span class="stat-label">Success</span>
                            </div>
                        </div>
                        <div class="workflow-actions">
                            <button class="btn btn-sm btn-primary" onclick="githubManager.viewWorkflowDetails('${repo}')">
                                View Details
                            </button>
                            <button class="btn btn-sm btn-secondary" onclick="githubManager.provisionRunner('${repo}')">
                                Provision Runner
                            </button>
                        </div>
                    </div>
                `;
            }
        }

        html += '</div>';
        container.innerHTML = html;
    }

    // Render runners in the dashboard
    renderRunners() {
        const container = document.getElementById('github-runners-container');
        if (!container) return;

        // Check if there's an error to display
        if (this.runnersError) {
            container.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                    <p style="color: #ef4444; margin-bottom: 10px;">⚠️ Error loading runners</p>
                    <p style="color: #6b7280; font-size: 14px;">${this._escapeHtml(this.runnersError)}</p>
                    <p style="color: #6b7280; font-size: 12px; margin-top: 10px;">Please check GitHub authentication and permissions.</p>
                </div>
            `;
            return;
        }

        let html = '<div class="runners-list">';

        if (!this.runners || this.runners.length === 0) {
            html += '<p class="empty-state">No self-hosted runners configured</p>';
        } else {
            for (const runner of this.runners) {
                const statusClass = runner.status === 'online' ? 'status-online' : 'status-offline';
                html += `
                    <div class="runner-item">
                        <div class="runner-info">
                            <div class="runner-name">
                                <span class="${statusClass}"></span>
                                ${runner.name}
                            </div>
                            <div class="runner-labels">
                                ${runner.labels.map(l => `<span class="label">${l.name}</span>`).join('')}
                            </div>
                        </div>
                        <div class="runner-status">
                            <span class="runner-os">${runner.os}</span>
                            <span class="runner-busy">${runner.busy ? 'Busy' : 'Idle'}</span>
                        </div>
                    </div>
                `;
            }
        }

        html += '</div>';
        container.innerHTML = html;
    }

    // Render issues in the dashboard
    renderIssues() {
        const container = document.getElementById('github-issues-container');
        if (!container) return;

        let html = '<div class="issues-list">';
        let totalIssues = 0;

        // Check if there's an error to display
        if (this.issuesError) {
            html += `
                <div style="padding: 20px; text-align: center;">
                    <p style="color: #ef4444; margin-bottom: 10px;">⚠️ Error loading issues</p>
                    <p style="color: #6b7280; font-size: 14px;">${this._escapeHtml(this.issuesError)}</p>
                    <p style="color: #6b7280; font-size: 12px; margin-top: 10px;">Please check GitHub authentication and permissions.</p>
                </div>
            `;
        } else if (!this.issues || Object.keys(this.issues).length === 0) {
            html += '<p class="empty-state">No open issues found</p>';
        } else {
            for (const [repo, issues] of Object.entries(this.issues)) {
                totalIssues += issues.length;
                html += `<div class="repo-section">`;
                html += `<h5 class="repo-title">${repo} (${issues.length})</h5>`;
                
                for (const issue of issues.slice(0, 5)) { // Show max 5 issues per repo
                    const createdDate = new Date(issue.createdAt).toLocaleDateString();
                    html += `
                        <div class="issue-item">
                            <div class="issue-info">
                                <a href="${issue.url}" target="_blank" class="issue-title">
                                    #${issue.number}: ${this._escapeHtml(issue.title)}
                                </a>
                                <div class="issue-meta">
                                    <span class="issue-author">by ${issue.author.login}</span>
                                    <span class="issue-date">${createdDate}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                if (issues.length > 5) {
                    html += `<p class="more-items">...and ${issues.length - 5} more</p>`;
                }
                html += `</div>`;
            }
        }

        if (!this.issuesError) {
            html += `<div class="summary">Total: ${totalIssues} open issues</div>`;
        }
        html += '</div>';
        container.innerHTML = html;
    }

    // Render pull requests in the dashboard  
    renderPullRequests() {
        const container = document.getElementById('github-prs-container');
        if (!container) return;

        let html = '<div class="prs-list">';
        let totalPRs = 0;

        // Check if there's an error to display
        if (this.pullRequestsError) {
            html += `
                <div style="padding: 20px; text-align: center;">
                    <p style="color: #ef4444; margin-bottom: 10px;">⚠️ Error loading pull requests</p>
                    <p style="color: #6b7280; font-size: 14px;">${this._escapeHtml(this.pullRequestsError)}</p>
                    <p style="color: #6b7280; font-size: 12px; margin-top: 10px;">Please check GitHub authentication and permissions.</p>
                </div>
            `;
        } else if (!this.pullRequests || Object.keys(this.pullRequests).length === 0) {
            html += '<p class="empty-state">No open pull requests found</p>';
        } else {
            for (const [repo, prs] of Object.entries(this.pullRequests)) {
                totalPRs += prs.length;
                html += `<div class="repo-section">`;
                html += `<h5 class="repo-title">${repo} (${prs.length})</h5>`;
                
                for (const pr of prs.slice(0, 5)) { // Show max 5 PRs per repo
                    const createdDate = new Date(pr.createdAt).toLocaleDateString();
                    html += `
                        <div class="pr-item">
                            <div class="pr-info">
                                <a href="${pr.url}" target="_blank" class="pr-title">
                                    #${pr.number}: ${this._escapeHtml(pr.title)}
                                </a>
                                <div class="pr-meta">
                                    <span class="pr-author">by ${pr.author.login}</span>
                                    <span class="pr-date">${createdDate}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                if (prs.length > 5) {
                    html += `<p class="more-items">...and ${prs.length - 5} more</p>`;
                }
                html += `</div>`;
            }
        }

        if (!this.pullRequestsError) {
            html += `<div class="summary">Total: ${totalPRs} open pull requests</div>`;
        }
        html += '</div>';
        container.innerHTML = html;
    }

    // View workflow details
    async viewWorkflowDetails(repo) {
        const workflows = this.workflows[repo];
        if (!workflows) return;

        let html = `
            <div class="modal-overlay" onclick="githubManager.closeModal()">
                <div class="modal-content" onclick="event.stopPropagation()">
                    <div class="modal-header">
                        <h3>Workflows for ${repo}</h3>
                        <button class="close-btn" onclick="githubManager.closeModal()">×</button>
                    </div>
                    <div class="modal-body">
                        <div class="workflows-list">
        `;

        for (const workflow of workflows) {
            const statusClass = workflow.status === 'in_progress' ? 'status-running' : 
                              workflow.conclusion === 'success' ? 'status-success' : 
                              workflow.conclusion === 'failure' ? 'status-failure' : 'status-unknown';
            
            html += `
                <div class="workflow-item">
                    <div class="workflow-item-header">
                        <span class="workflow-name">${workflow.workflowName || workflow.name}</span>
                        <span class="badge ${statusClass}">${workflow.status}/${workflow.conclusion || 'pending'}</span>
                    </div>
                    <div class="workflow-item-details">
                        <span>ID: #${workflow.databaseId}</span>
                        <span>Branch: ${workflow.headBranch || 'N/A'}</span>
                        <span>Event: ${workflow.event || 'N/A'}</span>
                    </div>
                    <div class="workflow-item-times">
                        <span>Created: ${new Date(workflow.createdAt).toLocaleString()}</span>
                        <span>Updated: ${new Date(workflow.updatedAt).toLocaleString()}</span>
                    </div>
                    <div class="workflow-item-actions" style="margin-top: 10px; display: flex; gap: 8px;">
                        <button class="btn btn-sm btn-primary" onclick="githubManager.viewWorkflowLogs('${repo}', '${workflow.databaseId}')" style="font-size: 12px; padding: 4px 12px;">📄 View Logs</button>
                    </div>
                </div>
            `;
        }

        html += `
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="githubManager.closeModal()">Close</button>
                        <button class="btn btn-primary" onclick="githubManager.refreshWorkflows()">Refresh</button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', html);
    }

    // View workflow logs
    async viewWorkflowLogs(repo, runId) {
        try {
            console.log('[GitHub Workflows] Fetching logs for:', repo, runId);
            showToast('Fetching workflow logs...', 'info');
            
            // First, get workflow details to get job IDs
            const detailsResult = await this.mcp.request('tools/call', {
                name: 'gh_get_workflow_details',
                arguments: {
                    repo: repo,
                    run_id: String(runId),
                    include_jobs: true
                }
            });
            
            console.log('[GitHub Workflows] Workflow details:', detailsResult);
            
            if (detailsResult && detailsResult.jobs && detailsResult.jobs.length > 0) {
                // Display jobs and allow user to select which one to view logs for
                let html = `
                    <div class="modal-overlay" onclick="githubManager.closeModal()">
                        <div class="modal-content" onclick="event.stopPropagation()" style="max-width: 900px; max-height: 90vh; overflow-y: auto;">
                            <div class="modal-header">
                                <h3>Workflow Logs - ${repo} #${runId}</h3>
                                <button class="close-btn" onclick="githubManager.closeModal()">×</button>
                            </div>
                            <div class="modal-body">
                                <div style="margin-bottom: 20px;">
                                    <h4 style="margin-bottom: 10px;">Select a job to view logs:</h4>
                                    <div class="jobs-list" style="display: grid; gap: 10px;">
                `;
                
                for (const job of detailsResult.jobs) {
                    const statusClass = job.status === 'in_progress' ? 'status-running' : 
                                      job.conclusion === 'success' ? 'status-success' : 
                                      job.conclusion === 'failure' ? 'status-failure' : 'status-unknown';
                    const statusIcon = job.conclusion === 'success' ? '✅' : 
                                     job.conclusion === 'failure' ? '❌' : 
                                     job.status === 'in_progress' ? '⏳' : '⚪';
                    
                    html += `
                        <div class="job-item" style="border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; background: white;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="font-weight: 600; margin-bottom: 4px;">${statusIcon} ${this._escapeHtml(job.name)}</div>
                                    <div style="font-size: 12px; color: #6b7280;">
                                        Status: ${job.status} | Conclusion: ${job.conclusion || 'N/A'}
                                    </div>
                                </div>
                                <button class="btn btn-sm btn-primary" onclick="githubManager.fetchJobLogs('${repo}', '${runId}', '${job.id}', '${this._escapeHtml(job.name)}')" style="font-size: 12px;">
                                    View Logs
                                </button>
                            </div>
                        </div>
                    `;
                }
                
                html += `
                                    </div>
                                </div>
                                <div id="job-logs-container" style="display: none;">
                                    <h4 style="margin-bottom: 10px;" id="job-logs-title">Job Logs</h4>
                                    <pre id="job-logs-content" style="background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 12px; line-height: 1.4; max-height: 500px; overflow-y: auto;"></pre>
                                </div>
                            </div>
                            <div class="modal-footer">
                                <button class="btn btn-secondary" onclick="githubManager.closeModal()">Close</button>
                            </div>
                        </div>
                    </div>
                `;
                
                document.body.insertAdjacentHTML('beforeend', html);
            } else {
                showToast('No jobs found for this workflow run', 'warning');
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching workflow logs:', error);
            showToast('Error fetching workflow logs', 'error');
        }
    }

    // Fetch specific job logs
    async fetchJobLogs(repo, runId, jobId, jobName) {
        try {
            console.log('[GitHub Workflows] Fetching job logs:', repo, runId, jobId);
            showToast('Loading job logs...', 'info');
            
            const result = await this.mcp.request('tools/call', {
                name: 'gh_get_workflow_logs',
                arguments: {
                    repo: repo,
                    run_id: String(runId),
                    job_id: String(jobId),
                    tail_lines: 1000
                }
            });
            
            console.log('[GitHub Workflows] Job logs result:', result);
            
            const logsContainer = document.getElementById('job-logs-container');
            const logsTitle = document.getElementById('job-logs-title');
            const logsContent = document.getElementById('job-logs-content');
            
            if (logsContainer && logsTitle && logsContent) {
                logsContainer.style.display = 'block';
                logsTitle.textContent = `Logs for: ${jobName}`;
                
                if (result && result.logs) {
                    logsContent.textContent = result.logs;
                    showToast('Logs loaded successfully', 'success');
                    // Scroll to logs
                    logsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else if (result && result.error) {
                    logsContent.textContent = `Error: ${result.error}`;
                    showToast('Failed to load logs', 'error');
                } else {
                    logsContent.textContent = 'No logs available';
                    showToast('No logs available', 'warning');
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching job logs:', error);
            showToast('Error fetching job logs', 'error');
        }
    }

    // Provision runner for a repository using MCP SDK
    async provisionRunner(repo) {
        try {
            if (this.mcp) {
                // Use MCP SDK to call gh_provision_runners tool
                console.log('[GitHub Workflows] Calling gh_provision_runners via MCP SDK for:', repo);
                const result = await this.mcp.request('tools/call', {
                    name: 'gh_provision_runners',
                    arguments: {
                        since_days: 1,
                        max_runners: 1
                    }
                });
                
                console.log('[GitHub Workflows] Provisioning result:', result);
                
                if (result && result.success) {
                    showToast(`Runner provisioned for ${repo}: ${result.runners_provisioned} runner(s)`, 'success');
                    await this.fetchRunners();
                } else if (result && result.error) {
                    showToast(`Failed to provision runner: ${result.error}`, 'error');
                } else {
                    showToast('Failed to provision runner: Unknown error', 'error');
                }
            } else {
                // Fallback to direct API
                console.warn('[GitHub Workflows] Falling back to direct API call');
                const response = await fetch('/api/github/provision-runner', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ repo })
                });

                if (response.ok) {
                    const result = await response.json();
                    if (result.success) {
                        showToast(`Runner token generated for ${repo}`, 'success');
                        await this.fetchRunners();
                    } else {
                        showToast(`Failed to provision runner: ${result.error}`, 'error');
                    }
                } else {
                    showToast('Failed to provision runner', 'error');
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error provisioning runner:', error);
            showToast('Error provisioning runner', 'error');
        }
    }

    // Close modal
    closeModal() {
        const modal = document.querySelector('.modal-overlay');
        if (modal) {
            modal.remove();
        }
    }

    // Refresh workflows
    async refreshWorkflows() {
        await this.fetchWorkflows();
        showToast('Workflows refreshed', 'success');
    }

    // Refresh runners
    async refreshRunners() {
        await this.fetchRunners();
        showToast('Runners refreshed', 'success');
    }

    // View API Call Log
    async viewAPICallLog() {
        try {
            console.log('[GitHub Workflows] Fetching API call log...');
            showToast('Loading API call log...', 'info');
            
            const result = await this.mcp.request('tools/call', {
                name: 'gh_get_api_call_log',
                arguments: {
                    limit: 50
                }
            });
            
            console.log('[GitHub Workflows] API call log:', result);
            
            if (result && result.api_calls) {
                this.displayAPICallLog(result);
            } else {
                showToast('No API call log available', 'warning');
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching API call log:', error);
            showToast('Error fetching API call log', 'error');
        }
    }

    // Display API Call Log Modal
    displayAPICallLog(logData) {
        const calls = logData.api_calls || [];
        const summary = logData.summary || {};
        const totalStats = logData.total_stats || {};
        
        let html = `
            <div class="modal-overlay" onclick="githubManager.closeModal()">
                <div class="modal-content" onclick="event.stopPropagation()" style="max-width: 900px; max-height: 90vh; overflow-y: auto;">
                    <div class="modal-header">
                        <h3>📋 GitHub API Call Log</h3>
                        <button class="close-btn" onclick="githubManager.closeModal()">×</button>
                    </div>
                    <div class="modal-body">
                        <div style="margin-bottom: 20px; padding: 15px; background: #f3f4f6; border-radius: 8px;">
                            <h4 style="margin: 0 0 10px 0;">📊 Total Statistics</h4>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                                <div><strong>🔹 REST API Calls:</strong> ${totalStats.rest_total || 0}</div>
                                <div><strong>🔸 GraphQL API Calls:</strong> ${totalStats.graphql_total || 0}</div>
                                <div><strong>🔍 CodeQL API Calls:</strong> ${totalStats.code_scanning_total || 0}</div>
                                <div><strong>💾 Cache Hits:</strong> ${totalStats.cache_hits || 0}</div>
                                <div><strong>❌ Cache Misses:</strong> ${totalStats.cache_misses || 0}</div>
                                <div><strong>🎯 Hit Rate:</strong> ${(totalStats.hit_rate * 100).toFixed(1)}%</div>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 15px;">
                            <h4 style="margin: 0 0 10px 0;">Recent API Calls (Last ${calls.length})</h4>
                            <div style="font-size: 12px; color: #6b7280; margin-bottom: 10px;">
                                Showing: ${summary.rest || 0} REST, ${summary.graphql || 0} GraphQL, ${summary.code_scanning || 0} CodeQL
                            </div>
                        </div>
                        
                        <div style="max-height: 400px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 6px;">
                            <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                                <thead style="position: sticky; top: 0; background: #f9fafb; border-bottom: 2px solid #e5e7eb;">
                                    <tr>
                                        <th style="padding: 8px; text-align: left; font-weight: 600;">Time</th>
                                        <th style="padding: 8px; text-align: left; font-weight: 600;">API Type</th>
                                        <th style="padding: 8px; text-align: left; font-weight: 600;">Operation</th>
                                        <th style="padding: 8px; text-align: right; font-weight: 600;">#</th>
                                    </tr>
                                </thead>
                                <tbody>
        `;
        
        // Display calls in reverse order (newest first)
        for (let i = calls.length - 1; i >= 0; i--) {
            const call = calls[i];
            const time = new Date(call.timestamp * 1000).toLocaleTimeString();
            const icon = call.api_type === 'graphql' ? '🔸' : call.api_type === 'code_scanning' ? '🔍' : '🔹';
            const bgColor = i % 2 === 0 ? '#ffffff' : '#f9fafb';
            
            html += `
                <tr style="background: ${bgColor}; border-bottom: 1px solid #e5e7eb;">
                    <td style="padding: 8px;">${time}</td>
                    <td style="padding: 8px;">${icon} ${call.api_type}</td>
                    <td style="padding: 8px; font-family: monospace; font-size: 11px;">${this._escapeHtml(call.operation)}</td>
                    <td style="padding: 8px; text-align: right; color: #6b7280;">${call.count}</td>
                </tr>
            `;
        }
        
        html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="githubManager.closeModal()">Close</button>
                        <button class="btn btn-primary" onclick="githubManager.viewAPICallLog()">🔄 Refresh</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', html);
    }

    // Start auto-refresh
    startAutoRefresh(interval = 30000) {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.updateInterval = setInterval(async () => {
            console.log('[GitHub Workflows] Auto-refreshing data via MCP SDK...');
            await Promise.all([
                this.fetchWorkflows(),
                this.fetchRunners(),
                this.fetchIssues(),
                this.fetchPullRequests(),
                this.fetchCacheStats(),
                this.fetchRateLimit()
            ]);
        }, interval);
        
        console.log(`[GitHub Workflows] Auto-refresh enabled (${interval}ms interval)`);
    }

    // Stop auto-refresh
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    // Get workflow statistics
    getStatistics() {
        const stats = {
            totalWorkflows: 0,
            runningWorkflows: 0,
            failedWorkflows: 0,
            successfulWorkflows: 0,
            repositories: Object.keys(this.workflows).length,
            totalRunners: this.runners.length,
            onlineRunners: this.runners.filter(r => r.status === 'online').length,
            busyRunners: this.runners.filter(r => r.busy).length
        };

        for (const workflows of Object.values(this.workflows)) {
            stats.totalWorkflows += workflows.length;
            stats.runningWorkflows += workflows.filter(w => w.status === 'in_progress').length;
            stats.failedWorkflows += workflows.filter(w => w.conclusion === 'failure').length;
            stats.successfulWorkflows += workflows.filter(w => w.conclusion === 'success').length;
        }

        return stats;
    }

    // Render statistics in the overview
    renderStatistics() {
        const stats = this.getStatistics();
        const container = document.getElementById('github-stats-overview');
        if (!container) return;

        container.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">📦</div>
                    <div class="stat-value">${stats.repositories}</div>
                    <div class="stat-label">Repositories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">⚡</div>
                    <div class="stat-value">${stats.runningWorkflows}</div>
                    <div class="stat-label">Running</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">❌</div>
                    <div class="stat-value">${stats.failedWorkflows}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🖥️</div>
                    <div class="stat-value">${stats.onlineRunners}/${stats.totalRunners}</div>
                    <div class="stat-label">Runners</div>
                </div>
            </div>
        `;
    }

    // Render cache statistics
    renderCacheStats() {
        if (!this.cacheStats) return;
        
        const container = document.getElementById('github-cache-stats');
        if (!container) {
            console.warn('[GitHub Workflows] Cache stats container not found');
            return;
        }

        const hitRate = (this.cacheStats.hit_rate * 100).toFixed(1);
        const p2pStatus = this.cacheStats.p2p_enabled ? '✓ Enabled' : '✗ Disabled';
        const multiformatsStatus = this.cacheStats.content_addressing_available ? '✓ Available' : '✗ Unavailable';
        
        container.innerHTML = `
            <div class="cache-stats-grid">
                <div class="cache-stat-card">
                    <div class="stat-icon">💾</div>
                    <div class="stat-value">${this.cacheStats.cache_size}/${this.cacheStats.max_cache_size}</div>
                    <div class="stat-label">Cache Entries</div>
                </div>
                <div class="cache-stat-card">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-value">${hitRate}%</div>
                    <div class="stat-label">Hit Rate</div>
                </div>
                <div class="cache-stat-card">
                    <div class="stat-icon">💰</div>
                    <div class="stat-value">${this.cacheStats.api_calls_saved || 0}</div>
                    <div class="stat-label">API Calls Saved</div>
                </div>
                <div class="cache-stat-card">
                    <div class="stat-icon">🔗</div>
                    <div class="stat-value">${this.cacheStats.p2p_peers?.connected || 0}</div>
                    <div class="stat-label">P2P Peers</div>
                </div>
            </div>
            <div class="cache-details">
                <div class="cache-detail-row">
                    <span class="detail-label">P2P Cache Sharing:</span>
                    <span class="detail-value">${p2pStatus}</span>
                </div>
                <div class="cache-detail-row">
                    <span class="detail-label">Content Addressing:</span>
                    <span class="detail-value">${multiformatsStatus}</span>
                </div>
                <div class="cache-detail-row">
                    <span class="detail-label">Total Requests:</span>
                    <span class="detail-value">${this.cacheStats.total_requests || 0}</span>
                </div>
                <div class="cache-detail-row">
                    <span class="detail-label">Cache Hits (Local):</span>
                    <span class="detail-value">${this.cacheStats.local_hits || 0}</span>
                </div>
                <div class="cache-detail-row">
                    <span class="detail-label">Cache Hits (P2P):</span>
                    <span class="detail-value">${this.cacheStats.peer_hits || 0}</span>
                </div>
                <div class="cache-detail-section" style="margin-top: 15px; padding-top: 15px; border-top: 2px solid #e5e7eb;">
                    <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #1f2937;">📊 API Statistics</h4>
                    <div class="cache-detail-row">
                        <span class="detail-label">🔹 REST API Calls:</span>
                        <span class="detail-value">${this.cacheStats.api_calls_made || 0}</span>
                    </div>
                    <div class="cache-detail-row">
                        <span class="detail-label">🔸 GraphQL API Calls:</span>
                        <span class="detail-value">${this.cacheStats.graphql_api_calls_made || 0}</span>
                    </div>
                    <div class="cache-detail-row">
                        <span class="detail-label">🔸 GraphQL Cache Hits:</span>
                        <span class="detail-value">${this.cacheStats.graphql_cache_hits || 0}</span>
                    </div>
                    <div class="cache-detail-row">
                        <span class="detail-label">🔍 CodeQL API Calls:</span>
                        <span class="detail-value">${this.cacheStats.code_scanning_api_calls || 0}</span>
                    </div>
                    <div style="margin-top: 10px;">
                        <button class="btn btn-sm btn-info" onclick="githubManager.viewAPICallLog()" style="font-size: 12px; padding: 6px 12px;">
                            📋 View API Call Log
                        </button>
                    </div>
                </div>
                ${this.cacheStats.aggregate ? this._renderAggregateStats(this.cacheStats.aggregate) : ''}
            </div>
        `;
    }

    // Render rate limit information
    renderRateLimit() {
        if (!this.rateLimit) return;
        
        const container = document.getElementById('github-rate-limit');
        if (!container) {
            console.warn('[GitHub Workflows] Rate limit container not found');
            return;
        }

        // Support both flat structure and nested resources.core structure
        const core = this.rateLimit.resources?.core || this.rateLimit;
        const graphql = this.rateLimit.resources?.graphql || null;
        const codeScanning = this.rateLimit.resources?.code_scanning || null;
        
        const restRemaining = core.remaining || 0;
        const restLimit = core.limit || 5000;
        const restResetDate = core.reset ? new Date(core.reset * 1000).toLocaleString() : 'Unknown';
        const restUsagePercent = ((restLimit - restRemaining) / restLimit * 100).toFixed(1);
        
        const statusClass = restRemaining > 1000 ? 'status-good' : restRemaining > 100 ? 'status-warning' : 'status-critical';
        
        let graphqlSection = '';
        let codeScanningSection = '';
        if (graphql) {
            const graphqlRemaining = graphql.remaining || 0;
            const graphqlLimit = graphql.limit || 5000;
            const graphqlResetDate = graphql.reset ? new Date(graphql.reset * 1000).toLocaleString() : 'Unknown';
            const graphqlUsagePercent = ((graphqlLimit - graphqlRemaining) / graphqlLimit * 100).toFixed(1);
            const graphqlStatusClass = graphqlRemaining > 1000 ? 'status-good' : graphqlRemaining > 100 ? 'status-warning' : 'status-critical';
            
            graphqlSection = `
                <div style="margin-top: 20px;">
                    <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">🔸 GraphQL API Rate Limit</h3>
                    <div class="rate-limit-summary">
                        <div class="rate-limit-gauge ${graphqlStatusClass}">
                            <div class="gauge-value">${graphqlRemaining}</div>
                            <div class="gauge-label">Remaining</div>
                            <div class="gauge-total">/ ${graphqlLimit}</div>
                        </div>
                        <div class="rate-limit-details">
                            <div class="rate-limit-row">
                                <span class="label">Usage:</span>
                                <span class="value">${graphqlUsagePercent}%</span>
                            </div>
                            <div class="rate-limit-row">
                                <span class="label">Resets at:</span>
                                <span class="value">${graphqlResetDate}</span>
                            </div>
                            <div class="rate-limit-row">
                                <span class="label">Used:</span>
                                <span class="value">${graphqlLimit - graphqlRemaining}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Show placeholder if GraphQL data not available
            graphqlSection = `
                <div style="margin-top: 20px;">
                    <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">🔸 GraphQL API Rate Limit</h3>
                    <div style="padding: 20px; text-align: center; color: #6b7280; font-size: 14px;">
                        <p>GraphQL rate limit data not available in response</p>
                        <p style="font-size: 12px; margin-top: 10px;">Make a GraphQL API call to populate this data</p>
                    </div>
                </div>
            `;
        }
        
        // Add CodeQL/Code Scanning section
        if (codeScanning) {
            const csRemaining = codeScanning.remaining || 0;
            const csLimit = codeScanning.limit || 5000;
            const csResetDate = codeScanning.reset ? new Date(codeScanning.reset * 1000).toLocaleString() : 'Unknown';
            const csUsagePercent = ((csLimit - csRemaining) / csLimit * 100).toFixed(1);
            const csStatusClass = csRemaining > 1000 ? 'status-good' : csRemaining > 100 ? 'status-warning' : 'status-critical';
            
            codeScanningSection = `
                <div style="margin-top: 20px;">
                    <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">🔍 CodeQL / Code Scanning API Rate Limit</h3>
                    <div class="rate-limit-summary">
                        <div class="rate-limit-gauge ${csStatusClass}">
                            <div class="gauge-value">${csRemaining}</div>
                            <div class="gauge-label">Remaining</div>
                            <div class="gauge-total">/ ${csLimit}</div>
                        </div>
                        <div class="rate-limit-details">
                            <div class="rate-limit-row">
                                <span class="label">Usage:</span>
                                <span class="value">${csUsagePercent}%</span>
                            </div>
                            <div class="rate-limit-row">
                                <span class="label">Resets at:</span>
                                <span class="value">${csResetDate}</span>
                            </div>
                            <div class="rate-limit-row">
                                <span class="label">Used:</span>
                                <span class="value">${csLimit - csRemaining}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Show placeholder if CodeQL data not available
            codeScanningSection = `
                <div style="margin-top: 20px;">
                    <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">🔍 CodeQL / Code Scanning API Rate Limit</h3>
                    <div style="padding: 20px; text-align: center; color: #6b7280; font-size: 14px;">
                        <p>CodeQL rate limit data not available in response</p>
                        <p style="font-size: 12px; margin-top: 10px;">Make a CodeQL API call to populate this data</p>
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = `
            <div style="margin-bottom: 20px;">
                <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">🔹 REST API Rate Limit</h3>
                <div class="rate-limit-summary">
                    <div class="rate-limit-gauge ${statusClass}">
                        <div class="gauge-value">${restRemaining}</div>
                        <div class="gauge-label">Remaining</div>
                        <div class="gauge-total">/ ${restLimit}</div>
                    </div>
                    <div class="rate-limit-details">
                        <div class="rate-limit-row">
                            <span class="label">Usage:</span>
                            <span class="value">${restUsagePercent}%</span>
                        </div>
                        <div class="rate-limit-row">
                            <span class="label">Resets at:</span>
                            <span class="value">${restResetDate}</span>
                        </div>
                        <div class="rate-limit-row">
                            <span class="label">Used:</span>
                            <span class="value">${restLimit - restRemaining}</span>
                        </div>
                    </div>
                </div>
            </div>
            ${graphqlSection}
            ${codeScanningSection}
        `;
    }

    // Render aggregate statistics across all peers
    _renderAggregateStats(aggregate) {
        if (!aggregate || aggregate.total_peers <= 1) {
            return '';  // Don't show if no other peers
        }
        
        const lastSynced = aggregate.last_synced ? new Date(aggregate.last_synced * 1000).toLocaleTimeString() : 'Never';
        
        return `
            <div class="cache-detail-section" style="margin-top: 15px; padding-top: 15px; border-top: 2px solid #e5e7eb;">
                <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #1f2937;">📊 Aggregate Stats (All Peers)</h4>
                <div class="cache-detail-row">
                    <span class="detail-label">Total API Calls (All Peers):</span>
                    <span class="detail-value" style="color: #ef4444; font-weight: 700;">${aggregate.total_api_calls || 0}</span>
                </div>
                <div class="cache-detail-row">
                    <span class="detail-label">Total Cache Hits (All Peers):</span>
                    <span class="detail-value" style="color: #10b981; font-weight: 700;">${aggregate.total_cache_hits || 0}</span>
                </div>
                <div class="cache-detail-row">
                    <span class="detail-label">Connected Peers:</span>
                    <span class="detail-value">${aggregate.total_peers || 0}</span>
                </div>
                <div class="cache-detail-row">
                    <span class="detail-label">Last Synced:</span>
                    <span class="detail-value">${lastSynced}</span>
                </div>
            </div>
        `;
    }

    // Invalidate cache via MCP SDK
    async invalidateCache(pattern = null) {
        try {
            if (this.mcp) {
                console.log('[GitHub Workflows] Calling gh_invalidate_cache via MCP SDK...');
                const result = await this.mcp.request('tools/call', {
                    name: 'gh_invalidate_cache',
                    arguments: pattern ? { pattern } : {}
                });
                
                console.log('[GitHub Workflows] Cache invalidation result:', result);
                
                if (result && result.success) {
                    showToast(`Cache cleared: ${result.invalidated} entries`, 'success');
                    await this.fetchCacheStats();
                } else if (result && result.error) {
                    showToast(`Failed to clear cache: ${result.error}`, 'error');
                } else {
                    showToast('Failed to clear cache', 'error');
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error invalidating cache:', error);
            showToast('Error clearing cache', 'error');
        }
    }
    
    // ==================== Token & Environment Configuration ====================
    
    async setGitHubToken() {
        const tokenInput = document.getElementById('github-token-input');
        const token = tokenInput?.value.trim();
        
        if (!token) {
            this.showTokenStatus('error', 'Please enter a token');
            return;
        }
        
        try {
            const result = await this.mcp.request('tools/call', {
                name: 'gh_set_token',
                arguments: { token }
            });
            
            if (result.error) {
                this.showTokenStatus('error', `Error: ${result.error}`);
            } else {
                this.showTokenStatus('success', result.message || 'Token configured');
                tokenInput.value = '';
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error setting token:', error);
            this.showTokenStatus('error', `Failed: ${error.message}`);
        }
    }
    
    toggleTokenVisibility() {
        const tokenInput = document.getElementById('github-token-input');
        const toggleBtn = document.getElementById('toggle-token-btn');
        
        if (tokenInput && toggleBtn) {
            tokenInput.type = tokenInput.type === 'password' ? 'text' : 'password';
            toggleBtn.textContent = tokenInput.type === 'password' ? '👁️ Show' : '🔒 Hide';
        }
    }
    
    showTokenStatus(type, message) {
        const statusDiv = document.getElementById('token-status');
        if (!statusDiv) return;
        
        statusDiv.style.display = 'block';
        statusDiv.className = type === 'success' ? 'success-message' : 'error-message';
        statusDiv.textContent = message;
        setTimeout(() => { statusDiv.style.display = 'none'; }, 5000);
    }
    
    async fetchEnvVars() {
        try {
            const result = await this.mcp.request('tools/call', {
                name: 'gh_get_env_vars',
                arguments: {}
            });
            
            if (!result.error) {
                this.displayEnvVars(result.env_vars);
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching env vars:', error);
        }
    }
    
    displayEnvVars(envVars) {
        const displayDiv = document.getElementById('env-vars-display');
        if (!displayDiv) return;
        
        displayDiv.style.display = 'block';
        let html = '<div style="max-height: 300px; overflow-y: auto; margin-top: 10px; padding: 10px; background: #f9fafb; border-radius: 4px;">';
        html += '<table style="width: 100%; font-size: 12px;"><thead><tr style="background: #e5e7eb;"><th style="padding: 8px;">Variable</th><th style="padding: 8px;">Value</th></tr></thead><tbody>';
        
        for (const [key, value] of Object.entries(envVars)) {
            html += `<tr style="border-bottom: 1px solid #e5e7eb;"><td style="padding: 8px;">${key}</td><td style="padding: 8px;">${value || '<span style="color: #9ca3af;">Not set</span>'}</td></tr>`;
        }
        
        html += '</tbody></table></div>';
        displayDiv.innerHTML = html;
    }
    
    // ==================== Autoscaler & Runner Management ====================
    
    async configureAutoscaler() {
        const enabled = document.getElementById('autoscaler-enabled')?.value === 'true';
        const pollInterval = parseInt(document.getElementById('autoscaler-poll-interval')?.value) || 120;
        const maxRunners = document.getElementById('autoscaler-max-runners')?.value;
        const sinceDays = parseInt(document.getElementById('autoscaler-since-days')?.value) || 1;
        const owner = document.getElementById('autoscaler-owner')?.value.trim();
        
        try {
            const args = { enabled, poll_interval: pollInterval, since_days: sinceDays };
            if (maxRunners) args.max_runners = parseInt(maxRunners);
            if (owner) args.owner = owner;
            
            const result = await this.mcp.request('tools/call', {
                name: 'gh_configure_autoscaler',
                arguments: args
            });
            
            if (!result.error) {
                showToast(result.message || 'Autoscaler configured', 'success');
                this.displayAutoscalerStatus(result);
            } else {
                showToast(`Error: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error:', error);
            showToast(`Failed: ${error.message}`, 'error');
        }
    }
    
    async getAutoscalerStatus() {
        try {
            const result = await this.mcp.request('tools/call', {
                name: 'gh_autoscaler_status',
                arguments: {}
            });
            
            if (!result.error) {
                this.displayAutoscalerStatus(result);
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error:', error);
        }
    }
    
    displayAutoscalerStatus(status) {
        const displayDiv = document.getElementById('autoscaler-status-display');
        if (!displayDiv) return;
        
        const config = status.config || {};
        let html = '<div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 15px; margin-top: 10px;">';
        html += '<div style="display: grid; gap: 8px; font-size: 14px;">';
        html += `<div><strong>Status:</strong> <span style="color: ${status.enabled ? '#10b981' : '#ef4444'};">${status.enabled ? '✓ Enabled' : '✗ Disabled'}</span></div>`;
        html += `<div><strong>Poll Interval:</strong> ${config.poll_interval}s</div>`;
        html += `<div><strong>Max Runners:</strong> ${config.max_runners}</div>`;
        html += '</div></div>';
        displayDiv.innerHTML = html;
    }
    
    async listActiveRunners() {
        try {
            const result = await this.mcp.request('tools/call', {
                name: 'gh_list_active_runners',
                arguments: { include_docker: true }
            });
            
            if (!result.error) {
                this.displayActiveRunners(result.active_runners || []);
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error:', error);
        }
    }
    
    displayActiveRunners(runners) {
        const container = document.getElementById('active-runners-container');
        if (!container) return;
        
        if (runners.length === 0) {
            container.innerHTML = '<p style="color: #6b7280; padding: 20px; text-align: center;">No active runners found. Click "Track" to load runners.</p>';
            return;
        }
        
        let html = '<div style="display: grid; gap: 15px;">';
        for (const runner of runners) {
            const p2p = runner.p2p_status || {};
            const status = runner.status || 'unknown';
            const statusColor = status === 'online' ? '#10b981' : status === 'offline' ? '#ef4444' : '#f59e0b';
            const statusText = status.toUpperCase();
            const busy = runner.busy ? '(Busy)' : '(Idle)';
            const os = runner.os || 'Unknown OS';
            const labels = runner.labels || [];
            
            html += `
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; background: ${status === 'online' ? '#f0fdf4' : '#fff'};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="margin: 0;">${this._escapeHtml(runner.name || 'Unknown Runner')}</h4>
                        <span style="color: ${statusColor}; font-weight: bold;">● ${statusText} ${busy}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px;">
                        <div><strong>ID:</strong> ${runner.id || 'N/A'}</div>
                        <div><strong>OS:</strong> ${os}</div>
                        <div><strong>P2P Cache:</strong> <span style="color: ${p2p.cache_enabled ? '#10b981' : '#6b7280'};">${p2p.cache_enabled ? '✓ Enabled' : '✗ Disabled'}</span></div>
                        <div><strong>Libp2p:</strong> <span style="color: ${p2p.libp2p_bootstrapped ? '#10b981' : '#6b7280'};">${p2p.libp2p_bootstrapped ? '✓ Bootstrapped' : '✗ Not bootstrapped'}</span></div>
                    </div>
                    ${labels.length > 0 ? `
                    <div style="margin-top: 10px;">
                        <strong style="font-size: 12px;">Labels:</strong>
                        <div style="display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px;">
                            ${labels.map(l => `<span style="background: #e0e7ff; color: #4338ca; padding: 2px 8px; border-radius: 4px; font-size: 11px;">${this._escapeHtml(typeof l === 'string' ? l : l.name || l)}</span>`).join('')}
                        </div>
                    </div>
                    ` : ''}
                </div>
            `;
        }
        html += '</div>';
        container.innerHTML = html;
    }
    
    async trackRunners() {
        const repo = document.getElementById('runner-repo-input')?.value.trim();
        const org = document.getElementById('runner-org-input')?.value.trim();
        
        try {
            let result;
            
            // If no repo/org specified, list active runners with P2P info
            if (!repo && !org) {
                result = await this.mcp.request('tools/call', {
                    name: 'gh_list_active_runners',
                    arguments: { include_docker: true }
                });
                
                if (!result.error) {
                    const runners = result.active_runners || [];
                    showToast(`Found ${runners.length} active runner(s)`, 'success');
                    this.displayActiveRunners(runners);
                }
            } else {
                // If repo/org specified, list all runners for that scope
                const args = {};
                if (repo) args.repo = repo;
                if (org) args.org = org;
                
                result = await this.mcp.request('tools/call', {
                    name: 'gh_list_runners',
                    arguments: args
                });
                
                if (!result.error) {
                    const runners = result.runners || [];
                    showToast(`Found ${runners.length} runner(s)`, 'success');
                    // Display in the active runners section
                    this.displayActiveRunners(runners);
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error:', error);
            showToast('Error fetching runners', 'error');
        }
    }
    
    async getRunnerDetails() {
        const repo = document.getElementById('runner-repo-input')?.value.trim();
        const org = document.getElementById('runner-org-input')?.value.trim();
        
        if (!repo && !org) {
            showToast('Please specify repo or org', 'error');
            return;
        }
        
        try {
            const args = {};
            if (repo) args.repo = repo;
            if (org) args.org = org;
            
            const result = await this.mcp.request('tools/call', {
                name: 'gh_get_runner_details',
                arguments: args
            });
            
            if (!result.error) {
                showToast(`Retrieved details for ${result.total_count} runner(s)`, 'success');
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error:', error);
        }
    }
    
    async showP2PStats() {
        try {
            const result = await this.mcp.request('tools/call', {
                name: 'gh_get_cache_stats',
                arguments: {}
            });
            
            if (!result.error) {
                const agg = result.aggregate || {};
                alert(`P2P Stats:\nTotal API Calls: ${agg.total_api_calls || 0}\nTotal Cache Hits: ${agg.total_cache_hits || 0}\nConnected Peers: ${agg.total_peers || 0}`);
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error:', error);
        }
    }
    
    async refreshActiveRunners() {
        await this.listActiveRunners();
    }
}

// Global instance

// Global instance
let githubManager = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Create MCP client for GitHub workflows
    const mcpClient = typeof MCPClient !== 'undefined' ? new MCPClient() : null;
    githubManager = new GitHubWorkflowsManager(mcpClient);
    
    // Check if GitHub tab exists before initializing
    const githubTab = document.getElementById('github-workflows');
    if (githubTab) {
        githubManager.initialize();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GitHubWorkflowsManager;
}

// Initialize and load status on page load
document.addEventListener('DOMContentLoaded', () => {
    if (githubManager && document.getElementById('github-workflows')) {
        setTimeout(() => {
            githubManager.getAutoscalerStatus?.();
            githubManager.listActiveRunners?.();
        }, 1500);
    }
});

// Helper for collapsible sections
function toggleSection(sectionId) {
    const el = document.getElementById(sectionId);
    if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}
