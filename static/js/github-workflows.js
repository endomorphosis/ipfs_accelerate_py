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

    // Initialize the workflows manager
    async initialize() {
        console.log('[GitHub Workflows] Initializing with MCP SDK...');
        await Promise.all([
            this.fetchWorkflows(),
            this.fetchRunners(),
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
                    this.renderWorkflows();
                    console.log(`[GitHub Workflows] Loaded ${Object.keys(this.workflows).length} repositories`);
                } else if (result && result.error) {
                    console.error('[GitHub Workflows] MCP tool returned error:', result.error);
                } else {
                    console.error('[GitHub Workflows] Invalid response from MCP tool');
                }
            } else {
                // Fallback to direct API
                console.warn('[GitHub Workflows] Falling back to direct API call');
                const response = await fetch('/api/github/workflows');
                if (response.ok) {
                    this.workflows = await response.json();
                    this.renderWorkflows();
                } else {
                    console.error('[GitHub Workflows] Failed to fetch workflows:', response.statusText);
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching workflows:', error);
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
                    this.renderRunners();
                    console.log(`[GitHub Workflows] Loaded ${this.runners.length} runners`);
                } else if (result && result.error) {
                    console.error('[GitHub Workflows] MCP tool returned error:', result.error);
                } else {
                    console.error('[GitHub Workflows] Invalid response from MCP tool');
                }
            } else {
                // Fallback to direct API
                console.warn('[GitHub Workflows] Falling back to direct API call');
                const response = await fetch('/api/github/runners');
                if (response.ok) {
                    this.runners = await response.json();
                    this.renderRunners();
                } else {
                    console.error('[GitHub Workflows] Failed to fetch runners:', response.statusText);
                }
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching runners:', error);
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

        let html = '<div class="workflows-grid">';

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

        html += '</div>';
        container.innerHTML = html;
    }

    // Render runners in the dashboard
    renderRunners() {
        const container = document.getElementById('github-runners-container');
        if (!container) return;

        let html = '<div class="runners-list">';

        if (this.runners.length === 0) {
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
        const multiformatsStatus = this.cacheStats.content_addressing?.multiformats_available ? '✓ IPLD/Multiformats' : '✗ Unavailable';
        
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

        const core = this.rateLimit.resources?.core || {};
        const remaining = core.remaining || 0;
        const limit = core.limit || 5000;
        const resetDate = core.reset ? new Date(core.reset * 1000).toLocaleTimeString() : 'Unknown';
        const usagePercent = ((limit - remaining) / limit * 100).toFixed(1);
        
        const statusClass = remaining > 1000 ? 'status-good' : remaining > 100 ? 'status-warning' : 'status-critical';
        
        container.innerHTML = `
            <div class="rate-limit-summary">
                <div class="rate-limit-gauge ${statusClass}">
                    <div class="gauge-value">${remaining}</div>
                    <div class="gauge-label">Remaining</div>
                    <div class="gauge-total">/ ${limit}</div>
                </div>
                <div class="rate-limit-details">
                    <div class="rate-limit-row">
                        <span class="label">Usage:</span>
                        <span class="value">${usagePercent}%</span>
                    </div>
                    <div class="rate-limit-row">
                        <span class="label">Resets at:</span>
                        <span class="value">${resetDate}</span>
                    </div>
                    <div class="rate-limit-row">
                        <span class="label">Used:</span>
                        <span class="value">${limit - remaining}</span>
                    </div>
                </div>
            </div>
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
}

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
