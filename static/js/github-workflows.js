// GitHub Workflows Integration for IPFS Accelerate Dashboard
// Provides workflow monitoring and runner management features

class GitHubWorkflowsManager {
    constructor() {
        this.workflows = {};
        this.runners = [];
        this.updateInterval = null;
    }

    // Initialize the workflows manager
    async initialize() {
        console.log('[GitHub Workflows] Initializing...');
        await this.fetchWorkflows();
        await this.fetchRunners();
        this.startAutoRefresh();
    }

    // Fetch workflows from the server
    async fetchWorkflows() {
        try {
            const response = await fetch('/api/github/workflows');
            if (response.ok) {
                this.workflows = await response.json();
                this.renderWorkflows();
            } else {
                console.error('[GitHub Workflows] Failed to fetch workflows:', response.statusText);
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching workflows:', error);
        }
    }

    // Fetch runners from the server
    async fetchRunners() {
        try {
            const response = await fetch('/api/github/runners');
            if (response.ok) {
                this.runners = await response.json();
                this.renderRunners();
            } else {
                console.error('[GitHub Workflows] Failed to fetch runners:', response.statusText);
            }
        } catch (error) {
            console.error('[GitHub Workflows] Error fetching runners:', error);
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

    // Provision runner for a repository
    async provisionRunner(repo) {
        try {
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
            await this.fetchWorkflows();
            await this.fetchRunners();
        }, interval);
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
}

// Global instance
let githubManager = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    githubManager = new GitHubWorkflowsManager();
    
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
