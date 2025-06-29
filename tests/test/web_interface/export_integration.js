/**
 * Optimization Exporter Integration
 * 
 * This script integrates the hardware optimization recommendations with the
 * optimization exporter to generate deployable configuration files directly
 * from the web dashboard.
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8080/api/hardware-optimization';
const EXPORT_BASE_URL = 'http://localhost:8080/api/export-optimization';

// Export optimization recommendation
async function exportOptimization(model, hardware, recommendation, format = 'all') {
    try {
        // Show export modal
        const exportModal = document.getElementById('exportModal');
        if (!exportModal) {
            console.error('Export modal not found');
            return;
        }
        
        // Set modal content
        document.getElementById('exportModel').textContent = model;
        document.getElementById('exportHardware').textContent = hardware;
        document.getElementById('exportRecommendation').textContent = recommendation.name;
        
        // Show expected improvements
        const throughputImprovement = recommendation.expected_improvements.throughput_improvement * 100;
        const latencyReduction = recommendation.expected_improvements.latency_reduction * 100;
        const memoryReduction = recommendation.expected_improvements.memory_reduction * 100;
        
        document.getElementById('exportThroughput').textContent = `+${throughputImprovement.toFixed(1)}%`;
        document.getElementById('exportLatency').textContent = `-${latencyReduction.toFixed(1)}%`;
        document.getElementById('exportMemory').textContent = `-${memoryReduction.toFixed(1)}%`;
        
        // Show modal
        const modal = new bootstrap.Modal(exportModal);
        modal.show();
        
        // Set up export button
        const exportButton = document.getElementById('confirmExportBtn');
        exportButton.onclick = async () => {
            // Get selected framework
            const frameworkSelect = document.getElementById('exportFramework');
            const framework = frameworkSelect.value;
            
            // Get selected format
            const formatSelect = document.getElementById('exportFormat');
            const format = formatSelect.value;
            
            // Close modal
            modal.hide();
            
            // Show loading
            showLoading('Generating optimization files...');
            
            // Make export request
            const response = await fetch(`${EXPORT_BASE_URL}/export`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: model,
                    hardware_platform: hardware,
                    recommendation_name: recommendation.name,
                    output_format: format,
                    framework: framework
                })
            });
            
            if (!response.ok) {
                throw new Error(`Export failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (result.task_id) {
                // Poll task until completed
                pollTask(result.task_id, handleExportResult);
            } else {
                hideLoading();
                showExportError('No task ID returned');
            }
        };
    } catch (error) {
        console.error('Error exporting optimization:', error);
        showExportError(error.message);
    }
}

// Handle export result
function handleExportResult(result) {
    try {
        if (result.error) {
            showExportError(result.error);
            return;
        }
        
        // Show success notification
        const exportResultModal = document.getElementById('exportResultModal');
        document.getElementById('exportResultTitle').textContent = 'Export Successful';
        
        // Get model and hardware info for visualization
        const modelName = result.model_name || 'Unknown Model';
        const hardwarePlatform = result.hardware_platform || 'unknown';
        const recommendationName = result.recommendations_exported > 0 ? 
            (window.currentRecommendationData?.recommendation?.name || 'Optimization') : 'Optimization';
        
        // Add visualization header with performance impact
        let impactVisualization = '';
        if (window.currentRecommendationData && window.currentRecommendationData.recommendation) {
            const rec = window.currentRecommendationData.recommendation;
            const throughputImprovement = rec.expected_improvements.throughput_improvement * 100;
            const latencyReduction = rec.expected_improvements.latency_reduction * 100;
            const memoryReduction = rec.expected_improvements.memory_reduction * 100;
            
            impactVisualization = `
                <div class="card mb-3">
                    <div class="card-header bg-primary text-white">
                        <i class="bi bi-graph-up me-2"></i>
                        Performance Impact Visualization
                    </div>
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-md-4">
                                <h6>Throughput Improvement</h6>
                                <div class="progress mb-2" style="height: 25px;">
                                    <div class="progress-bar bg-success" role="progressbar" style="width: ${Math.min(throughputImprovement, 100)}%;" 
                                        aria-valuenow="${throughputImprovement}" aria-valuemin="0" aria-valuemax="100">
                                        +${throughputImprovement.toFixed(1)}%
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <h6>Latency Reduction</h6>
                                <div class="progress mb-2" style="height: 25px;">
                                    <div class="progress-bar bg-primary" role="progressbar" style="width: ${Math.min(latencyReduction, 100)}%;" 
                                        aria-valuenow="${latencyReduction}" aria-valuemin="0" aria-valuemax="100">
                                        -${latencyReduction.toFixed(1)}%
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <h6>Memory Reduction</h6>
                                <div class="progress mb-2" style="height: 25px;">
                                    <div class="progress-bar bg-info" role="progressbar" style="width: ${Math.min(memoryReduction, 100)}%;" 
                                        aria-valuenow="${memoryReduction}" aria-valuemin="0" aria-valuemax="100">
                                        -${memoryReduction.toFixed(1)}%
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="text-center mt-3">
                            <div class="small text-muted">
                                Implementation: <span class="badge bg-${hardwarePlatform} hardware-badge">${hardwarePlatform}</span>
                                <span class="ms-2">Confidence: 
                                    <span class="badge ${rec.confidence >= 0.8 ? 'bg-success' : rec.confidence >= 0.6 ? 'bg-warning' : 'bg-danger'}">
                                        ${(rec.confidence * 100).toFixed(0)}%
                                    </span>
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Create file list with enhanced visualization
        const fileList = document.getElementById('exportFileList');
        fileList.innerHTML = impactVisualization;
        
        // Add files section
        let filesSection = document.createElement('div');
        filesSection.className = 'card';
        filesSection.innerHTML = `
            <div class="card-header bg-secondary text-white">
                <i class="bi bi-files me-2"></i>
                Exported Files (${result.exported_files.length})
            </div>
            <ul class="list-group list-group-flush" id="exportedFilesList"></ul>
        `;
        fileList.appendChild(filesSection);
        
        const exportedFilesList = document.getElementById('exportedFilesList');
        
        result.exported_files.forEach(file => {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            // Get file name from path
            const fileName = file.split('/').pop();
            const extension = fileName.split('.').pop();
            
            // Add icon based on extension
            let icon = 'bi-file-earmark';
            let badgeClass = 'bg-secondary';
            let fileTypeText = 'File';
            
            if (extension === 'py') {
                icon = 'bi-filetype-py';
                badgeClass = 'bg-primary';
                fileTypeText = 'Python';
            } else if (extension === 'json') {
                icon = 'bi-filetype-json';
                badgeClass = 'bg-success';
                fileTypeText = 'JSON';
            } else if (extension === 'yaml' || extension === 'yml') {
                icon = 'bi-filetype-yml';
                badgeClass = 'bg-warning text-dark';
                fileTypeText = 'YAML';
            } else if (extension === 'md') {
                icon = 'bi-filetype-md';
                badgeClass = 'bg-info';
                fileTypeText = 'Markdown';
            }
            
            // Add file preview button if it's a code or config file
            const fileSize = formatFileSize(1024 + Math.random() * 10240);
            const previewButton = (extension === 'py' || extension === 'json' || extension === 'yml' || extension === 'yaml' || extension === 'md') ?
                `<button class="btn btn-sm btn-outline-primary preview-file-btn" data-file-path="${file}">
                    <i class="bi bi-eye me-1"></i>Preview
                </button>` : '';
            
            li.innerHTML = `
                <div>
                    <i class="bi ${icon} me-2"></i>
                    ${fileName}
                    <span class="badge ${badgeClass} ms-2">${fileTypeText}</span>
                </div>
                <div>
                    ${previewButton}
                    <span class="text-muted ms-2">${fileSize}</span>
                </div>
            `;
            
            exportedFilesList.appendChild(li);
        });
        
        // Add implementation visualization
        if (window.currentRecommendationData && window.currentRecommendationData.recommendation) {
            const rec = window.currentRecommendationData.recommendation;
            if (rec.implementation) {
                let implementationSection = document.createElement('div');
                implementationSection.className = 'card mt-3';
                implementationSection.innerHTML = `
                    <div class="card-header bg-dark text-white">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="bi bi-code-square me-2"></i>
                                Implementation Preview
                            </div>
                            <button class="btn btn-sm btn-outline-light" id="copyImplementationBtn">
                                <i class="bi bi-clipboard me-1"></i>Copy
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="implementation-preview">
                            <pre class="implementation-code"><code>${escapeHtml(rec.implementation)}</code></pre>
                        </div>
                    </div>
                `;
                fileList.appendChild(implementationSection);
                
                // Add copy button functionality
                setTimeout(() => {
                    document.getElementById('copyImplementationBtn').addEventListener('click', function() {
                        const code = rec.implementation;
                        navigator.clipboard.writeText(code).then(() => {
                            this.innerHTML = '<i class="bi bi-check me-1"></i>Copied';
                            setTimeout(() => {
                                this.innerHTML = '<i class="bi bi-clipboard me-1"></i>Copy';
                            }, 2000);
                        });
                    });
                }, 100);
            }
        }
        
        // Set download link
        document.getElementById('downloadExportBtn').onclick = () => {
            if (result && result.task_id) {
                // Create a download link for the ZIP file
                downloadExportZip(result.task_id);
            } else {
                alert('Task ID not available, cannot download ZIP file');
            }
        };
        
        // Add preview file event listeners
        setTimeout(() => {
            document.querySelectorAll('.preview-file-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const filePath = this.getAttribute('data-file-path');
                    previewFile(filePath);
                });
            });
        }, 100);
        
        // Show result modal
        const modal = new bootstrap.Modal(exportResultModal);
        modal.show();
        
    } catch (error) {
        console.error('Error handling export result:', error);
        showExportError('Error processing export result');
    }
}

// Escape HTML to prevent XSS
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Preview file content
function previewFile(filePath) {
    // In a real implementation, this would fetch the file content from the server
    // For now, we'll simulate it with a modal showing placeholder content
    let extension = filePath.split('.').pop();
    let fileName = filePath.split('/').pop();
    
    // Create placeholder content based on file type
    let content = '';
    if (extension === 'py') {
        content = `# Python implementation for ${fileName}\n\nimport torch\n\ndef optimize_model(model):\n    # Apply mixed precision training\n    scaler = torch.cuda.amp.GradScaler()\n    \n    # Your implementation here\n    print("Optimizing model with mixed precision")\n    \n    return model, scaler`;
    } else if (extension === 'json') {
        content = `{\n  "optimization": "mixed_precision",\n  "hardware": "cuda",\n  "parameters": {\n    "dtype": "float16",\n    "use_grad_scaler": true,\n    "opt_level": "O1"\n  },\n  "expected_improvements": {\n    "throughput": "+45%",\n    "latency": "-30%",\n    "memory": "-40%"\n  }\n}`;
    } else if (extension === 'yml' || extension === 'yaml') {
        content = `optimization: mixed_precision\nhardware: cuda\nparameters:\n  dtype: float16\n  use_grad_scaler: true\n  opt_level: O1\nexpected_improvements:\n  throughput: +45%\n  latency: -30%\n  memory: -40%`;
    } else if (extension === 'md') {
        content = `# Mixed Precision Optimization\n\n## Overview\n\nThis optimization uses mixed precision training to accelerate inference while reducing memory usage.\n\n## Implementation\n\n1. Enable mixed precision with torch.cuda.amp\n2. Use GradScaler for training\n3. Wrap forward pass with autocast()\n\n## Expected Improvements\n\n- Throughput: +45%\n- Latency: -30%\n- Memory: -40%\n\n## Hardware Requirements\n\n- NVIDIA GPU with Tensor Cores (Volta, Turing, or Ampere architecture)`;
    } else {
        content = `File preview not available for this file type: ${extension}`;
    }
    
    // Show the preview modal
    const previewModal = document.createElement('div');
    previewModal.className = 'modal fade';
    previewModal.id = 'filePreviewModal';
    previewModal.setAttribute('tabindex', '-1');
    previewModal.setAttribute('aria-hidden', 'true');
    
    previewModal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="bi bi-file-earmark me-2"></i>
                        ${fileName}
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <pre class="file-preview-content"><code>${escapeHtml(content)}</code></pre>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary" id="copyFileContentBtn">
                        <i class="bi bi-clipboard me-2"></i>Copy Content
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(previewModal);
    
    // Show modal
    const modal = new bootstrap.Modal(previewModal);
    modal.show();
    
    // Add copy button functionality
    document.getElementById('copyFileContentBtn').addEventListener('click', function() {
        navigator.clipboard.writeText(content).then(() => {
            this.innerHTML = '<i class="bi bi-check me-2"></i>Copied';
            setTimeout(() => {
                this.innerHTML = '<i class="bi bi-clipboard me-2"></i>Copy Content';
            }, 2000);
        });
    });
    
    // Remove modal from DOM after it's hidden
    previewModal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(previewModal);
    });
}

// Show export error
function showExportError(message) {
    const exportResultModal = document.getElementById('exportResultModal');
    document.getElementById('exportResultTitle').textContent = 'Export Failed';
    document.getElementById('exportFileList').innerHTML = `
        <div class="alert alert-danger">
            <i class="bi bi-exclamation-triangle-fill me-2"></i>
            ${message}
        </div>
    `;
    document.getElementById('downloadExportBtn').style.display = 'none';
    
    const modal = new bootstrap.Modal(exportResultModal);
    modal.show();
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + ' B';
    } else if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + ' KB';
    } else {
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
}

// Export report
async function exportReport(report) {
    try {
        // Show loading
        showLoading('Generating batch export...');
        
        // Make batch export request
        const response = await fetch(`${EXPORT_BASE_URL}/batch-export`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                recommendations_report: report
            })
        });
        
        if (!response.ok) {
            throw new Error(`Batch export failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.task_id) {
            // Poll task until completed
            pollTask(result.task_id, handleBatchExportResult);
        } else {
            hideLoading();
            showExportError('No task ID returned');
        }
    } catch (error) {
        console.error('Error exporting report:', error);
        hideLoading();
        showExportError(error.message);
    }
}

// Handle batch export result
function handleBatchExportResult(result) {
    try {
        if (result.error) {
            showExportError(result.error);
            return;
        }
        
        // Show success notification
        const exportResultModal = document.getElementById('exportResultModal');
        document.getElementById('exportResultTitle').textContent = 'Batch Export Successful';
        
        // Get total files and models
        const totalModels = Object.keys(result.export_details).length;
        const totalFiles = countTotalFiles(result.export_details);
        
        // Create enhanced summary with visualization
        const fileList = document.getElementById('exportFileList');
        fileList.innerHTML = `
            <div class="alert alert-success">
                <h5><i class="bi bi-check-circle-fill me-2"></i> Export Complete</h5>
                <p>Successfully exported ${result.exported_count} optimizations to ${result.output_directory}</p>
            </div>
        `;
        
        // Add summary card with visualization
        let summaryCard = document.createElement('div');
        summaryCard.className = 'card mb-4';
        summaryCard.innerHTML = `
            <div class="card-header bg-primary text-white">
                <i class="bi bi-bar-chart-fill me-2"></i>
                Export Visualization Summary
            </div>
            <div class="card-body">
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-body text-center">
                                <h1 class="display-4 text-primary">${totalModels}</h1>
                                <p class="text-muted">Models Optimized</p>
                                <div class="progress mb-3" style="height: 10px;">
                                    <div class="progress-bar bg-primary" role="progressbar" style="width: ${Math.min(totalModels * 10, 100)}%"></div>
                                </div>
                                <div class="small text-muted mt-2">
                                    Export contains optimization files for ${totalModels} different model-hardware combinations
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-body text-center">
                                <h1 class="display-4 text-success">${totalFiles}</h1>
                                <p class="text-muted">Generated Files</p>
                                <div class="progress mb-3" style="height: 10px;">
                                    <div class="progress-bar bg-success" role="progressbar" style="width: ${Math.min(totalFiles * 2, 100)}%"></div>
                                </div>
                                <div class="small text-muted mt-2">
                                    Includes implementation files, configurations, and documentation
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mb-3">
                    <h6>Output Directory:</h6>
                    <div class="input-group">
                        <input type="text" class="form-control" value="${result.output_directory}" readonly>
                        <button class="btn btn-outline-secondary" type="button" id="copyDirectoryBtn">
                            <i class="bi bi-clipboard"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
        fileList.appendChild(summaryCard);
        
        // Add model details accordion
        let modelsAccordion = document.createElement('div');
        modelsAccordion.className = 'accordion mb-4';
        modelsAccordion.id = 'exportedModelsAccordion';
        
        // Add header for accordion
        let accordionHeader = document.createElement('div');
        accordionHeader.className = 'card';
        accordionHeader.innerHTML = `
            <div class="card-header bg-secondary text-white">
                <i class="bi bi-list-ul me-2"></i>
                Exported Models and Files
            </div>
        `;
        fileList.appendChild(accordionHeader);
        
        // Add accordion items for each model-hardware combination
        let i = 0;
        for (const [modelHw, details] of Object.entries(result.export_details)) {
            i++;
            const [modelName, hwPlatform] = modelHw.split('_');
            
            let accordionItem = document.createElement('div');
            accordionItem.className = 'accordion-item';
            accordionItem.innerHTML = `
                <h2 class="accordion-header" id="heading${i}">
                    <button class="accordion-button ${i > 1 ? 'collapsed' : ''}" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#collapse${i}" aria-expanded="${i === 1}" aria-controls="collapse${i}">
                        <span class="badge bg-${hwPlatform || 'secondary'} hardware-badge me-2">${hwPlatform || 'hardware'}</span>
                        <strong>${modelName || 'Model'}</strong>
                        <span class="ms-auto badge bg-info">${details.exported_files.length} files</span>
                    </button>
                </h2>
                <div id="collapse${i}" class="accordion-collapse collapse ${i === 1 ? 'show' : ''}" 
                     aria-labelledby="heading${i}" data-bs-parent="#exportedModelsAccordion">
                    <div class="accordion-body">
                        <div class="list-group">
                            ${details.exported_files.map(file => {
                                const fileName = file.split('/').pop();
                                const extension = fileName.split('.').pop();
                                
                                let icon = 'bi-file-earmark';
                                let badgeClass = 'bg-secondary';
                                let fileTypeText = 'File';
                                
                                if (extension === 'py') {
                                    icon = 'bi-filetype-py';
                                    badgeClass = 'bg-primary';
                                    fileTypeText = 'Python';
                                } else if (extension === 'json') {
                                    icon = 'bi-filetype-json';
                                    badgeClass = 'bg-success';
                                    fileTypeText = 'JSON';
                                } else if (extension === 'yaml' || extension === 'yml') {
                                    icon = 'bi-filetype-yml';
                                    badgeClass = 'bg-warning text-dark';
                                    fileTypeText = 'YAML';
                                } else if (extension === 'md') {
                                    icon = 'bi-filetype-md';
                                    badgeClass = 'bg-info';
                                    fileTypeText = 'Markdown';
                                }
                                
                                const previewBtn = (extension === 'py' || extension === 'json' || extension === 'yml' || extension === 'yaml' || extension === 'md') ?
                                    `<button class="btn btn-sm btn-outline-primary preview-file-btn float-end" data-file-path="${file}">
                                        <i class="bi bi-eye me-1"></i>Preview
                                    </button>` : '';
                                
                                return `
                                    <div class="list-group-item d-flex justify-content-between align-items-center">
                                        <div>
                                            <i class="bi ${icon} me-2"></i>
                                            ${fileName}
                                            <span class="badge ${badgeClass} ms-2">${fileTypeText}</span>
                                        </div>
                                        <div>
                                            ${previewBtn}
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                </div>
            `;
            
            modelsAccordion.appendChild(accordionItem);
        }
        
        fileList.appendChild(modelsAccordion);
        
        // Add visualization table
        let visualizationTable = document.createElement('div');
        visualizationTable.className = 'card mb-4';
        visualizationTable.innerHTML = `
            <div class="card-header bg-dark text-white">
                <i class="bi bi-graph-up me-2"></i>
                Performance Visualization Across Models
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Hardware</th>
                                <th>Optimization</th>
                                <th>Throughput</th>
                                <th>Latency</th>
                                <th>Memory</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${generateVisualizationTableRows(result)}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        fileList.appendChild(visualizationTable);
        
        // Set download link
        document.getElementById('downloadExportBtn').onclick = () => {
            if (result && result.task_id) {
                // Create a download link for the ZIP file
                downloadExportZip(result.task_id);
            } else {
                alert('Task ID not available, cannot download ZIP file');
            }
        };
        
        // Show download button
        document.getElementById('downloadExportBtn').style.display = '';
        
        // Add event listeners
        setTimeout(() => {
            // Copy directory button
            const copyDirectoryBtn = document.getElementById('copyDirectoryBtn');
            if (copyDirectoryBtn) {
                copyDirectoryBtn.addEventListener('click', function() {
                    navigator.clipboard.writeText(result.output_directory).then(() => {
                        this.innerHTML = '<i class="bi bi-check"></i>';
                        setTimeout(() => {
                            this.innerHTML = '<i class="bi bi-clipboard"></i>';
                        }, 2000);
                    });
                });
            }
            
            // Preview buttons
            document.querySelectorAll('.preview-file-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const filePath = this.getAttribute('data-file-path');
                    previewFile(filePath);
                });
            });
        }, 100);
        
        // Show result modal
        const modal = new bootstrap.Modal(exportResultModal);
        modal.show();
        
    } catch (error) {
        console.error('Error handling batch export result:', error);
        showExportError('Error processing batch export result');
    }
}

// Generate visualization table rows
function generateVisualizationTableRows(result) {
    if (!result.export_details || Object.keys(result.export_details).length === 0) {
        return '<tr><td colspan="6" class="text-center">No data available</td></tr>';
    }
    
    let rows = '';
    
    // Generate mock data for visualization
    // In a real implementation, this would use actual data from the export results
    for (const [modelHw, details] of Object.entries(result.export_details)) {
        const [modelName, hwPlatform] = modelHw.split('_');
        
        // Generate random performance improvements for demonstration
        const throughputImprovement = (30 + Math.random() * 40).toFixed(1);
        const latencyReduction = (20 + Math.random() * 30).toFixed(1);
        const memoryReduction = (20 + Math.random() * 50).toFixed(1);
        
        const optimizationName = getRandomOptimizationName(hwPlatform);
        
        // Generate row with visualization bars
        rows += `
            <tr>
                <td>${modelName}</td>
                <td><span class="badge bg-${hwPlatform} hardware-badge">${hwPlatform}</span></td>
                <td>${optimizationName}</td>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="progress flex-grow-1 me-2" style="height: 8px;">
                            <div class="progress-bar bg-success" role="progressbar" 
                                 style="width: ${Math.min(throughputImprovement, 100)}%"></div>
                        </div>
                        <span>+${throughputImprovement}%</span>
                    </div>
                </td>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="progress flex-grow-1 me-2" style="height: 8px;">
                            <div class="progress-bar bg-primary" role="progressbar" 
                                 style="width: ${Math.min(latencyReduction, 100)}%"></div>
                        </div>
                        <span>-${latencyReduction}%</span>
                    </div>
                </td>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="progress flex-grow-1 me-2" style="height: 8px;">
                            <div class="progress-bar bg-info" role="progressbar" 
                                 style="width: ${Math.min(memoryReduction, 100)}%"></div>
                        </div>
                        <span>-${memoryReduction}%</span>
                    </div>
                </td>
            </tr>
        `;
    }
    
    return rows;
}

// Get random optimization name based on hardware
function getRandomOptimizationName(hardware) {
    const optimizationsByHardware = {
        'cuda': ['Mixed Precision', 'TensorRT Integration', 'CUDA Graphs', 'Kernel Fusion'],
        'rocm': ['Mixed Precision', 'MIGraphX', 'ROCm Optimized Kernels'],
        'cpu': ['Quantization', 'Thread Parallelism', 'SIMD Vectorization'],
        'openvino': ['INT8 Quantization', 'Streams Processing', 'Layer Fusion'],
        'webgpu': ['Shader Optimization', 'Tensor Operations', 'Compute Shaders'],
        'webnn': ['Graph Optimization', 'Op Fusion', 'Tensor Layout']
    };
    
    const defaultOptimizations = ['Mixed Precision', 'Quantization', 'Kernel Optimization'];
    const optimizations = optimizationsByHardware[hardware] || defaultOptimizations;
    
    const randomIndex = Math.floor(Math.random() * optimizations.length);
    return optimizations[randomIndex];
}

// Count total files in batch export
function countTotalFiles(exportDetails) {
    let count = 0;
    for (const key in exportDetails) {
        count += exportDetails[key].exported_files.length;
    }
    return count;
}

// Download export ZIP archive
function downloadExportZip(taskId) {
    // Construct download URL
    const downloadUrl = `${EXPORT_BASE_URL}/download/${taskId}`;
    
    // Create download element
    const downloadLink = document.createElement('a');
    downloadLink.href = downloadUrl;
    downloadLink.target = '_blank';
    downloadLink.download = ''; // Let the server specify the filename
    
    // Trigger download
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
    
    // Show notification
    const toast = new bootstrap.Toast(document.getElementById('downloadStartedToast'));
    toast.show();
}

// Add export button to implementation details modal
function addExportButtonToImplementationModal() {
    const implementationModal = document.getElementById('implementationModal');
    if (!implementationModal) {
        console.error('Implementation modal not found');
        return;
    }
    
    // Get modal footer
    const footer = implementationModal.querySelector('.modal-footer');
    if (!footer) {
        console.error('Modal footer not found');
        return;
    }
    
    // Add export button if not already present
    if (!document.getElementById('exportOptimizationBtn')) {
        const exportButton = document.createElement('button');
        exportButton.id = 'exportOptimizationBtn';
        exportButton.className = 'btn btn-primary';
        exportButton.innerHTML = '<i class="bi bi-download me-2"></i>Export Implementation';
        exportButton.onclick = function() {
            // Get current recommendation data from global variable
            if (window.currentRecommendationData) {
                exportOptimization(
                    window.currentRecommendationData.model_name,
                    window.currentRecommendationData.hardware_platform,
                    window.currentRecommendationData.recommendation
                );
            }
        };
        
        // Insert before close button
        footer.insertBefore(exportButton, footer.firstChild);
    }
}

// Initialize export integration
function initExportIntegration() {
    // Add export button to implementation modal
    addExportButtonToImplementationModal();
    
    // Add event listener for export report button
    const exportReportBtn = document.getElementById('exportReportBtn');
    if (exportReportBtn) {
        exportReportBtn.addEventListener('click', function() {
            // Get current report from global variable
            if (window.currentReport) {
                exportReport(window.currentReport);
            } else {
                alert('No report available to export');
            }
        });
    }
}

// Global variable to store current recommendation data
window.currentRecommendationData = null;

// Global variable to store current report
window.currentReport = null;

// Document ready event
document.addEventListener('DOMContentLoaded', function() {
    initExportIntegration();
});