/**
 * Kitchen Sink AI Model Testing Interface
 * Enhanced JavaScript application logic with improved UX
 */

class KitchenSinkApp {
    constructor() {
        this.currentResults = {};
        this.models = [];
        this.notifications = [];
        this.init();
    }

    init() {
        console.log('Initializing Kitchen Sink AI Testing Interface...');
        
        // Initialize components
        this.setupAutocomplete();
        this.setupFormHandlers();
        this.setupRangeInputs();
        this.setupModelManager();
        this.setupKeyboardShortcuts();
        this.setupNotificationSystem();
        this.setupHuggingFaceBrowser();
        
        // Load initial data
        this.loadModels();
        this.checkSystemStatus();
        
        console.log('Kitchen Sink App initialized');
    }

    // Enhanced notification system
    setupNotificationSystem() {
        // Create notification container if it doesn't exist
        if (!document.getElementById('notification-container')) {
            const container = document.createElement('div');
            container.id = 'notification-container';
            container.className = 'notification-container';
            document.body.appendChild(container);
        }
    }

    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type} fade-in`;
        
        const icon = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        }[type] || 'fas fa-info-circle';
        
        notification.innerHTML = `
            <i class="${icon}"></i>
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        const container = document.getElementById('notification-container');
        container.appendChild(notification);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.classList.add('fade-out');
                    setTimeout(() => notification.remove(), 300);
                }
            }, duration);
        }
        
        return notification;
    }

    // Enhanced system status check
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            if (response.ok) {
                const modelCount = data.models ? data.models.length : 0;
                this.updateStatusMessage(
                    `AI components loaded successfully. ${modelCount} models available.`, 
                    'success'
                );
                this.models = data.models || [];
                
                if (modelCount === 0) {
                    this.showNotification(
                        'No models found. Some features may be limited.', 
                        'warning', 
                        8000
                    );
                }
            } else {
                this.updateStatusMessage('Running in demo mode - limited functionality', 'warning');
                this.showNotification(
                    'Running in demo mode. Some AI features may not work.', 
                    'warning', 
                    10000
                );
            }
        } catch (error) {
            this.updateStatusMessage('Error connecting to AI backend', 'danger');
            this.showNotification(
                'Failed to connect to AI backend. Please check your connection.', 
                'error', 
                0
            );
        }
    }

    updateStatusMessage(message, type = 'info') {
        const alertElement = document.getElementById('status-alert');
        const messageElement = document.getElementById('status-message');
        
        if (alertElement && messageElement) {
            alertElement.className = `alert alert-${type} alert-dismissible fade show`;
            const statusIndicator = type === 'success' ? 'online' : type === 'warning' ? 'demo' : 'offline';
            messageElement.innerHTML = `<span class="status-indicator status-${statusIndicator}"></span>${message}`;
        }
    }

    // Enhanced autocomplete with better UX
    setupAutocomplete() {
        $('.model-autocomplete').each((index, element) => {
            $(element).autocomplete({
                source: async (request, response) => {
                    try {
                        // Show loading indicator
                        $(element).addClass('loading');
                        
                        const res = await fetch(`/api/models/search?q=${encodeURIComponent(request.term)}&limit=10`);
                        const data = await res.json();
                        
                        const suggestions = data.models.map(model => ({
                            label: `${model.model_name} (${model.architecture})`,
                            value: model.model_id,
                            description: model.description,
                            modelData: model
                        }));
                        
                        response(suggestions);
                    } catch (error) {
                        console.error('Autocomplete error:', error);
                        this.showNotification('Model search failed', 'error', 3000);
                        response([]);
                    } finally {
                        $(element).removeClass('loading');
                    }
                },
                minLength: 1,
                delay: 300,
                select: (event, ui) => {
                    // Show model info with animation
                    this.showModelInfo(ui.item.modelData, $(event.target).closest('.card'));
                },
                open: function() {
                    $(this).removeClass("ui-corner-all").addClass("ui-corner-top");
                },
                close: function() {
                    $(this).removeClass("ui-corner-top").addClass("ui-corner-all");
                }
            });
            
            // Add visual feedback for focus/blur
            $(element).on('focus', function() {
                $(this).closest('.form-group, .mb-3').addClass('focused');
            }).on('blur', function() {
                $(this).closest('.form-group, .mb-3').removeClass('focused');
            });
        });
    }

    showModelInfo(model, container) {
        // Remove existing model info with animation
        const existingInfo = container.find('.model-info');
        if (existingInfo.length) {
            existingInfo.slideUp(200, function() { $(this).remove(); });
        }
        
        // Create enhanced model info display
        const infoHtml = `
            <div class="model-info" style="display: none;">
                <div class="model-info-header">
                    <h6><i class="fas fa-robot"></i> ${model.model_name}</h6>
                    <span class="model-type-badge">${model.model_type}</span>
                </div>
                <div class="model-info-body">
                    <p><strong>Architecture:</strong> <code>${model.architecture}</code></p>
                    <p><strong>Description:</strong> ${model.description}</p>
                    ${model.tags ? `<div class="model-tags">
                        ${model.tags.map(tag => `<span class="model-tag">${tag}</span>`).join('')}
                    </div>` : ''}
                </div>
            </div>
        `;
        
        container.find('.card-body').append(infoHtml);
        container.find('.model-info').slideDown(300);
    }

    // Enhanced form handlers with better error handling
    setupFormHandlers() {
        // Text Generation Form
        $('#generation-form').on('submit', (e) => {
            e.preventDefault();
            this.handleTextGeneration();
        });

        // Classification Form
        $('#classification-form').on('submit', (e) => {
            e.preventDefault();
            this.handleTextClassification();
        });

        // Embeddings Form
        $('#embeddings-form').on('submit', (e) => {
            e.preventDefault();
            this.handleEmbeddingGeneration();
        });

        // Recommendations Form
        $('#recommendations-form').on('submit', (e) => {
            e.preventDefault();
            this.handleModelRecommendation();
        });

        // Enhanced feedback buttons
        $(document).on('click', '.feedback-btn', (e) => {
            this.handleFeedback($(e.target));
        });

        // Model refresh with loading indicator
        $('#refresh-models').on('click', (e) => {
            e.preventDefault();
            $(e.target).addClass('loading');
            this.loadModels().finally(() => {
                $(e.target).removeClass('loading');
            });
        });
    }

    // Enhanced keyboard shortcuts
    setupKeyboardShortcuts() {
        $(document).on('keydown', (e) => {
            // Ctrl/Cmd + Enter to submit active form
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const activeTab = $('.tab-pane.active');
                const submitButton = activeTab.find('button[type="submit"]');
                if (submitButton.length && !submitButton.prop('disabled')) {
                    submitButton.click();
                    e.preventDefault();
                }
            }
            
            // Escape key to clear results
            if (e.key === 'Escape') {
                $('.result-area').empty().append('<p class="text-muted">Results cleared...</p>');
                $('.metadata-area, .feedback-area').hide();
            }
        });
    }

    // Enhanced range input setup
    setupRangeInputs() {
        $('#gen-max-length').on('input', (e) => {
            $('#gen-max-length-value').text(e.target.value);
        });

        $('#gen-temperature').on('input', (e) => {
            $('#gen-temperature-value').text(parseFloat(e.target.value).toFixed(1));
        });
    }

    // Enhanced text generation with better feedback
    async handleTextGeneration() {
        const formData = {
            prompt: $('#gen-prompt').val(),
            model_id: $('#gen-model').val() || null,
            max_length: parseInt($('#gen-max-length').val()),
            temperature: parseFloat($('#gen-temperature').val()),
            hardware: $('#gen-hardware').val()
        };

        // Validation
        if (!formData.prompt.trim()) {
            this.showNotification('Please enter a text prompt', 'warning', 3000);
            $('#gen-prompt').focus();
            return;
        }

        this.showLoading('generation');
        this.showNotification('Generating text...', 'info', 2000);

        try {
            const response = await fetch('/api/inference/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            this.hideLoading('generation');

            if (response.ok) {
                this.displayGenerationResult(result);
                this.currentResults.generation = { result, formData };
                this.showNotification('Text generated successfully!', 'success', 3000);
            } else {
                this.showError('generation', result.error || 'Generation failed');
                this.showNotification('Text generation failed', 'error', 5000);
            }
        } catch (error) {
            this.hideLoading('generation');
            this.showError('generation', 'Network error occurred');
            this.showNotification('Network error during text generation', 'error', 5000);
        }
    }

    displayGenerationResult(result) {
        const resultHtml = `
            <div class="result-content">
                <div class="generated-text">${result.generated_text}</div>
            </div>
        `;
        
        $('#generation-result').html(resultHtml);
        
        $('#gen-model-used').text(result.model_used);
        $('#gen-processing-time').text(result.processing_time.toFixed(3));
        $('#gen-token-count').text(result.token_count);
        
        $('#generation-metadata').slideDown(300);
        $('#generation .feedback-area').slideDown(300);
    }

    // Enhanced text classification
    async handleTextClassification() {
        const formData = {
            text: $('#class-text').val(),
            model_id: $('#class-model').val() || null,
            hardware: $('#class-hardware').val()
        };

        if (!formData.text.trim()) {
            this.showNotification('Please enter text to classify', 'warning', 3000);
            $('#class-text').focus();
            return;
        }

        this.showLoading('classification');
        this.showNotification('Classifying text...', 'info', 2000);

        try {
            const response = await fetch('/api/inference/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            this.hideLoading('classification');

            if (response.ok) {
                this.displayClassificationResult(result);
                this.currentResults.classification = { result, formData };
                this.showNotification('Text classified successfully!', 'success', 3000);
            } else {
                this.showError('classification', result.error || 'Classification failed');
                this.showNotification('Text classification failed', 'error', 5000);
            }
        } catch (error) {
            this.hideLoading('classification');
            this.showError('classification', 'Network error occurred');
            this.showNotification('Network error during classification', 'error', 5000);
        }
    }

    displayClassificationResult(result) {
        const resultHtml = `
            <div class="classification-result">
                <div class="prediction-header">
                    <h5>Prediction: <span class="badge bg-primary prediction-badge">${result.prediction}</span></h5>
                    <div class="confidence-display">
                        <strong>Confidence:</strong> 
                        <span class="confidence-value">${(result.confidence * 100).toFixed(1)}%</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="classification-metadata">
                    <p><strong>Model:</strong> <code>${result.model_used}</code></p>
                    <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s</p>
                </div>
                
                <div class="classification-scores">
                    <h6><i class="fas fa-chart-bar"></i> All Scores:</h6>
                    ${Object.entries(result.all_scores).map(([label, score]) => `
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${score * 100}%"></div>
                            <div class="score-label">${label}</div>
                            <div class="score-value">${(score * 100).toFixed(1)}%</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        $('#classification-result').html(resultHtml);
        $('#classification .feedback-area').slideDown(300);
    }

    // Enhanced embedding generation
    async handleEmbeddingGeneration() {
        const formData = {
            text: $('#embed-text').val(),
            model_id: $('#embed-model').val() || null,
            normalize: $('#embed-normalize').is(':checked')
        };

        if (!formData.text.trim()) {
            this.showNotification('Please enter text to embed', 'warning', 3000);
            $('#embed-text').focus();
            return;
        }

        this.showLoading('embeddings');
        this.showNotification('Generating embeddings...', 'info', 2000);

        try {
            const response = await fetch('/api/inference/embed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            this.hideLoading('embeddings');

            if (response.ok) {
                this.displayEmbeddingResult(result);
                this.currentResults.embeddings = { result, formData };
                this.showNotification('Embeddings generated successfully!', 'success', 3000);
            } else {
                this.showError('embeddings', result.error || 'Embedding generation failed');
                this.showNotification('Embedding generation failed', 'error', 5000);
            }
        } catch (error) {
            this.hideLoading('embeddings');
            this.showError('embeddings', 'Network error occurred');
            this.showNotification('Network error during embedding generation', 'error', 5000);
        }
    }

    displayEmbeddingResult(result) {
        const embeddingHtml = `
            <div class="embedding-result">
                <div class="embedding-header">
                    <h5><i class="fas fa-vector-square"></i> Embedding Vector</h5>
                    <div class="embedding-stats">
                        <span class="stat"><strong>Dimensions:</strong> ${result.dimensions}</span>
                        <span class="stat"><strong>Normalized:</strong> ${result.normalized ? 'Yes' : 'No'}</span>
                    </div>
                </div>
                
                <div class="embedding-metadata">
                    <p><strong>Model:</strong> <code>${result.model_used}</code></p>
                    <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s</p>
                </div>
                
                <div class="embedding-vector">
                    <div class="vector-header">
                        <span>Vector Values:</span>
                        <button class="btn btn-sm btn-outline-secondary" onclick="app.copyEmbedding(${JSON.stringify(result.embedding).replace(/"/g, '&quot;')})">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                    </div>
                    <div class="vector-display">
                        ${result.embedding.map((val, idx) => 
                            `<span class="embedding-dimension" title="Dimension ${idx}: ${val}">${val.toFixed(4)}</span>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
        
        $('#embeddings-result').html(embeddingHtml);
        $('#embeddings .feedback-area').slideDown(300);
    }

    copyEmbedding(embedding) {
        navigator.clipboard.writeText(JSON.stringify(embedding)).then(() => {
            this.showNotification('Embedding vector copied to clipboard!', 'success', 2000);
        }).catch(() => {
            this.showNotification('Failed to copy embedding vector', 'error', 3000);
        });
    }

    // Enhanced model recommendation
    async handleModelRecommendation() {
        const formData = {
            task_type: $('#rec-task-type').val(),
            hardware: $('#rec-hardware').val(),
            input_type: $('#rec-input-type').val(),
            output_type: $('#rec-output-type').val()
        };

        this.showLoading('recommendations');
        this.showNotification('Getting model recommendation...', 'info', 2000);

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            this.hideLoading('recommendations');

            if (response.ok) {
                this.displayRecommendationResult(result);
                this.showNotification('Recommendation generated!', 'success', 3000);
            } else {
                this.showError('recommendations', result.error || 'Recommendation failed');
                this.showNotification('Failed to get recommendation', 'error', 5000);
            }
        } catch (error) {
            this.hideLoading('recommendations');
            this.showError('recommendations', 'Network error occurred');
            this.showNotification('Network error during recommendation', 'error', 5000);
        }
    }

    displayRecommendationResult(result) {
        const confidenceClass = result.confidence_score > 0.8 ? 'confidence-high' : 
                               result.confidence_score > 0.5 ? 'confidence-medium' : 'confidence-low';
        
        const resultHtml = `
            <div class="recommendation-card card border-success">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0"><i class="fas fa-star"></i> Recommended Model</h5>
                </div>
                <div class="card-body">
                    <h6 class="model-id">${result.model_id}</h6>
                    
                    <div class="recommendation-metrics">
                        <div class="metric">
                            <label>Confidence:</label>
                            <span class="confidence-indicator ${confidenceClass}">
                                ${(result.confidence_score * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div class="metric">
                            <label>Predicted Performance:</label>
                            <span class="performance-score">
                                ${(result.predicted_performance * 100).toFixed(1)}%
                            </span>
                        </div>
                    </div>
                    
                    <div class="reasoning">
                        <label><i class="fas fa-lightbulb"></i> Reasoning:</label>
                        <p>${result.reasoning}</p>
                    </div>
                    
                    <div class="action-buttons">
                        <button class="btn btn-primary" onclick="app.useRecommendedModel('${result.model_id}')">
                            <i class="fas fa-arrow-right"></i> Use This Model
                        </button>
                        <button class="btn btn-outline-secondary" onclick="app.getModelDetails('${result.model_id}')">
                            <i class="fas fa-info"></i> Model Details
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        $('#recommendations-result').html(resultHtml);
    }

    // Enhanced use recommended model
    useRecommendedModel(modelId) {
        // Fill all model inputs with the recommended model
        $('.model-autocomplete').val(modelId);
        
        // Trigger change events to update UI
        $('.model-autocomplete').trigger('change');
        
        // Show success notification
        this.showNotification(`Model "${modelId}" applied to all inference tabs`, 'success', 4000);
        
        // Optionally switch to the first inference tab
        $('#generation-tab').click();
    }

    async getModelDetails(modelId) {
        try {
            const response = await fetch(`/api/models/${modelId}`);
            const model = await response.json();
            
            if (response.ok) {
                this.showModelDetailsModal(model);
            } else {
                this.showNotification('Failed to load model details', 'error', 3000);
            }
        } catch (error) {
            this.showNotification('Network error loading model details', 'error', 3000);
        }
    }

    showModelDetailsModal(model) {
        const modalHtml = `
            <div class="modal fade" id="modelDetailsModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="fas fa-robot"></i> ${model.model_name}
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="model-details">
                                <div class="detail-section">
                                    <h6>Basic Information</h6>
                                    <table class="table table-sm">
                                        <tr><td><strong>Model ID:</strong></td><td><code>${model.model_id}</code></td></tr>
                                        <tr><td><strong>Type:</strong></td><td>${model.model_type}</td></tr>
                                        <tr><td><strong>Architecture:</strong></td><td>${model.architecture}</td></tr>
                                        <tr><td><strong>Description:</strong></td><td>${model.description}</td></tr>
                                    </table>
                                </div>
                                
                                ${model.tags ? `
                                <div class="detail-section">
                                    <h6>Tags</h6>
                                    <div class="model-tags">
                                        ${model.tags.map(tag => `<span class="model-tag">${tag}</span>`).join('')}
                                    </div>
                                </div>` : ''}
                                
                                ${model.input_spec ? `
                                <div class="detail-section">
                                    <h6>Input Specification</h6>
                                    <p><strong>Data Type:</strong> ${model.input_spec.data_type}</p>
                                    <p><strong>Description:</strong> ${model.input_spec.description}</p>
                                </div>` : ''}
                                
                                ${model.output_spec ? `
                                <div class="detail-section">
                                    <h6>Output Specification</h6>
                                    <p><strong>Data Type:</strong> ${model.output_spec.data_type}</p>
                                    <p><strong>Description:</strong> ${model.output_spec.description}</p>
                                </div>` : ''}
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary" onclick="app.useRecommendedModel('${model.model_id}')" data-bs-dismiss="modal">
                                Use This Model
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal
        $('#modelDetailsModal').remove();
        
        // Add new modal
        $('body').append(modalHtml);
        
        // Show modal
        $('#modelDetailsModal').modal('show');
    }

    // Enhanced feedback handling
    async handleFeedback(button) {
        const score = parseFloat(button.data('score'));
        const tabPane = button.closest('.tab-pane');
        const tabId = tabPane.attr('id');
        
        // Animate button selection
        button.siblings().removeClass('active').removeClass('btn-primary btn-success btn-danger');
        button.addClass('active');
        
        // Apply appropriate button style based on score
        if (score >= 0.7) {
            button.removeClass('btn-outline-success').addClass('btn-success');
        } else if (score >= 0.4) {
            button.removeClass('btn-outline-warning').addClass('btn-warning');
        } else {
            button.removeClass('btn-outline-danger').addClass('btn-danger');
        }
        
        if (this.currentResults[tabId]) {
            const { result, formData } = this.currentResults[tabId];
            
            try {
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_id: result.model_used,
                        task_type: tabId,
                        score: score,
                        hardware: formData.hardware
                    })
                });
                
                this.showNotification('Feedback recorded - thank you!', 'success', 3000);
            } catch (error) {
                console.error('Feedback error:', error);
                this.showNotification('Failed to record feedback', 'error', 3000);
            }
        }
    }

    // Enhanced model manager
    setupModelManager() {
        $('#model-search').on('input', debounce(() => this.filterModels(), 300));
        $('#model-type-filter').on('change', () => this.filterModels());
        $('#model-arch-filter').on('change', () => this.filterModels());
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            if (response.ok) {
                this.models = data.models || [];
                this.displayModels(this.models);
                
                if (this.models.length === 0) {
                    this.showNotification('No models available. Running in demo mode.', 'warning', 8000);
                }
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.showNotification('Failed to load models', 'error', 5000);
        }
    }

    displayModels(models) {
        const tbody = document.getElementById('models-table-body');
        
        if (models.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center text-muted py-4">
                        <i class="fas fa-database fa-2x mb-2"></i><br>
                        No models available. Try refreshing or check your configuration.
                    </td>
                </tr>
            `;
            return;
        }
        
        tbody.innerHTML = '';
        
        models.forEach(model => {
            const row = document.createElement('tr');
            row.className = 'model-row';
            row.innerHTML = `
                <td><code class="model-id-code">${model.model_id}</code></td>
                <td>${model.model_name}</td>
                <td><span class="badge bg-secondary">${model.model_type}</span></td>
                <td><span class="architecture-tag">${model.architecture}</span></td>
                <td>
                    ${(model.tags || []).map(tag => `<span class="model-tag">${tag}</span>`).join('')}
                </td>
                <td>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary" onclick="app.getModelDetails('${model.model_id}')" title="View Details">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-outline-success" onclick="app.useRecommendedModel('${model.model_id}')" title="Use Model">
                            <i class="fas fa-play"></i>
                        </button>
                    </div>
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    filterModels() {
        const searchTerm = $('#model-search').val().toLowerCase();
        const typeFilter = $('#model-type-filter').val();
        const archFilter = $('#model-arch-filter').val();
        
        const filteredModels = this.models.filter(model => {
            const matchesSearch = !searchTerm || 
                model.model_id.toLowerCase().includes(searchTerm) ||
                model.model_name.toLowerCase().includes(searchTerm) ||
                model.description.toLowerCase().includes(searchTerm);
            
            const matchesType = !typeFilter || model.model_type === typeFilter;
            const matchesArch = !archFilter || model.architecture === archFilter;
            
            return matchesSearch && matchesType && matchesArch;
        });
        
        this.displayModels(filteredModels);
        
        // Update filter info
        if (filteredModels.length !== this.models.length) {
            this.showNotification(
                `Showing ${filteredModels.length} of ${this.models.length} models`, 
                'info', 
                2000
            );
        }
    }

    // Enhanced UI helper methods
    showLoading(section) {
        const resultArea = $(`#${section}-result`);
        const submitButton = $(`#${section}-form button[type="submit"]`);
        
        resultArea.html(`
            <div class="loading-state">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="loading-text mt-2">Processing your request...</div>
            </div>
        `);
        
        submitButton.prop('disabled', true)
                   .html('<span class="spinner-border spinner-border-sm me-2" role="status"></span>Processing...')
                   .addClass('loading');
    }

    hideLoading(section) {
        const submitButton = $(`#${section}-form button[type="submit"]`);
        submitButton.prop('disabled', false).removeClass('loading');
        
        // Restore button text based on section
        const buttonTexts = {
            generation: '<i class="fas fa-play"></i> Generate Text',
            classification: '<i class="fas fa-play"></i> Classify Text',
            embeddings: '<i class="fas fa-play"></i> Generate Embeddings',
            recommendations: '<i class="fas fa-search"></i> Get Recommendation'
        };
        
        submitButton.html(buttonTexts[section]);
    }

    showError(section, message) {
        $(`#${section}-result`).html(`
            <div class="error-state">
                <div class="error-icon">
                    <i class="fas fa-exclamation-triangle fa-2x text-danger"></i>
                </div>
                <div class="error-message mt-2">
                    <strong>Error:</strong> ${message}
                </div>
                <div class="error-actions mt-3">
                    <button class="btn btn-outline-primary btn-sm" onclick="location.reload()">
                        <i class="fas fa-redo"></i> Refresh Page
                    </button>
                </div>
            </div>
        `);
    }

    // HuggingFace Browser functionality
    setupHuggingFaceBrowser() {
        console.log('Setting up HuggingFace browser...');
        
        // Search button handler
        $('#hf-search-btn').on('click', () => {
            this.searchHuggingFaceModels();
        });
        
        // Enter key search
        $('#hf-search-query').on('keypress', (e) => {
            if (e.which === 13) {
                this.searchHuggingFaceModels();
            }
        });
        
        // Quick search buttons
        $('.hf-quick-search').on('click', (e) => {
            const $btn = $(e.currentTarget);
            const query = $btn.data('query');
            const task = $btn.data('task');
            
            $('#hf-search-query').val(query);
            $('#hf-task-filter').val(task);
            this.searchHuggingFaceModels();
        });
        
        // Task links in sidebar
        $('.hf-task-link').on('click', (e) => {
            const task = $(e.currentTarget).data('task');
            $('#hf-task-filter').val(task);
            $('#hf-search-query').val('');
            this.searchHuggingFaceModels();
        });
        
        // Add to manager button
        $('#hf-add-to-manager').on('click', () => {
            this.addHuggingFaceModelToManager();
        });
        
        // Load initial stats
        this.loadHuggingFaceStats();
    }
    
    async searchHuggingFaceModels() {
        console.log('Searching HuggingFace models...');
        
        const query = $('#hf-search-query').val().trim();
        const taskFilter = $('#hf-task-filter').val();
        const sortBy = $('#hf-sort-by').val();
        const limit = 20;
        
        // Show loading
        $('#hf-search-results').html(`
            <div class="text-center p-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Searching HuggingFace Hub...</p>
            </div>
        `);
        
        try {
            const response = await fetch('/api/hf/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    task_filter: taskFilter,
                    sort_by: sortBy,
                    limit: limit
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayHuggingFaceResults(data.models);
                this.updateHuggingFaceStats(data.total);
                this.showNotification(`Found ${data.models.length} models`, 'success');
            } else {
                throw new Error(data.error || 'Search failed');
            }
            
        } catch (error) {
            console.error('HuggingFace search error:', error);
            $('#hf-search-results').html(`
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    Error searching models: ${error.message}
                </div>
            `);
            this.showNotification(`Search failed: ${error.message}`, 'error');
        }
    }
    
    displayHuggingFaceResults(models) {
        if (!models || models.length === 0) {
            $('#hf-search-results').html(`
                <div class="text-center text-muted p-4">
                    <i class="fas fa-search fa-3x mb-3"></i>
                    <p>No models found</p>
                    <p class="small">Try adjusting your search criteria</p>
                </div>
            `);
            return;
        }
        
        let html = `
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Task</th>
                        <th>Downloads</th>
                        <th>Likes</th>
                        <th>License</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        models.forEach(model => {
            const downloads = model.downloads ? model.downloads.toLocaleString() : '0';
            const likes = model.likes ? model.likes.toLocaleString() : '0';
            const description = model.description ? 
                (model.description.length > 100 ? model.description.substring(0, 100) + '...' : model.description) : 
                'No description available';
            
            html += `
                <tr class="hf-model-row" data-model-id="${model.model_id}">
                    <td>
                        <div>
                            <strong>${model.model_id}</strong>
                            <br>
                            <small class="text-muted">${description}</small>
                            <br>
                            <div class="mt-1">
                                ${model.tags.slice(0, 3).map(tag => 
                                    `<span class="badge bg-secondary me-1">${tag}</span>`
                                ).join('')}
                            </div>
                        </div>
                    </td>
                    <td>
                        ${model.pipeline_tag ? 
                            `<span class="badge bg-primary">${model.pipeline_tag}</span>` : 
                            '<span class="text-muted">Unknown</span>'
                        }
                    </td>
                    <td>${downloads}</td>
                    <td>${likes}</td>
                    <td>
                        ${model.license ? 
                            `<span class="badge bg-info">${model.license}</span>` : 
                            '<span class="text-muted">Unknown</span>'
                        }
                    </td>
                    <td>
                        <button class="btn btn-outline-primary btn-sm hf-view-details" data-model-id="${model.model_id}">
                            <i class="fas fa-eye"></i> Details
                        </button>
                    </td>
                </tr>
            `;
        });
        
        html += '</tbody></table>';
        $('#hf-search-results').html(html);
        
        // Add click handlers for details buttons
        $('.hf-view-details').on('click', (e) => {
            const modelId = $(e.currentTarget).data('model-id');
            this.loadHuggingFaceModelDetails(modelId);
        });
        
        // Add click handlers for model rows
        $('.hf-model-row').on('click', (e) => {
            if (!$(e.target).hasClass('btn') && !$(e.target).closest('.btn').length) {
                const modelId = $(e.currentTarget).data('model-id');
                this.loadHuggingFaceModelDetails(modelId);
            }
        });
    }
    
    async loadHuggingFaceModelDetails(modelId) {
        console.log('Loading model details for:', modelId);
        
        try {
            // Show loading in details panel
            $('#hf-model-details').show();
            $('#hf-model-info').html(`
                <div class="text-center">
                    <div class="spinner-border spinner-border-sm text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="small mt-2">Loading model details...</p>
                </div>
            `);
            
            const response = await fetch(`/api/hf/model/${encodeURIComponent(modelId)}`);
            const model = await response.json();
            
            if (response.ok) {
                this.displayHuggingFaceModelDetails(model);
                this.currentHfModel = model;
            } else {
                throw new Error(model.error || 'Failed to load model details');
            }
            
        } catch (error) {
            console.error('Error loading model details:', error);
            $('#hf-model-info').html(`
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    Error loading details: ${error.message}
                </div>
            `);
        }
    }
    
    displayHuggingFaceModelDetails(model) {
        const formatNumber = (num) => num ? num.toLocaleString() : '0';
        const formatDate = (dateStr) => {
            if (!dateStr) return 'Unknown';
            try {
                return new Date(dateStr).toLocaleDateString();
            } catch {
                return 'Unknown';
            }
        };
        
        let html = `
            <div class="model-details">
                <h6>${model.model_id}</h6>
                <p class="text-muted small">${model.description || 'No description available'}</p>
                
                <div class="details-section">
                    <strong>Author:</strong> ${model.author || 'Unknown'}<br>
                    <strong>Task:</strong> ${model.pipeline_tag ? 
                        `<span class="badge bg-primary">${model.pipeline_tag}</span>` : 
                        'Unknown'}<br>
                    <strong>Library:</strong> ${model.library_name || 'Unknown'}<br>
                    <strong>License:</strong> ${model.license ? 
                        `<span class="badge bg-info">${model.license}</span>` : 
                        'Unknown'}
                </div>
                
                <div class="details-section mt-3">
                    <div class="row text-center">
                        <div class="col-6">
                            <div class="border rounded p-2">
                                <div class="h6 mb-0">${formatNumber(model.downloads)}</div>
                                <small class="text-muted">Downloads</small>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="border rounded p-2">
                                <div class="h6 mb-0">${formatNumber(model.likes)}</div>
                                <small class="text-muted">Likes</small>
                            </div>
                        </div>
                    </div>
                </div>
        `;
        
        // Add repository information if available
        if (model.repository) {
            html += `
                <div class="details-section mt-3">
                    <strong>Repository:</strong><br>
                    <small>
                        Files: ${model.repository.total_files}<br>
                        Size: ${this.formatBytes(model.repository.total_size)}<br>
                        IPFS: ${model.repository.ipfs_enabled ? '✅ Enabled' : '❌ Disabled'}
                    </small>
                </div>
            `;
        }
        
        // Add tags
        if (model.tags && model.tags.length > 0) {
            html += `
                <div class="details-section mt-3">
                    <strong>Tags:</strong><br>
                    <div class="mt-1">
                        ${model.tags.slice(0, 10).map(tag => 
                            `<span class="badge bg-secondary me-1 mb-1">${tag}</span>`
                        ).join('')}
                        ${model.tags.length > 10 ? 
                            `<span class="text-muted small">+${model.tags.length - 10} more</span>` : 
                            ''
                        }
                    </div>
                </div>
            `;
        }
        
        // Add dates
        html += `
            <div class="details-section mt-3">
                <strong>Dates:</strong><br>
                <small>
                    Created: ${formatDate(model.created_at)}<br>
                    Modified: ${formatDate(model.last_modified)}
                </small>
            </div>
        `;
        
        html += '</div>';
        
        $('#hf-model-info').html(html);
    }
    
    async addHuggingFaceModelToManager() {
        if (!this.currentHfModel) {
            this.showNotification('No model selected', 'warning');
            return;
        }
        
        const modelId = this.currentHfModel.model_id;
        
        try {
            // Show loading on button
            const $btn = $('#hf-add-to-manager');
            const originalText = $btn.html();
            $btn.html('<i class="fas fa-spinner fa-spin"></i> Adding...');
            $btn.prop('disabled', true);
            
            const response = await fetch('/api/hf/add-to-manager', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_id: modelId
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.showNotification(`Model ${modelId} added to manager successfully!`, 'success');
                // Refresh the models list
                this.loadModels();
            } else {
                throw new Error(data.error || 'Failed to add model');
            }
            
        } catch (error) {
            console.error('Error adding model to manager:', error);
            this.showNotification(`Failed to add model: ${error.message}`, 'error');
        } finally {
            // Restore button
            const $btn = $('#hf-add-to-manager');
            $btn.html('<i class="fas fa-plus"></i> Add to Model Manager');
            $btn.prop('disabled', false);
        }
    }
    
    async loadHuggingFaceStats() {
        try {
            const response = await fetch('/api/hf/stats');
            const stats = await response.json();
            
            if (response.ok) {
                $('#hf-cached-models').text(stats.total_models || 0);
                // Calculate cache coverage (example)
                const coverage = Math.min((stats.total_models || 0) / 100 * 100, 100);
                $('#hf-cache-progress').css('width', `${coverage}%`);
            }
        } catch (error) {
            console.log('Could not load HF stats:', error);
        }
    }
    
    updateHuggingFaceStats(totalResults) {
        $('#hf-total-results').text(totalResults || 0);
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize the application when DOM is ready
$(document).ready(() => {
    window.app = new KitchenSinkApp();
});