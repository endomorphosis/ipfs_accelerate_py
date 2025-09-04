/**
 * Kitchen Sink AI Model Testing Interface
 * Main JavaScript application logic
 */

class KitchenSinkApp {
    constructor() {
        this.currentResults = {};
        this.models = [];
        this.init();
    }

    init() {
        console.log('Initializing Kitchen Sink AI Testing Interface...');
        
        // Initialize components
        this.setupAutocomplete();
        this.setupFormHandlers();
        this.setupRangeInputs();
        this.setupModelManager();
        
        // Load initial data
        this.loadModels();
        this.checkSystemStatus();
        
        console.log('Kitchen Sink App initialized');
    }

    // System status check
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            if (response.ok) {
                this.updateStatusMessage('AI components loaded successfully', 'success');
                this.models = data.models || [];
            } else {
                this.updateStatusMessage('Running in demo mode - limited functionality', 'warning');
            }
        } catch (error) {
            this.updateStatusMessage('Error connecting to AI backend', 'danger');
        }
    }

    updateStatusMessage(message, type = 'info') {
        const alertElement = document.getElementById('status-alert');
        const messageElement = document.getElementById('status-message');
        
        alertElement.className = `alert alert-${type} alert-dismissible fade show`;
        messageElement.innerHTML = `<span class="status-indicator status-${type === 'success' ? 'online' : type === 'warning' ? 'demo' : 'offline'}"></span>${message}`;
    }

    // Model autocomplete setup
    setupAutocomplete() {
        $('.model-autocomplete').each((index, element) => {
            $(element).autocomplete({
                source: async (request, response) => {
                    try {
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
                        response([]);
                    }
                },
                minLength: 2,
                select: (event, ui) => {
                    // Show model info
                    this.showModelInfo(ui.item.modelData, $(event.target).closest('.card'));
                }
            });
        });
    }

    showModelInfo(model, container) {
        // Remove existing model info
        container.find('.model-info').remove();
        
        // Create model info display
        const infoHtml = `
            <div class="model-info">
                <h6>${model.model_name}</h6>
                <p><strong>Architecture:</strong> ${model.architecture}</p>
                <p><strong>Type:</strong> ${model.model_type}</p>
                <p><strong>Description:</strong> ${model.description}</p>
                ${model.tags ? `<p><strong>Tags:</strong> ${model.tags.map(tag => `<span class="model-tag">${tag}</span>`).join('')}</p>` : ''}
            </div>
        `;
        
        container.find('.card-body').append(infoHtml);
    }

    // Form handlers setup
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

        // Feedback buttons
        $(document).on('click', '.feedback-btn', (e) => {
            this.handleFeedback($(e.target));
        });

        // Model refresh
        $('#refresh-models').on('click', () => {
            this.loadModels();
        });
    }

    // Range input display updates
    setupRangeInputs() {
        $('#gen-max-length').on('input', (e) => {
            $('#gen-max-length-value').text(e.target.value);
        });

        $('#gen-temperature').on('input', (e) => {
            $('#gen-temperature-value').text(e.target.value);
        });
    }

    // Text Generation Handler
    async handleTextGeneration() {
        const formData = {
            prompt: $('#gen-prompt').val(),
            model_id: $('#gen-model').val() || null,
            max_length: parseInt($('#gen-max-length').val()),
            temperature: parseFloat($('#gen-temperature').val()),
            hardware: $('#gen-hardware').val()
        };

        this.showLoading('generation');

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
            } else {
                this.showError('generation', result.error || 'Generation failed');
            }
        } catch (error) {
            this.hideLoading('generation');
            this.showError('generation', 'Network error occurred');
        }
    }

    displayGenerationResult(result) {
        $('#generation-result').html(result.generated_text);
        
        $('#gen-model-used').text(result.model_used);
        $('#gen-processing-time').text(result.processing_time.toFixed(3));
        $('#gen-token-count').text(result.token_count);
        
        $('#generation-metadata').show();
        $('#generation .feedback-area').show();
    }

    // Text Classification Handler
    async handleTextClassification() {
        const formData = {
            text: $('#class-text').val(),
            model_id: $('#class-model').val() || null,
            hardware: $('#class-hardware').val()
        };

        this.showLoading('classification');

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
            } else {
                this.showError('classification', result.error || 'Classification failed');
            }
        } catch (error) {
            this.hideLoading('classification');
            this.showError('classification', 'Network error occurred');
        }
    }

    displayClassificationResult(result) {
        const resultHtml = `
            <div class="classification-result">
                <h5>Prediction: <span class="badge bg-primary">${result.prediction}</span></h5>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                <p><strong>Model:</strong> ${result.model_used}</p>
                <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s</p>
                
                <div class="classification-scores">
                    <h6>All Scores:</h6>
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
        $('#classification .feedback-area').show();
    }

    // Embedding Generation Handler
    async handleEmbeddingGeneration() {
        const formData = {
            text: $('#embed-text').val(),
            model_id: $('#embed-model').val() || null,
            normalize: $('#embed-normalize').is(':checked')
        };

        this.showLoading('embeddings');

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
            } else {
                this.showError('embeddings', result.error || 'Embedding generation failed');
            }
        } catch (error) {
            this.hideLoading('embeddings');
            this.showError('embeddings', 'Network error occurred');
        }
    }

    displayEmbeddingResult(result) {
        const embeddingHtml = `
            <div class="embedding-result">
                <h5>Embedding Vector</h5>
                <p><strong>Dimensions:</strong> ${result.dimensions}</p>
                <p><strong>Model:</strong> ${result.model_used}</p>
                <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s</p>
                <p><strong>Normalized:</strong> ${result.normalized ? 'Yes' : 'No'}</p>
                
                <div class="embedding-vector">
                    ${result.embedding.map((val, idx) => 
                        `<span class="embedding-dimension" title="Dimension ${idx}">${val.toFixed(4)}</span>`
                    ).join('')}
                </div>
            </div>
        `;
        
        $('#embeddings-result').html(embeddingHtml);
        $('#embeddings .feedback-area').show();
    }

    // Model Recommendation Handler
    async handleModelRecommendation() {
        const formData = {
            task_type: $('#rec-task-type').val(),
            hardware: $('#rec-hardware').val(),
            input_type: $('#rec-input-type').val(),
            output_type: $('#rec-output-type').val()
        };

        this.showLoading('recommendations');

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
            } else {
                this.showError('recommendations', result.error || 'Recommendation failed');
            }
        } catch (error) {
            this.hideLoading('recommendations');
            this.showError('recommendations', 'Network error occurred');
        }
    }

    displayRecommendationResult(result) {
        const confidenceClass = result.confidence_score > 0.8 ? 'confidence-high' : 
                               result.confidence_score > 0.5 ? 'confidence-medium' : 'confidence-low';
        
        const resultHtml = `
            <div class="recommendation-card card">
                <div class="card-header">
                    <h5><i class="fas fa-star"></i> Recommended Model</h5>
                </div>
                <div class="card-body">
                    <h6>${result.model_id}</h6>
                    <p><strong>Confidence:</strong> 
                        <span class="confidence-indicator ${confidenceClass}">
                            ${(result.confidence_score * 100).toFixed(1)}%
                        </span>
                    </p>
                    <p><strong>Predicted Performance:</strong> ${(result.predicted_performance * 100).toFixed(1)}%</p>
                    <p><strong>Reasoning:</strong> ${result.reasoning}</p>
                    
                    <button class="btn btn-primary btn-sm" onclick="app.useRecommendedModel('${result.model_id}')">
                        <i class="fas fa-arrow-right"></i> Use This Model
                    </button>
                </div>
            </div>
        `;
        
        $('#recommendations-result').html(resultHtml);
    }

    // Use recommended model in other tabs
    useRecommendedModel(modelId) {
        // Fill all model inputs with the recommended model
        $('.model-autocomplete').val(modelId);
        
        // Show success message
        this.showSuccessMessage('Model applied to all inference tabs');
        
        // Optionally switch to the first inference tab
        $('#generation-tab').click();
    }

    // Feedback handling
    async handleFeedback(button) {
        const score = parseFloat(button.data('score'));
        const tabPane = button.closest('.tab-pane');
        const tabId = tabPane.attr('id');
        
        // Mark button as active
        button.siblings().removeClass('active');
        button.addClass('active');
        
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
                
                this.showSuccessMessage('Feedback recorded - thank you!');
            } catch (error) {
                console.error('Feedback error:', error);
            }
        }
    }

    // Model Manager
    setupModelManager() {
        // Model search and filtering will be handled here
        $('#model-search').on('input', () => this.filterModels());
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
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    displayModels(models) {
        const tbody = document.getElementById('models-table-body');
        tbody.innerHTML = '';
        
        models.forEach(model => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><code>${model.model_id}</code></td>
                <td>${model.model_name}</td>
                <td><span class="badge bg-secondary">${model.model_type}</span></td>
                <td>${model.architecture}</td>
                <td>
                    ${(model.tags || []).map(tag => `<span class="model-tag">${tag}</span>`).join('')}
                </td>
                <td>
                    <button class="btn btn-outline-primary btn-sm" onclick="app.viewModelDetails('${model.model_id}')">
                        <i class="fas fa-eye"></i> View
                    </button>
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
                model.model_name.toLowerCase().includes(searchTerm);
            
            const matchesType = !typeFilter || model.model_type === typeFilter;
            const matchesArch = !archFilter || model.architecture === archFilter;
            
            return matchesSearch && matchesType && matchesArch;
        });
        
        this.displayModels(filteredModels);
    }

    async viewModelDetails(modelId) {
        try {
            const response = await fetch(`/api/models/${modelId}`);
            const model = await response.json();
            
            if (response.ok) {
                this.showModelDetailsModal(model);
            }
        } catch (error) {
            console.error('Error loading model details:', error);
        }
    }

    showModelDetailsModal(model) {
        // This would show a modal with detailed model information
        // For now, just log the details
        console.log('Model details:', model);
        alert(`Model: ${model.model_name}\nType: ${model.model_type}\nArchitecture: ${model.architecture}\nDescription: ${model.description}`);
    }

    // UI helper methods
    showLoading(section) {
        $(`#${section}-result`).html('<div class="text-center"><div class="spinner-border" role="status"></div><div class="mt-2">Processing...</div></div>');
        $(`#${section}-form button[type="submit"]`).prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status"></span> Processing...');
    }

    hideLoading(section) {
        $(`#${section}-form button[type="submit"]`).prop('disabled', false);
        
        // Restore button text based on section
        const buttonTexts = {
            generation: '<i class="fas fa-play"></i> Generate Text',
            classification: '<i class="fas fa-play"></i> Classify Text',
            embeddings: '<i class="fas fa-play"></i> Generate Embeddings',
            recommendations: '<i class="fas fa-search"></i> Get Recommendation'
        };
        
        $(`#${section}-form button[type="submit"]`).html(buttonTexts[section]);
    }

    showError(section, message) {
        $(`#${section}-result`).html(`<div class="error-message"><i class="fas fa-exclamation-triangle"></i> ${message}</div>`);
    }

    showSuccessMessage(message) {
        // Show a temporary success message
        const alertHtml = `
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                <i class="fas fa-check-circle"></i> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        $('body').prepend(alertHtml);
        
        // Auto-dismiss after 3 seconds
        setTimeout(() => {
            $('.alert-success').alert('close');
        }, 3000);
    }
}

// Initialize the application when DOM is ready
$(document).ready(() => {
    window.app = new KitchenSinkApp();
});