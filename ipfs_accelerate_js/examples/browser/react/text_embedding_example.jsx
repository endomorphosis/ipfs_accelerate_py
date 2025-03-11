/**
 * Example React Component using IPFS Accelerate JavaScript SDK
 */

import React, { useState } from 'react';
import { useModel, useHardwareInfo, useP2PStatus } from "./react_hooks";

/**
 * Text Embedding Component using IPFS Accelerate SDK
 */
function TextEmbeddingComponent() {
  // React hook for easy model loading
  const { model, status, error, switchBackend } = useModel({
    modelId: 'bert-base-uncased',
    autoHardwareSelection: true,
    fallbackOrder: ['webgpu', 'webnn', 'wasm']
  });

  // Hook for hardware information
  const { capabilities, isReady, optimalBackend } = useHardwareInfo();

  // Hook for P2P network status
  const { isEnabled, peerCount, networkHealth, enableP2P, disableP2P } = useP2PStatus();

  // Component state
  const [input, setInput] = useState('');
  const [embedding, setEmbedding] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [processingTime, setProcessingTime] = useState(0);

  // Generate embedding function
  async function generateEmbedding() {
    if (model && input && !processing) {
      setProcessing(true);
      
      try {
        const startTime = performance.now();
        const result = await model.getEmbeddings(input);
        const endTime = performance.now();
        
        setEmbedding(result);
        setProcessingTime(endTime - startTime);
      } catch (err) {
        console.error('Error generating embedding:', err);
      } finally {
        setProcessing(false);
      }
    }
  }

  // Handle backend switch
  async function handleBackendSwitch(newBackend) {
    if (model) {
      try {
        await switchBackend(newBackend);
      } catch (err) {
        console.error('Error switching backend:', err);
      }
    }
  }

  return (
    <div className="embedding-component">
      <div className="hardware-info">
        <h2>Hardware Status</h2>
        {isReady ? (
          <div>
            <p>Browser: {capabilities.browserName}</p>
            <p>WebGPU: {capabilities.webgpu.supported ? '✅' : '❌'}</p>
            <p>WebNN: {capabilities.webnn.supported ? '✅' : '❌'}</p>
            <p>WebAssembly: {capabilities.wasm.supported ? '✅' : '❌'}</p>
            <p>Optimal backend: {optimalBackend}</p>
          </div>
        ) : (
          <p>Detecting hardware capabilities...</p>
        )}
      </div>
      
      <div className="p2p-status">
        <h2>P2P Network</h2>
        <div className="status">
          <p>Status: {isEnabled ? '✅ Enabled' : '❌ Disabled'}</p>
          {isEnabled && (
            <div>
              <p>Connected peers: {peerCount}</p>
              <p>Network health: {(networkHealth * 100).toFixed(0)}%</p>
            </div>
          )}
        </div>
        <div className="controls">
          <button onClick={enableP2P} disabled={isEnabled}>Enable P2P</button>
          <button onClick={disableP2P} disabled={!isEnabled}>Disable P2P</button>
        </div>
      </div>
      
      <div className="model-section">
        <h2>Text Embedding</h2>
        <div className="backend-selector">
          <h3>Select Backend</h3>
          <div className="buttons">
            <button 
              onClick={() => handleBackendSwitch('webgpu')} 
              disabled={!capabilities?.webgpu?.supported || status !== 'loaded'}
              className={model?.getBackend() === 'webgpu' ? 'active' : ''}
            >
              WebGPU
            </button>
            <button 
              onClick={() => handleBackendSwitch('webnn')} 
              disabled={!capabilities?.webnn?.supported || status !== 'loaded'}
              className={model?.getBackend() === 'webnn' ? 'active' : ''}
            >
              WebNN
            </button>
            <button 
              onClick={() => handleBackendSwitch('wasm')} 
              disabled={status !== 'loaded'}
              className={model?.getBackend() === 'wasm' ? 'active' : ''}
            >
              WebAssembly
            </button>
          </div>
        </div>
        
        <div className="input-section">
          <h3>Input Text</h3>
          <textarea 
            value={input} 
            onChange={e => setInput(e.target.value)} 
            placeholder="Enter text to embed"
            rows={5}
          />
          <button 
            onClick={generateEmbedding} 
            disabled={status !== 'loaded' || !input || processing}
          >
            {processing ? 'Processing...' : 'Generate Embedding'}
          </button>
        </div>
        
        {status === 'loading' && <p className="status-message">Loading model...</p>}
        {error && <p className="error-message">Error: {error.message}</p>}
        
        {embedding && (
          <div className="result-section">
            <h3>Result</h3>
            <p>Embedding dimensions: {embedding.length}</p>
            <p>Processing time: {processingTime.toFixed(2)} ms</p>
            <p>Backend used: {model?.getBackend()}</p>
            <div className="embedding-visualization">
              <h4>Embedding Visualization</h4>
              <div className="vector-preview">
                {/* Simple visualization of the first 20 values */}
                {Array.from({length: Math.min(20, embedding.length)}).map((_, i) => (
                  <div 
                    key={i} 
                    className="vector-bar" 
                    style={{
                      width: `${Math.abs(embedding[i] * 100)}%`,
                      backgroundColor: embedding[i] >= 0 ? '#4CAF50' : '#F44336'
                    }}
                    title={`Dimension ${i}: ${embedding[i]}`}
                  />
                ))}
              </div>
              <p className="note">Showing first 20 of {embedding.length} dimensions</p>
            </div>
          </div>
        )}
      </div>
      
      <style jsx>{`
        .embedding-component {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
          font-family: Arial, sans-serif;
        }
        
        h2 {
          color: #333;
          border-bottom: 1px solid #eee;
          padding-bottom: 10px;
        }
        
        .hardware-info, .p2p-status, .model-section {
          margin-bottom: 30px;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 5px;
          background-color: #f9f9f9;
        }
        
        .backend-selector .buttons {
          display: flex;
          gap: 10px;
          margin-bottom: 15px;
        }
        
        button {
          padding: 8px 16px;
          background-color: #4CAF50;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
        }
        
        button:hover:not(:disabled) {
          background-color: #45a049;
        }
        
        button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
        
        button.active {
          background-color: #2196F3;
        }
        
        textarea {
          width: 100%;
          padding: 10px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-family: Arial, sans-serif;
          margin-bottom: 10px;
        }
        
        .status-message {
          color: #2196F3;
        }
        
        .error-message {
          color: #F44336;
        }
        
        .result-section {
          margin-top: 20px;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 5px;
          background-color: #f0f8ff;
        }
        
        .embedding-visualization {
          margin-top: 15px;
        }
        
        .vector-preview {
          display: flex;
          flex-direction: column;
          gap: 3px;
          margin: 10px 0;
        }
        
        .vector-bar {
          height: 10px;
          min-width: 1px;
          border-radius: 2px;
        }
        
        .note {
          font-size: 12px;
          color: #666;
          font-style: italic;
        }
      `}</style>
    </div>
  );
}

/**
 * Image Classification Component using IPFS Accelerate SDK
 */
function ImageClassificationComponent() {
  // React hook for model loading
  const { model, status, error } = useModel({
    modelId: 'vit-base-patch16-224',
    modelType: 'vision',
    autoHardwareSelection: true
  });

  // Component state
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [processingTime, setProcessingTime] = useState(0);

  // Handle image upload
  function handleImageUpload(e) {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        setSelectedImage(file);
        setImagePreview(event.target.result);
        setPredictions(null);
      };
      
      reader.readAsDataURL(file);
    }
  }

  // Process image function
  async function processImage() {
    if (model && selectedImage && !processing) {
      setProcessing(true);
      
      try {
        const startTime = performance.now();
        const result = await model.processImage(imagePreview);
        const endTime = performance.now();
        
        setPredictions(result.classPredictions);
        setProcessingTime(endTime - startTime);
      } catch (err) {
        console.error('Error processing image:', err);
      } finally {
        setProcessing(false);
      }
    }
  }

  return (
    <div className="image-component">
      <h2>Image Classification</h2>
      
      <div className="image-section">
        <h3>Upload Image</h3>
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload} 
          disabled={status !== 'loaded' || processing}
        />
        
        {imagePreview && (
          <div className="image-preview">
            <img src={imagePreview} alt="Preview" />
          </div>
        )}
        
        <button 
          onClick={processImage} 
          disabled={status !== 'loaded' || !selectedImage || processing}
        >
          {processing ? 'Processing...' : 'Classify Image'}
        </button>
      </div>
      
      {status === 'loading' && <p className="status-message">Loading model...</p>}
      {error && <p className="error-message">Error: {error.message}</p>}
      
      {predictions && (
        <div className="result-section">
          <h3>Results</h3>
          <p>Processing time: {processingTime.toFixed(2)} ms</p>
          <p>Backend used: {model?.getBackend()}</p>
          
          <div className="predictions">
            <h4>Top Predictions</h4>
            <ul>
              {predictions.map((prediction, index) => (
                <li key={index}>
                  <span className="prediction-label">{prediction.label}</span>
                  <div className="prediction-bar-container">
                    <div 
                      className="prediction-bar" 
                      style={{width: `${prediction.score * 100}%`}}
                    />
                  </div>
                  <span className="prediction-score">{(prediction.score * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      
      <style jsx>{`
        .image-component {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
          font-family: Arial, sans-serif;
        }
        
        h2 {
          color: #333;
          border-bottom: 1px solid #eee;
          padding-bottom: 10px;
        }
        
        .image-section {
          margin-bottom: 30px;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 5px;
          background-color: #f9f9f9;
        }
        
        .image-preview {
          margin: 15px 0;
          max-width: 100%;
        }
        
        .image-preview img {
          max-width: 100%;
          max-height: 300px;
          border-radius: 5px;
        }
        
        button {
          padding: 8px 16px;
          background-color: #4CAF50;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          margin-top: 10px;
        }
        
        button:hover:not(:disabled) {
          background-color: #45a049;
        }
        
        button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
        
        .status-message {
          color: #2196F3;
        }
        
        .error-message {
          color: #F44336;
        }
        
        .result-section {
          margin-top: 20px;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 5px;
          background-color: #f0f8ff;
        }
        
        .predictions ul {
          list-style: none;
          padding: 0;
        }
        
        .predictions li {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
        }
        
        .prediction-label {
          width: 120px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          font-weight: bold;
        }
        
        .prediction-bar-container {
          flex-grow: 1;
          height: 20px;
          background-color: #f0f0f0;
          border-radius: 3px;
          margin: 0 10px;
        }
        
        .prediction-bar {
          height: 100%;
          background-color: #4CAF50;
          border-radius: 3px;
        }
        
        .prediction-score {
          width: 50px;
          text-align: right;
          font-weight: bold;
        }
      `}</style>
    </div>
  );
}

/**
 * Main App component combining all examples
 */
export default function App() {
  const [activeTab, setActiveTab] = useState('text');
  
  return (
    <div className="app">
      <header>
        <h1>IPFS Accelerate JavaScript SDK Demo</h1>
        <p>Accelerate AI models in the browser with WebGPU and WebNN</p>
      </header>
      
      <div className="tabs">
        <button 
          onClick={() => setActiveTab('text')}
          className={activeTab === 'text' ? 'active' : ''}
        >
          Text Embedding
        </button>
        <button 
          onClick={() => setActiveTab('image')}
          className={activeTab === 'image' ? 'active' : ''}
        >
          Image Classification
        </button>
      </div>
      
      <div className="tab-content">
        {activeTab === 'text' && <TextEmbeddingComponent />}
        {activeTab === 'image' && <ImageClassificationComponent />}
      </div>
      
      <footer>
        <p>IPFS Accelerate JavaScript SDK - Version 0.4.0</p>
      </footer>
      
      <style jsx>{`
        .app {
          max-width: 900px;
          margin: 0 auto;
          padding: 20px;
          font-family: Arial, sans-serif;
        }
        
        header {
          text-align: center;
          margin-bottom: 30px;
        }
        
        h1 {
          color: #333;
        }
        
        .tabs {
          display: flex;
          justify-content: center;
          margin-bottom: 20px;
        }
        
        .tabs button {
          padding: 10px 20px;
          margin: 0 5px;
          background-color: #f0f0f0;
          color: #333;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 16px;
        }
        
        .tabs button.active {
          background-color: #4CAF50;
          color: white;
        }
        
        .tab-content {
          min-height: 500px;
        }
        
        footer {
          margin-top: 50px;
          text-align: center;
          color: #666;
          font-size: 14px;
        }
      `}</style>
    </div>
  );
}
