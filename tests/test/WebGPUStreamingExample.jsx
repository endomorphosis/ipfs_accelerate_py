import React, { useState, useEffect, useRef } from 'react';

/**
 * WebGPUStreamingExample - A simple React component that demonstrates WebGPU streaming inference
 * with different precision options and performance metrics visualization.
 */
const WebGPUStreamingExample = ({ modelId = 'llama-7b', initialPrecision = '4-bit' }) => {
  // State for input and configuration
  const [prompt, setPrompt] = useState('Explain how WebGPU streaming works for language models:');
  const [precision, setPrecision] = useState(initialPrecision);
  const [output, setOutput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  
  // State for performance metrics
  const [metrics, setMetrics] = useState({
    timeToFirstToken: null,
    tokensPerSecond: null,
    totalTokens: 0,
    memoryUsage: null,
    generationTime: null
  });
  
  // References for webgpu components and timing
  const webgpuRef = useRef(null);
  const startTimeRef = useRef(null);
  const tokenCountRef = useRef(0);
  const outputRef = useRef(null);
  
  // Effect to initialize WebGPU components
  useEffect(() => {
    const initializeWebGPU = async () => {
      try {
        console.log(`Initializing WebGPU with model: ${modelId}, precision: ${precision}`);
        
        // Simulate loading the WebGPU components
        // In a real implementation, this would use the actual WebGPU API
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Create mock WebGPU handler
        webgpuRef.current = {
          // Mock method to start streaming generation
          generate: async (text, options, callback) => {
            console.log(`Generating with: ${text}`);
            console.log(`Options:`, options);
            
            // Record start time
            const startTime = Date.now();
            startTimeRef.current = startTime;
            tokenCountRef.current = 0;
            
            // Get base delay based on precision
            const baseDelay = precision === '2-bit' ? 20 : 
                            precision === '3-bit' ? 40 : 
                            precision === '4-bit' ? 60 : 80;
            
            // Simulate prefill phase
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Generate tokens with simulated delays
            const words = text.split(' ');
            const maxTokens = options.maxTokens || 100;
            
            for (let i = 0; i < maxTokens; i++) {
              // Stop if component unmounted or generation stopped
              if (!webgpuRef.current || !startTimeRef.current) break;
              
              // Generate token
              tokenCountRef.current++;
              
              let token;
              if (i % 10 === 0) {
                token = '. ';
              } else if (i % 5 === 0) {
                token = ', ';
              } else {
                // Use a random word from input or placeholder
                const wordIndex = Math.floor(Math.random() * words.length);
                token = `${words[wordIndex] || 'token'} `;
              }
              
              // Calculate metrics for first token
              if (tokenCountRef.current === 1) {
                const timeToFirst = Date.now() - startTime;
                setMetrics(prev => ({
                  ...prev,
                  timeToFirstToken: timeToFirst
                }));
              }
              
              // Calculate token delay with jitter
              const jitter = Math.random() * 20 - 10;
              const delay = Math.max(10, baseDelay + jitter);
              
              // Wait for token generation
              await new Promise(resolve => setTimeout(resolve, delay));
              
              // Call the callback with the generated token
              callback(token, i === maxTokens - 1);
              
              // Update metrics periodically
              if (tokenCountRef.current % 5 === 0) {
                const elapsedTime = (Date.now() - startTime) / 1000;
                setMetrics(prev => ({
                  ...prev,
                  tokensPerSecond: tokenCountRef.current / elapsedTime,
                  totalTokens: tokenCountRef.current,
                  generationTime: elapsedTime,
                  memoryUsage: getMemoryUsage(precision)
                }));
              }
            }
            
            // Final metrics update
            const elapsedTime = (Date.now() - startTime) / 1000;
            setMetrics({
              timeToFirstToken: metrics.timeToFirstToken,
              tokensPerSecond: tokenCountRef.current / elapsedTime,
              totalTokens: tokenCountRef.current,
              generationTime: elapsedTime,
              memoryUsage: getMemoryUsage(precision)
            });
            
            return "Generation complete";
          },
          
          // Mock method to stop generation
          stopGeneration: () => {
            console.log('Stopping generation');
            startTimeRef.current = null;
          }
        };
        
        // Simulate memory usage based on precision
        function getMemoryUsage(precision) {
          // Placeholder values - would be actual measurements in real implementation
          const baseMemory = 1500; // Base memory usage in MB
          return precision === '2-bit' ? baseMemory * 0.125 : // 87.5% reduction
                precision === '3-bit' ? baseMemory * 0.1875 : // 81.25% reduction
                precision === '4-bit' ? baseMemory * 0.25 :   // 75% reduction
                baseMemory * 0.5;                            // 50% reduction for 8-bit
        }
        
        // Set initial memory usage
        setMetrics(prev => ({
          ...prev,
          memoryUsage: getMemoryUsage(precision)
        }));
        
      } catch (err) {
        console.error('Failed to initialize WebGPU:', err);
        setError('Could not initialize WebGPU streaming. Your browser may not support this feature.');
      }
    };
    
    initializeWebGPU();
    
    // Cleanup function
    return () => {
      // Cancel any ongoing generation
      if (webgpuRef.current) {
        webgpuRef.current.stopGeneration();
      }
      webgpuRef.current = null;
    };
  }, [modelId, precision]);

  // Handle form submission to start generation
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!webgpuRef.current || !prompt.trim() || isGenerating) return;
    
    try {
      setIsGenerating(true);
      setOutput('');
      setError(null);
      tokenCountRef.current = 0;
      
      // Reset metrics
      setMetrics({
        timeToFirstToken: null,
        tokensPerSecond: null,
        totalTokens: 0,
        memoryUsage: metrics.memoryUsage,
        generationTime: null
      });
      
      // Start generation with callback for each token
      await webgpuRef.current.generate(
        prompt, 
        { 
          maxTokens: 100,
          temperature: 0.7,
          precision: precision 
        },
        (token, isLast) => {
          // Append token to output
          setOutput(prev => prev + token);
          
          // Scroll to bottom
          if (outputRef.current) {
            outputRef.current.scrollTop = outputRef.current.scrollHeight;
          }
          
          // Handle completion
          if (isLast) {
            setIsGenerating(false);
          }
        }
      );
      
    } catch (err) {
      console.error('Generation error:', err);
      setError(`Generation failed: ${err.message}`);
      setIsGenerating(false);
    }
  };

  // Handle stopping generation
  const handleStop = () => {
    if (webgpuRef.current && isGenerating) {
      webgpuRef.current.stopGeneration();
      setIsGenerating(false);
    }
  };

  // Get memory reduction percentage based on precision
  const getMemoryReductionPercentage = () => {
    if (precision === '2-bit') return '87.5%';
    if (precision === '3-bit') return '81.25%';
    if (precision === '4-bit') return '75%';
    return '50%';  // 8-bit
  };

  return (
    <div className="webgpu-streaming-example">
      <div className="example-header">
        <h2>WebGPU Streaming Inference</h2>
        <div className="precision-selector">
          <label htmlFor="precision-select">Precision:</label>
          <select 
            id="precision-select"
            value={precision}
            onChange={(e) => setPrecision(e.target.value)}
            disabled={isGenerating}
          >
            <option value="2-bit">2-bit (Ultra-Low)</option>
            <option value="3-bit">3-bit (Very Low)</option>
            <option value="4-bit">4-bit (Low)</option>
            <option value="8-bit">8-bit (Standard)</option>
          </select>
          <div className="precision-info">
            <span className="memory-reduction">
              Memory reduction: {getMemoryReductionPercentage()}
            </span>
          </div>
        </div>
      </div>

      <div className="input-area">
        <form onSubmit={handleSubmit}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt here..."
            disabled={isGenerating}
            rows={4}
          />
          <div className="button-row">
            <button 
              type="submit" 
              disabled={!prompt.trim() || isGenerating}
              className="generate-button"
            >
              {isGenerating ? 'Generating...' : 'Generate'}
            </button>
            <button 
              type="button" 
              onClick={handleStop}
              disabled={!isGenerating}
              className="stop-button"
            >
              Stop
            </button>
          </div>
        </form>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="metrics-display">
        <div className="metric">
          <span className="metric-label">Time to First Token:</span>
          <span className="metric-value">
            {metrics.timeToFirstToken ? `${metrics.timeToFirstToken.toFixed(0)} ms` : '-'}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Tokens/Second:</span>
          <span className="metric-value">
            {metrics.tokensPerSecond ? `${metrics.tokensPerSecond.toFixed(2)}` : '-'}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Memory Usage:</span>
          <span className="metric-value">
            {metrics.memoryUsage ? `${metrics.memoryUsage.toFixed(0)} MB` : '-'}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Total Tokens:</span>
          <span className="metric-value">{metrics.totalTokens || '-'}</span>
        </div>
      </div>

      <div className="output-area" ref={outputRef}>
        {output ? (
          <div className="generated-text">
            {output}
            {isGenerating && <span className="cursor-blink"></span>}
          </div>
        ) : (
          <div className="placeholder-text">
            {isGenerating ? 'Generating...' : 'Generated text will appear here...'}
          </div>
        )}
      </div>

      <div className="precision-comparison">
        <h3>Precision Comparison</h3>
        <table>
          <thead>
            <tr>
              <th>Precision</th>
              <th>Memory Reduction</th>
              <th>Performance</th>
              <th>Quality</th>
            </tr>
          </thead>
          <tbody>
            <tr className={precision === '2-bit' ? 'active-row' : ''}>
              <td>2-bit</td>
              <td>87.5%</td>
              <td>Fastest</td>
              <td>Lowest</td>
            </tr>
            <tr className={precision === '3-bit' ? 'active-row' : ''}>
              <td>3-bit</td>
              <td>81.25%</td>
              <td>Very Fast</td>
              <td>Better</td>
            </tr>
            <tr className={precision === '4-bit' ? 'active-row' : ''}>
              <td>4-bit</td>
              <td>75%</td>
              <td>Fast</td>
              <td>Good</td>
            </tr>
            <tr className={precision === '8-bit' ? 'active-row' : ''}>
              <td>8-bit</td>
              <td>50%</td>
              <td>Standard</td>
              <td>High</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default WebGPUStreamingExample;