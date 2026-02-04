/**
 * Test Report Generator
 * 
 * Generates comprehensive HTML and JSON reports for test results
 */

import fs from 'fs';
import path from 'path';
import { ConsoleMessage } from '../fixtures/dashboard.fixture';
import { MCPServerLog } from '../fixtures/mcp-server.fixture';
import { LogCorrelation } from './log-correlator';

export interface TestResult {
  name: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
  error?: string;
  screenshots: string[];
  consoleLogs: ConsoleMessage[];
  serverLogs: MCPServerLog[];
  correlations: LogCorrelation[];
}

export class ReportGenerator {
  private results: TestResult[] = [];
  private outputDir: string;

  constructor(outputDir: string = 'test-results/reports') {
    this.outputDir = outputDir;
    fs.mkdirSync(outputDir, { recursive: true });
  }

  addResult(result: TestResult) {
    this.results.push(result);
  }

  /**
   * Generate JSON report
   */
  generateJSON(): string {
    const report = {
      summary: {
        total: this.results.length,
        passed: this.results.filter(r => r.status === 'passed').length,
        failed: this.results.filter(r => r.status === 'failed').length,
        skipped: this.results.filter(r => r.status === 'skipped').length,
        duration: this.results.reduce((sum, r) => sum + r.duration, 0),
      },
      timestamp: new Date().toISOString(),
      results: this.results,
    };

    const jsonPath = path.join(this.outputDir, 'test-report.json');
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    
    return jsonPath;
  }

  /**
   * Generate HTML report
   */
  generateHTML(): string {
    const summary = {
      total: this.results.length,
      passed: this.results.filter(r => r.status === 'passed').length,
      failed: this.results.filter(r => r.status === 'failed').length,
      skipped: this.results.filter(r => r.status === 'skipped').length,
      duration: this.results.reduce((sum, r) => sum + r.duration, 0),
    };

    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E2E Test Report - IPFS Accelerate Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }
        .summary-card .value {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .summary-card .label {
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }
        .passed { color: #28a745; }
        .failed { color: #dc3545; }
        .skipped { color: #ffc107; }
        .test-result {
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .test-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .test-name { font-size: 18px; font-weight: 600; }
        .test-status {
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
        }
        .status-passed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .status-skipped { background: #fff3cd; color: #856404; }
        .test-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .detail-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }
        .detail-section h3 {
            font-size: 14px;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 10px;
        }
        .log-entry {
            padding: 8px;
            margin-bottom: 5px;
            background: white;
            border-radius: 4px;
            font-size: 13px;
            font-family: 'Monaco', 'Courier New', monospace;
        }
        .log-error { background: #fff5f5; color: #c53030; }
        .log-warn { background: #fffaf0; color: #c05621; }
        .screenshots {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .screenshot-item img {
            width: 100%;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .correlation {
            background: white;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 4px;
            border-left: 3px solid #28a745;
        }
        .correlation-item {
            font-size: 12px;
            margin-bottom: 4px;
        }
        .correlation-item strong {
            display: inline-block;
            width: 100px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 E2E Test Report</h1>
            <p>IPFS Accelerate Dashboard - Playwright Testing Suite</p>
            <p style="color: #666; margin-top: 5px;">Generated: ${new Date().toLocaleString()}</p>
            
            <div class="summary">
                <div class="summary-card">
                    <div class="value">${summary.total}</div>
                    <div class="label">Total Tests</div>
                </div>
                <div class="summary-card">
                    <div class="value passed">${summary.passed}</div>
                    <div class="label">Passed</div>
                </div>
                <div class="summary-card">
                    <div class="value failed">${summary.failed}</div>
                    <div class="label">Failed</div>
                </div>
                <div class="summary-card">
                    <div class="value skipped">${summary.skipped}</div>
                    <div class="label">Skipped</div>
                </div>
                <div class="summary-card">
                    <div class="value">${(summary.duration / 1000).toFixed(2)}s</div>
                    <div class="label">Duration</div>
                </div>
            </div>
        </div>

        ${this.results.map(result => this.renderTestResult(result)).join('')}
    </div>
</body>
</html>
    `;

    const htmlPath = path.join(this.outputDir, 'test-report.html');
    fs.writeFileSync(htmlPath, html);
    
    return htmlPath;
  }

  private renderTestResult(result: TestResult): string {
    const statusClass = `status-${result.status}`;
    
    return `
        <div class="test-result">
            <div class="test-header">
                <div class="test-name">${result.name}</div>
                <div class="test-status ${statusClass}">${result.status.toUpperCase()}</div>
            </div>
            
            ${result.error ? `
                <div class="detail-section">
                    <h3>❌ Error</h3>
                    <div class="log-entry log-error">${result.error}</div>
                </div>
            ` : ''}
            
            <div class="test-details">
                <div class="detail-section">
                    <h3>📝 Console Logs (${result.consoleLogs.length})</h3>
                    ${result.consoleLogs.slice(0, 10).map(log => `
                        <div class="log-entry ${log.type === 'error' ? 'log-error' : log.type === 'warn' ? 'log-warn' : ''}">
                            [${log.type}] ${log.text.substring(0, 100)}
                        </div>
                    `).join('')}
                    ${result.consoleLogs.length > 10 ? `<p style="margin-top: 10px; color: #666;">...and ${result.consoleLogs.length - 10} more</p>` : ''}
                </div>
                
                <div class="detail-section">
                    <h3>🖥️ Server Logs (${result.serverLogs.length})</h3>
                    ${result.serverLogs.slice(0, 10).map(log => `
                        <div class="log-entry">
                            [${log.level}] ${log.message.substring(0, 100)}
                        </div>
                    `).join('')}
                    ${result.serverLogs.length > 10 ? `<p style="margin-top: 10px; color: #666;">...and ${result.serverLogs.length - 10} more</p>` : ''}
                </div>
                
                <div class="detail-section">
                    <h3>🔗 Log Correlations (${result.correlations.length})</h3>
                    ${result.correlations.slice(0, 5).map(corr => `
                        <div class="correlation">
                            <div class="correlation-item">
                                <strong>Dashboard:</strong> ${corr.dashboardLog.text.substring(0, 80)}
                            </div>
                            <div class="correlation-item">
                                <strong>Server:</strong> ${corr.serverLog.message.substring(0, 80)}
                            </div>
                            <div class="correlation-item">
                                <strong>Time Delta:</strong> ${corr.timeDelta}ms
                            </div>
                        </div>
                    `).join('')}
                    ${result.correlations.length > 5 ? `<p style="margin-top: 10px; color: #666;">...and ${result.correlations.length - 5} more</p>` : ''}
                </div>
            </div>
            
            ${result.screenshots.length > 0 ? `
                <div class="detail-section" style="margin-top: 20px;">
                    <h3>📸 Screenshots (${result.screenshots.length})</h3>
                    <div class="screenshots">
                        ${result.screenshots.map((screenshot, idx) => `
                            <div class="screenshot-item">
                                <img src="${path.relative(this.outputDir, screenshot)}" alt="Screenshot ${idx + 1}" />
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
  }
}
