/**
 * Log Correlation Utility
 * 
 * Correlates dashboard console logs with MCP server logs to ensure
 * end-to-end functionality is working correctly.
 */

import { ConsoleMessage } from '../fixtures/dashboard.fixture';
import { MCPServerLog } from '../fixtures/mcp-server.fixture';

export interface LogCorrelation {
  dashboardLog: ConsoleMessage;
  serverLog: MCPServerLog;
  timeDelta: number; // milliseconds between logs
  matched: boolean;
}

export interface CorrelationPattern {
  dashboardPattern: string | RegExp;
  serverPattern: string | RegExp;
  maxTimeDelta?: number; // maximum time difference in ms (default: 5000)
  description: string;
}

export class LogCorrelator {
  private correlations: LogCorrelation[] = [];

  /**
   * Find correlations between dashboard and server logs
   */
  findCorrelations(
    dashboardLogs: ConsoleMessage[],
    serverLogs: MCPServerLog[],
    patterns: CorrelationPattern[]
  ): LogCorrelation[] {
    this.correlations = [];

    for (const pattern of patterns) {
      const dashRegex = typeof pattern.dashboardPattern === 'string'
        ? new RegExp(pattern.dashboardPattern, 'i')
        : pattern.dashboardPattern;
      
      const serverRegex = typeof pattern.serverPattern === 'string'
        ? new RegExp(pattern.serverPattern, 'i')
        : pattern.serverPattern;

      const maxDelta = pattern.maxTimeDelta || 5000;

      // Find matching dashboard logs
      const matchingDashLogs = dashboardLogs.filter(log => 
        dashRegex.test(log.text)
      );

      // Find matching server logs
      const matchingServerLogs = serverLogs.filter(log =>
        serverRegex.test(log.message) ||
        (log.data && serverRegex.test(JSON.stringify(log.data)))
      );

      // Correlate based on timestamp proximity
      for (const dashLog of matchingDashLogs) {
        const dashTime = new Date(dashLog.timestamp).getTime();
        
        for (const serverLog of matchingServerLogs) {
          const serverTime = new Date(serverLog.timestamp).getTime();
          const timeDelta = Math.abs(dashTime - serverTime);

          if (timeDelta <= maxDelta) {
            this.correlations.push({
              dashboardLog: dashLog,
              serverLog: serverLog,
              timeDelta,
              matched: true,
            });
          }
        }
      }
    }

    return this.correlations;
  }

  /**
   * Verify that a specific correlation exists
   */
  assertCorrelation(
    dashboardPattern: string | RegExp,
    serverPattern: string | RegExp,
    dashboardLogs: ConsoleMessage[],
    serverLogs: MCPServerLog[],
    options: { maxTimeDelta?: number; description?: string } = {}
  ): boolean {
    const correlation = this.findCorrelations(
      dashboardLogs,
      serverLogs,
      [{
        dashboardPattern,
        serverPattern,
        maxTimeDelta: options.maxTimeDelta,
        description: options.description || 'Custom correlation',
      }]
    );

    return correlation.length > 0;
  }

  /**
   * Generate a correlation report
   */
  generateReport(): string {
    const lines: string[] = [];
    
    lines.push('='.repeat(80));
    lines.push('LOG CORRELATION REPORT');
    lines.push('='.repeat(80));
    lines.push('');
    lines.push(`Total Correlations Found: ${this.correlations.length}`);
    lines.push('');

    if (this.correlations.length === 0) {
      lines.push('⚠️  No correlations found');
      return lines.join('\n');
    }

    for (const [index, corr] of this.correlations.entries()) {
      lines.push(`Correlation #${index + 1}:`);
      lines.push(`  ✓ Dashboard: ${corr.dashboardLog.text.substring(0, 100)}`);
      lines.push(`  ✓ Server:    ${corr.serverLog.message.substring(0, 100)}`);
      lines.push(`  ⏱  Time Delta: ${corr.timeDelta}ms`);
      lines.push('');
    }

    lines.push('='.repeat(80));
    return lines.join('\n');
  }

  /**
   * Get common correlation patterns for the dashboard
   */
  static getCommonPatterns(): CorrelationPattern[] {
    return [
      {
        dashboardPattern: /MCP SDK client initialized/i,
        serverPattern: /MCP.*server.*start/i,
        description: 'MCP SDK initialization',
      },
      {
        dashboardPattern: /Downloading model.*(\w+)/i,
        serverPattern: /download.*model/i,
        maxTimeDelta: 10000,
        description: 'Model download',
      },
      {
        dashboardPattern: /Running inference/i,
        serverPattern: /inference.*request/i,
        maxTimeDelta: 10000,
        description: 'AI inference',
      },
      {
        dashboardPattern: /GitHub.*workflow/i,
        serverPattern: /gh_create_workflow_queues|workflow.*created/i,
        description: 'GitHub workflow creation',
      },
      {
        dashboardPattern: /runner.*provision/i,
        serverPattern: /runner.*created|provision.*runner/i,
        description: 'Runner provisioning',
      },
      {
        dashboardPattern: /search.*models/i,
        serverPattern: /search.*huggingface|model.*search/i,
        description: 'Model search',
      },
      {
        dashboardPattern: /hardware.*info/i,
        serverPattern: /hardware.*detected|system.*info/i,
        description: 'Hardware info',
      },
      {
        dashboardPattern: /network.*peers/i,
        serverPattern: /peer.*connected|network.*status/i,
        description: 'Network peer status',
      },
    ];
  }
}

/**
 * Log matcher for specific test scenarios
 */
export class LogMatcher {
  /**
   * Match a sequence of logs in order
   */
  static matchSequence(
    logs: ConsoleMessage[],
    patterns: (string | RegExp)[],
    options: { ordered?: boolean; timeout?: number } = {}
  ): boolean {
    const ordered = options.ordered !== false;
    
    if (ordered) {
      let lastIndex = -1;
      
      for (const pattern of patterns) {
        const regex = typeof pattern === 'string' ? new RegExp(pattern, 'i') : pattern;
        const index = logs.findIndex((log, idx) => idx > lastIndex && regex.test(log.text));
        
        if (index === -1) {
          return false;
        }
        
        lastIndex = index;
      }
      
      return true;
    } else {
      // All patterns must exist, but order doesn't matter
      for (const pattern of patterns) {
        const regex = typeof pattern === 'string' ? new RegExp(pattern, 'i') : pattern;
        const found = logs.some(log => regex.test(log.text));
        
        if (!found) {
          return false;
        }
      }
      
      return true;
    }
  }

  /**
   * Check if a log appears within a time window
   */
  static matchTimeWindow(
    logs: ConsoleMessage[],
    pattern: string | RegExp,
    startTime: Date,
    endTime: Date
  ): ConsoleMessage[] {
    const regex = typeof pattern === 'string' ? new RegExp(pattern, 'i') : pattern;
    
    return logs.filter(log => {
      const logTime = new Date(log.timestamp);
      return logTime >= startTime && 
             logTime <= endTime && 
             regex.test(log.text);
    });
  }
}
