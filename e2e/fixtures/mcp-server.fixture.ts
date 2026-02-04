import { test as base, expect } from '@playwright/test';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';

/**
 * MCP Server Fixture
 * 
 * Provides utilities for starting/stopping the MCP server and capturing its logs
 */

export interface MCPServerLog {
  timestamp: string;
  level: string;
  message: string;
  data?: any;
}

export interface MCPServerFixture {
  serverLogs: MCPServerLog[];
  waitForLog: (pattern: string | RegExp, timeout?: number) => Promise<MCPServerLog | null>;
  clearLogs: () => void;
  getLogsMatching: (pattern: string | RegExp) => MCPServerLog[];
}

export const test = base.extend<{ mcpServer: MCPServerFixture }>({
  mcpServer: async ({}, use) => {
    const serverLogs: MCPServerLog[] = [];
    let serverProcess: ChildProcess | null = null;

    // Log capture utilities
    const captureLog = (data: string, level: 'info' | 'error') => {
      const lines = data.toString().split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        const log: MCPServerLog = {
          timestamp: new Date().toISOString(),
          level: level.toUpperCase(),
          message: line,
        };
        
        // Try to parse JSON logs
        try {
          const jsonMatch = line.match(/\{.*\}/);
          if (jsonMatch) {
            log.data = JSON.parse(jsonMatch[0]);
          }
        } catch {
          // Not JSON, just keep as string
        }
        
        serverLogs.push(log);
      }
    };

    // Wait for a specific log pattern
    const waitForLog = async (
      pattern: string | RegExp,
      timeout: number = 30000
    ): Promise<MCPServerLog | null> => {
      const startTime = Date.now();
      const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
      
      while (Date.now() - startTime < timeout) {
        const matchingLog = serverLogs.find(log => 
          regex.test(log.message) || 
          (log.data && regex.test(JSON.stringify(log.data)))
        );
        
        if (matchingLog) {
          return matchingLog;
        }
        
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      return null;
    };

    // Get all logs matching a pattern
    const getLogsMatching = (pattern: string | RegExp): MCPServerLog[] => {
      const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
      return serverLogs.filter(log => 
        regex.test(log.message) || 
        (log.data && regex.test(JSON.stringify(log.data)))
      );
    };

    // Clear logs
    const clearLogs = () => {
      serverLogs.length = 0;
    };

    const fixture: MCPServerFixture = {
      serverLogs,
      waitForLog,
      clearLogs,
      getLogsMatching,
    };

    // Use the fixture
    await use(fixture);

    // Cleanup: stop server if running
    if (serverProcess) {
      serverProcess.kill();
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  },
});

export { expect };
