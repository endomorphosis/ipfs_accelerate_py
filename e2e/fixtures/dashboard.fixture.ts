import { test as base, Page, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';

/**
 * Dashboard Fixture
 * 
 * Provides utilities for testing the IPFS Accelerate Dashboard
 */

export interface ConsoleMessage {
  type: 'log' | 'info' | 'warn' | 'error' | 'debug';
  text: string;
  timestamp: string;
  args?: any[];
}

export interface DashboardFixture {
  consoleLogs: ConsoleMessage[];
  errors: Error[];
  screenshotCounter: number;
  
  // Navigation helpers
  navigateToTab: (tabName: string) => Promise<void>;
  waitForMCPReady: () => Promise<void>;
  
  // Screenshot helpers
  takeScreenshot: (name: string, fullPage?: boolean) => Promise<string>;
  
  // Console log helpers
  waitForConsoleLog: (pattern: string | RegExp, timeout?: number) => Promise<ConsoleMessage | null>;
  getConsoleLogs: (type?: string) => ConsoleMessage[];
  clearConsoleLogs: () => void;
  
  // MCP SDK helpers
  callMCPTool: (toolName: string, params?: any) => Promise<any>;
  waitForMCPTool: (toolName: string, timeout?: number) => Promise<boolean>;
}

export const test = base.extend<{ dashboard: DashboardFixture }>({
  dashboard: async ({ page }, use) => {
    const consoleLogs: ConsoleMessage[] = [];
    const errors: Error[] = [];
    let screenshotCounter = 0;
    
    // Create screenshots directory
    const screenshotsDir = path.join(process.cwd(), 'test-results', 'screenshots');
    fs.mkdirSync(screenshotsDir, { recursive: true });
    
    // Setup console log capture
    page.on('console', msg => {
      const consoleMsg: ConsoleMessage = {
        type: msg.type() as any,
        text: msg.text(),
        timestamp: new Date().toISOString(),
      };
      consoleLogs.push(consoleMsg);
    });
    
    // Setup error capture
    page.on('pageerror', error => {
      errors.push(error);
      console.error('Page error:', error);
    });
    
    // Navigate to a specific tab
    const navigateToTab = async (tabName: string) => {
      const tabButton = page.locator(`button.nav-tab:has-text("${tabName}")`);
      await expect(tabButton).toBeVisible({ timeout: 10000 });
      await tabButton.click();
      await page.waitForTimeout(1000); // Wait for tab content to load
    };
    
    // Wait for MCP SDK to be ready
    const waitForMCPReady = async () => {
      await page.waitForFunction(
        () => typeof (window as any).mcpClient !== 'undefined' && 
              (window as any).mcpClient !== null,
        { timeout: 30000 }
      );
    };
    
    // Take a screenshot with auto-incrementing counter
    const takeScreenshot = async (name: string, fullPage: boolean = false): Promise<string> => {
      screenshotCounter++;
      const filename = `${screenshotCounter.toString().padStart(2, '0')}_${name}.png`;
      const filepath = path.join(screenshotsDir, filename);
      
      await page.screenshot({
        path: filepath,
        fullPage,
      });
      
      console.log(`Screenshot saved: ${filename}`);
      return filepath;
    };
    
    // Wait for a console log matching a pattern
    const waitForConsoleLog = async (
      pattern: string | RegExp,
      timeout: number = 30000
    ): Promise<ConsoleMessage | null> => {
      const startTime = Date.now();
      const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
      
      while (Date.now() - startTime < timeout) {
        const matchingLog = consoleLogs.find(log => regex.test(log.text));
        if (matchingLog) {
          return matchingLog;
        }
        await page.waitForTimeout(100);
      }
      
      return null;
    };
    
    // Get console logs, optionally filtered by type
    const getConsoleLogs = (type?: string): ConsoleMessage[] => {
      if (type) {
        return consoleLogs.filter(log => log.type === type);
      }
      return [...consoleLogs];
    };
    
    // Clear console logs
    const clearConsoleLogs = () => {
      consoleLogs.length = 0;
    };
    
    // Call an MCP tool via the JavaScript SDK
    const callMCPTool = async (toolName: string, params: any = {}): Promise<any> => {
      const result = await page.evaluate(async ({ toolName, params }) => {
        const client = (window as any).mcpClient;
        if (!client) {
          throw new Error('MCP client not initialized');
        }
        
        return await client.request('tools/call', {
          name: toolName,
          arguments: params,
        });
      }, { toolName, params });
      
      return result;
    };
    
    // Wait for an MCP tool to be called
    const waitForMCPTool = async (toolName: string, timeout: number = 30000): Promise<boolean> => {
      const pattern = new RegExp(`tools/call.*${toolName}`, 'i');
      const log = await waitForConsoleLog(pattern, timeout);
      return log !== null;
    };
    
    const fixture: DashboardFixture = {
      consoleLogs,
      errors,
      screenshotCounter,
      navigateToTab,
      waitForMCPReady,
      takeScreenshot,
      waitForConsoleLog,
      getConsoleLogs,
      clearConsoleLogs,
      callMCPTool,
      waitForMCPTool,
    };
    
    await use(fixture);
  },
});

export { expect };
