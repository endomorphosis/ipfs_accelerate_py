/**
 * Dashboard Core Functionality Tests
 * 
 * Tests basic dashboard loading, navigation, and MCP SDK initialization
 */

import { test, expect } from '@playwright/test';
import { test as dashboardTest } from '../fixtures/dashboard.fixture';
import { test as mcpTest } from '../fixtures/mcp-server.fixture';
import { LogCorrelator } from '../utils/log-correlator';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('Dashboard Core Functionality', () => {
  test('should load dashboard and initialize MCP SDK', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('core-dashboard');
    
    // Navigate to dashboard
    await page.goto('/');
    
    // Take initial screenshot
    await screenshotMgr.captureAndCompare(page, 'dashboard-loaded');
    
    // Verify page title
    await expect(page).toHaveTitle(/IPFS Accelerate|MCP/i);
    
    // Verify MCP SDK is loaded
    const mcpLoaded = await page.evaluate(() => {
      return typeof (window as any).MCPClient !== 'undefined';
    });
    expect(mcpLoaded).toBeTruthy();
    
    // Verify MCP client is initialized
    await page.waitForFunction(
      () => (window as any).mcpClient !== null && (window as any).mcpClient !== undefined,
      { timeout: 30000 }
    );
    
    // Take screenshot after SDK init
    await screenshotMgr.captureAndCompare(page, 'sdk-initialized');
    
    // Verify essential UI elements
    await expect(page.locator('h1')).toContainText(/IPFS Accelerate/i);
    await expect(page.locator('.status-bar')).toBeVisible();
    await expect(page.locator('.nav-tabs')).toBeVisible();
  });

  test('should navigate through all tabs', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('tab-navigation');
    
    await page.goto('/');
    await page.waitForTimeout(2000); // Wait for initialization
    
    const tabs = [
      '🏠 Overview',
      '🤖 AI Inference',
      '🚀 Advanced AI',
      '📚 Model Manager',
      '📁 IPFS Manager',
      '🌐 Network & Status',
      '📊 Queue Monitor',
      '⚡ GitHub Workflows',
      '🏃 Runner Management',
      '🎮 SDK Playground',
      '🔧 MCP Tools',
      '🎯 Coverage Analysis',
      '📝 System Logs',
    ];
    
    for (const tabName of tabs) {
      // Click tab
      const tabButton = page.locator(`button.nav-tab:has-text("${tabName}")`);
      await expect(tabButton).toBeVisible({ timeout: 10000 });
      await tabButton.click();
      
      // Wait for tab content
      await page.waitForTimeout(1000);
      
      // Verify tab is active
      await expect(tabButton).toHaveClass(/active/);
      
      // Take screenshot
      const cleanName = tabName.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
      await screenshotMgr.captureAndCompare(page, `tab-${cleanName}`);
    }
  });

  test('should capture and validate console logs', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    // Capture console messages
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString(),
      });
    });
    
    await page.goto('/');
    await page.waitForTimeout(3000);
    
    // Verify expected logs
    const sdkInitLog = consoleLogs.find(log => 
      /MCP SDK client initialized/i.test(log.text)
    );
    expect(sdkInitLog).toBeDefined();
    
    // Check for errors
    const errorLogs = consoleLogs.filter(log => log.type === 'error');
    console.log('Error logs found:', errorLogs.length);
    
    // Allow some errors but not too many
    expect(errorLogs.length).toBeLessThan(5);
  });

  test('should display server status', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Check status indicators
    await expect(page.locator('#server-status')).toBeVisible();
    await expect(page.locator('#port-number')).toContainText(/\d+/);
    await expect(page.locator('#active-connections')).toBeVisible();
    await expect(page.locator('#uptime')).toBeVisible();
  });

  test('should handle responsive design', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('responsive');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Test different viewports
    const viewports = ScreenshotManager.getStandardViewports();
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.waitForTimeout(1000);
      
      await screenshotMgr.captureAndCompare(page, viewport.name);
      
      // Verify essential elements are still visible
      await expect(page.locator('.header')).toBeVisible();
    }
  });
});
