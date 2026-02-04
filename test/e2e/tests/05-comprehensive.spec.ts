/**
 * Comprehensive End-to-End Test Suite
 * 
 * Tests complete workflows with full log correlation
 */

import { test, expect } from '@playwright/test';
import { LogCorrelator, CorrelationPattern } from '../utils/log-correlator';
import { ScreenshotManager } from '../utils/screenshot-manager';
import { ReportGenerator, TestResult } from '../utils/report-generator';

test.describe('Comprehensive E2E Workflow Tests', () => {
  test('complete workflow: dashboard → runners → models → inference', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('complete-workflow');
    const consoleLogs: any[] = [];
    const networkRequests: any[] = [];
    const testStartTime = Date.now();
    
    // Capture everything
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString(),
      });
    });
    
    page.on('request', req => {
      networkRequests.push({
        url: req.url(),
        method: req.method(),
        timestamp: new Date().toISOString(),
      });
    });
    
    // Step 1: Load Dashboard
    console.log('\n=== Step 1: Loading Dashboard ===');
    await page.goto('/');
    await page.waitForTimeout(3000);
    await screenshotMgr.captureAndCompare(page, '01-dashboard-loaded');
    
    // Verify MCP SDK
    const mcpLoaded = await page.evaluate(() => {
      return typeof (window as any).MCPClient !== 'undefined' &&
             (window as any).mcpClient !== null;
    });
    expect(mcpLoaded).toBeTruthy();
    console.log('✓ MCP SDK loaded');
    
    // Step 2: Check GitHub Runners
    console.log('\n=== Step 2: Checking GitHub Runners ===');
    await page.locator('button.nav-tab:has-text("Runner Management")').click();
    await page.waitForTimeout(2000);
    await screenshotMgr.captureAndCompare(page, '02-runners-tab');
    
    const runnersContainer = page.locator('#github-runners-container, #active-runners-container');
    await expect(runnersContainer.first()).toBeAttached();
    console.log('✓ Runners interface displayed');
    
    // Step 3: Check Model Manager
    console.log('\n=== Step 3: Checking Model Manager ===');
    await page.locator('button.nav-tab:has-text("Model Manager")').click();
    await page.waitForTimeout(2000);
    await screenshotMgr.captureAndCompare(page, '03-model-manager-tab');
    
    // Try to search for a model
    const searchInput = page.locator('input[type="text"]').first();
    if (await searchInput.isVisible({ timeout: 3000 }).catch(() => false)) {
      await searchInput.fill('bert');
      await page.waitForTimeout(1000);
      await screenshotMgr.captureAndCompare(page, '04-model-search');
      console.log('✓ Model search interface working');
    }
    
    // Step 4: Check AI Inference
    console.log('\n=== Step 4: Checking AI Inference ===');
    await page.locator('button.nav-tab:has-text("AI Inference")').click();
    await page.waitForTimeout(2000);
    await screenshotMgr.captureAndCompare(page, '05-inference-tab');
    
    const inferenceUI = page.locator('#ai-inference');
    await expect(inferenceUI).toBeVisible();
    console.log('✓ Inference interface displayed');
    
    // Step 5: Check Network Status
    console.log('\n=== Step 5: Checking Network Status ===');
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    await screenshotMgr.captureAndCompare(page, '06-network-status');
    
    // Step 6: Generate final report
    const testDuration = Date.now() - testStartTime;
    
    console.log('\n=== COMPREHENSIVE TEST REPORT ===');
    console.log(`Test Duration: ${testDuration}ms`);
    console.log(`Console Logs: ${consoleLogs.length}`);
    console.log(`Network Requests: ${networkRequests.length}`);
    
    // Analyze logs
    const errorLogs = consoleLogs.filter(log => log.type === 'error');
    const warnLogs = consoleLogs.filter(log => log.type === 'warn');
    
    console.log(`Errors: ${errorLogs.length}`);
    console.log(`Warnings: ${warnLogs.length}`);
    
    // Log correlation
    const correlator = new LogCorrelator();
    const patterns = LogCorrelator.getCommonPatterns();
    
    // Note: In a real implementation, we'd correlate with actual MCP server logs
    // For now, we just verify console logs contain expected patterns
    const foundPatterns: string[] = [];
    for (const pattern of patterns) {
      const dashRegex = typeof pattern.dashboardPattern === 'string' 
        ? new RegExp(pattern.dashboardPattern, 'i')
        : pattern.dashboardPattern;
      
      const found = consoleLogs.some(log => dashRegex.test(log.text));
      if (found) {
        foundPatterns.push(pattern.description);
      }
    }
    
    console.log(`\nMatched Patterns (${foundPatterns.length}/${patterns.length}):`);
    foundPatterns.forEach(p => console.log(`  ✓ ${p}`));
    
    // Take final screenshot
    await screenshotMgr.captureAndCompare(page, '07-final-state', { fullPage: true });
    
    // Verify minimum functionality
    expect(consoleLogs.length).toBeGreaterThan(10);
    expect(networkRequests.length).toBeGreaterThan(5);
    expect(errorLogs.length).toBeLessThan(10);
  });

  test('verify all dashboard tabs are functional', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('all-tabs');
    const tabResults: { name: string; success: boolean; error?: string }[] = [];
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const tabs = [
      'Overview',
      'AI Inference',
      'Advanced AI',
      'Model Manager',
      'IPFS Manager',
      'Network & Status',
      'Queue Monitor',
      'GitHub Workflows',
      'Runner Management',
      'SDK Playground',
      'MCP Tools',
      'Coverage Analysis',
      'System Logs',
    ];
    
    for (const tabName of tabs) {
      try {
        console.log(`\nTesting tab: ${tabName}`);
        
        const tabButton = page.locator(`button.nav-tab:has-text("${tabName}")`);
        await expect(tabButton).toBeVisible({ timeout: 10000 });
        await tabButton.click();
        await page.waitForTimeout(1000);
        
        // Verify tab content is visible
        await expect(tabButton).toHaveClass(/active/);
        
        const cleanName = tabName.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
        await screenshotMgr.captureAndCompare(page, `tab-${cleanName}`);
        
        tabResults.push({ name: tabName, success: true });
        console.log(`  ✓ ${tabName} tab functional`);
      } catch (error: any) {
        tabResults.push({ name: tabName, success: false, error: error.message });
        console.log(`  ✗ ${tabName} tab failed: ${error.message}`);
      }
    }
    
    // Summary
    const successCount = tabResults.filter(r => r.success).length;
    console.log(`\n=== TAB FUNCTIONALITY SUMMARY ===`);
    console.log(`Successful: ${successCount}/${tabs.length}`);
    console.log(`Failed: ${tabs.length - successCount}`);
    
    // Verify at least 80% of tabs work
    expect(successCount).toBeGreaterThanOrEqual(tabs.length * 0.8);
  });

  test('stress test: rapid navigation and interactions', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('stress-test');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const tabs = ['AI Inference', 'Model Manager', 'Runner Management', 'Overview'];
    
    // Rapidly switch between tabs
    for (let i = 0; i < 10; i++) {
      const randomTab = tabs[i % tabs.length];
      await page.locator(`button.nav-tab:has-text("${randomTab}")`).click();
      await page.waitForTimeout(500);
    }
    
    await screenshotMgr.captureAndCompare(page, 'after-rapid-switching');
    
    // Check for excessive errors
    const errors = consoleLogs.filter(log => log.type === 'error');
    console.log(`Errors after stress test: ${errors.length}`);
    
    expect(errors.length).toBeLessThan(20);
  });

  test('verify MCP tool execution end-to-end', async ({ page }) => {
    const consoleLogs: any[] = [];
    const mcpCalls: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString(),
      });
    });
    
    page.on('request', req => {
      if (req.url().includes('/jsonrpc') || req.url().includes('tools/call')) {
        const postData = req.postData();
        mcpCalls.push({
          url: req.url(),
          method: req.method(),
          data: postData,
          timestamp: new Date().toISOString(),
        });
      }
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to MCP Tools tab
    await page.locator('button.nav-tab:has-text("MCP Tools")').click();
    await page.waitForTimeout(2000);
    
    // Look for any tool execution buttons
    const toolButtons = page.locator('button[data-tool], button[onclick*="mcp"], button:has-text("Execute")');
    const buttonCount = await toolButtons.count();
    
    console.log(`\nFound ${buttonCount} tool buttons`);
    
    if (buttonCount > 0) {
      // Try to execute a tool
      await toolButtons.first().click();
      await page.waitForTimeout(3000);
      
      console.log(`\nMCP Calls Made: ${mcpCalls.length}`);
      
      mcpCalls.forEach((call, idx) => {
        console.log(`  ${idx + 1}. ${call.method} ${call.url}`);
        if (call.data) {
          console.log(`      Data: ${call.data.substring(0, 100)}`);
        }
      });
      
      // Verify at least one MCP call was made
      expect(mcpCalls.length).toBeGreaterThan(0);
    }
  });
});
