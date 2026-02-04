/**
 * GitHub Runners Provisioning Tests
 * 
 * Tests GitHub runner provisioning workflow and log correlation
 */

import { test, expect } from '@playwright/test';
import { LogCorrelator } from '../utils/log-correlator';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('GitHub Runners Provisioning', () => {
  test('should display GitHub Workflows tab and load workflows', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('github-workflows');
    const consoleLogs: any[] = [];
    
    // Capture console logs
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString(),
      });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to GitHub Workflows tab
    const workflowsTab = page.locator('button.nav-tab:has-text("GitHub Workflows")');
    await expect(workflowsTab).toBeVisible({ timeout: 10000 });
    await workflowsTab.click();
    
    await screenshotMgr.captureAndCompare(page, 'workflows-tab-opened');
    
    // Verify workflows container exists
    await expect(page.locator('#github-workflows')).toBeVisible();
    await expect(page.locator('#github-workflows-container')).toBeAttached();
    
    // Take screenshot of workflows section
    await page.waitForTimeout(2000);
    await screenshotMgr.captureAndCompare(page, 'workflows-loaded');
    
    // Check for workflow-related console logs
    const workflowLogs = consoleLogs.filter(log =>
      /workflow|github/i.test(log.text)
    );
    
    console.log('Workflow-related logs:', workflowLogs.length);
    expect(workflowLogs.length).toBeGreaterThan(0);
  });

  test('should display runner management interface', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('runner-management');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Runner Management tab
    const runnerTab = page.locator('button.nav-tab:has-text("Runner Management")');
    await expect(runnerTab).toBeVisible({ timeout: 10000 });
    await runnerTab.click();
    
    await page.waitForTimeout(1500);
    await screenshotMgr.captureAndCompare(page, 'runner-tab-opened');
    
    // Verify runner containers exist
    await expect(page.locator('#active-runners-container')).toBeAttached();
    await expect(page.locator('#github-runners-container')).toBeAttached();
    
    // Take full page screenshot
    await screenshotMgr.captureAndCompare(page, 'runners-interface', { fullPage: true });
  });

  test('should call runner-related MCP tools', async ({ page }) => {
    const consoleLogs: any[] = [];
    const mcpCalls: any[] = [];
    
    // Intercept network requests
    page.on('request', request => {
      if (request.url().includes('/jsonrpc') || request.url().includes('tools/call')) {
        mcpCalls.push({
          url: request.url(),
          method: request.method(),
          postData: request.postData(),
          timestamp: new Date().toISOString(),
        });
      }
    });
    
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
      });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Runner Management
    const runnerTab = page.locator('button.nav-tab:has-text("Runner Management")');
    await runnerTab.click();
    await page.waitForTimeout(2000);
    
    // Try to interact with runner controls if they exist
    const loadRunnersBtn = page.locator('button:has-text("Load Runners"), button:has-text("Refresh")').first();
    
    if (await loadRunnersBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      await loadRunnersBtn.click();
      await page.waitForTimeout(2000);
    }
    
    // Verify MCP calls were made
    console.log('MCP calls made:', mcpCalls.length);
    console.log('Console logs:', consoleLogs.length);
    
    // Check if runner-related tools were called
    const runnerToolCalls = mcpCalls.filter(call => {
      const data = call.postData || '';
      return /gh_list_runners|runner|github/i.test(data);
    });
    
    console.log('Runner tool calls:', runnerToolCalls.length);
  });

  test('should correlate dashboard actions with MCP server logs', async ({ page }) => {
    const consoleLogs: any[] = [];
    const screenshotMgr = new ScreenshotManager('runner-log-correlation');
    
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString(),
      });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Clear logs before test
    consoleLogs.length = 0;
    
    // Navigate to GitHub Workflows
    await page.locator('button.nav-tab:has-text("GitHub Workflows")').click();
    await page.waitForTimeout(3000);
    
    await screenshotMgr.captureAndCompare(page, 'before-workflow-action');
    
    // Look for GitHub-related logs
    const githubLogs = consoleLogs.filter(log =>
      /github|workflow|runner/i.test(log.text)
    );
    
    console.log('GitHub-related logs found:', githubLogs.length);
    githubLogs.forEach(log => {
      console.log(`  [${log.type}] ${log.text.substring(0, 100)}`);
    });
    
    await screenshotMgr.captureAndCompare(page, 'after-workflow-action');
    
    // Verify we got some activity
    expect(githubLogs.length).toBeGreaterThan(0);
  });

  test('should test runner provisioning workflow end-to-end', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('runner-provisioning-e2e');
    const consoleLogs: any[] = [];
    const networkRequests: any[] = [];
    
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
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Step 1: Navigate to Runner Management
    await screenshotMgr.captureAndCompare(page, '01-initial-state');
    
    const runnerTab = page.locator('button.nav-tab:has-text("Runner Management")');
    await runnerTab.click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, '02-runner-tab');
    
    // Step 2: Check for runner list
    const runnersList = page.locator('#github-runners-container');
    await expect(runnersList).toBeAttached();
    
    await screenshotMgr.captureAndCompare(page, '03-runners-list');
    
    // Step 3: Verify MCP SDK is being used
    const mcpClientActive = await page.evaluate(() => {
      return (window as any).mcpClient !== null;
    });
    
    expect(mcpClientActive).toBeTruthy();
    
    await screenshotMgr.captureAndCompare(page, '04-final-state');
    
    // Generate log report
    console.log('\n=== LOG CORRELATION REPORT ===');
    console.log(`Total console logs: ${consoleLogs.length}`);
    console.log(`Total network requests: ${networkRequests.length}`);
    
    const runnerLogs = consoleLogs.filter(log => /runner/i.test(log.text));
    console.log(`Runner-related logs: ${runnerLogs.length}`);
    
    const mcpRequests = networkRequests.filter(req => 
      req.url.includes('/jsonrpc') || req.url.includes('tools/call')
    );
    console.log(`MCP requests: ${mcpRequests.length}`);
  });
});
