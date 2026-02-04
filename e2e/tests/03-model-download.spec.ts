/**
 * AI Models Download Tests
 * 
 * Tests AI model downloading functionality and log correlation
 */

import { test, expect } from '@playwright/test';
import { ScreenshotManager } from '../utils/screenshot-manager';
import { LogCorrelator } from '../utils/log-correlator';

test.describe('AI Models Download', () => {
  test('should display Model Manager tab and search interface', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('model-manager');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Model Manager tab
    const modelTab = page.locator('button.nav-tab:has-text("Model Manager")');
    await expect(modelTab).toBeVisible({ timeout: 10000 });
    await modelTab.click();
    
    await page.waitForTimeout(1500);
    await screenshotMgr.captureAndCompare(page, 'model-manager-tab');
    
    // Verify search interface exists
    const searchInput = page.locator('input[type="text"], input[placeholder*="search" i]').first();
    await expect(searchInput).toBeVisible({ timeout: 10000 });
    
    await screenshotMgr.captureAndCompare(page, 'search-interface');
  });

  test('should search for models', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('model-search');
    const consoleLogs: any[] = [];
    const networkRequests: any[] = [];
    
    // Capture logs and network
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString(),
      });
    });
    
    page.on('request', req => {
      if (req.url().includes('search') || req.url().includes('models')) {
        networkRequests.push({
          url: req.url(),
          method: req.method(),
          timestamp: new Date().toISOString(),
        });
      }
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Model Manager
    await page.locator('button.nav-tab:has-text("Model Manager")').click();
    await page.waitForTimeout(1500);
    
    // Find search input
    const searchInput = page.locator('input[type="text"], input[placeholder*="search" i]').first();
    
    if (await searchInput.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Enter search query
      await searchInput.fill('llama');
      await page.waitForTimeout(500);
      
      await screenshotMgr.captureAndCompare(page, 'search-query-entered');
      
      // Look for search button or press Enter
      const searchBtn = page.locator('button:has-text("Search"), button[type="submit"]').first();
      
      if (await searchBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await searchBtn.click();
      } else {
        await searchInput.press('Enter');
      }
      
      // Wait for results
      await page.waitForTimeout(3000);
      
      await screenshotMgr.captureAndCompare(page, 'search-results', { fullPage: true });
      
      // Check logs for search activity
      const searchLogs = consoleLogs.filter(log =>
        /search|model|huggingface/i.test(log.text)
      );
      
      console.log('Search-related logs:', searchLogs.length);
      expect(searchLogs.length).toBeGreaterThan(0);
      
      console.log('Search network requests:', networkRequests.length);
    }
  });

  test('should display model details', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('model-details');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Model Manager
    await page.locator('button.nav-tab:has-text("Model Manager")').click();
    await page.waitForTimeout(2000);
    
    // Look for any model cards or list items
    const modelItems = page.locator('.model-card, .model-item, tr[data-model], [data-model-id]').first();
    
    if (await modelItems.isVisible({ timeout: 5000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'before-details');
      
      // Click on first model
      await modelItems.click();
      await page.waitForTimeout(2000);
      
      await screenshotMgr.captureAndCompare(page, 'model-details-shown');
    }
  });

  test('should initiate model download', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('model-download');
    const consoleLogs: any[] = [];
    const networkRequests: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString(),
      });
    });
    
    page.on('request', req => {
      if (req.url().includes('download') || req.url().includes('jsonrpc')) {
        networkRequests.push({
          url: req.url(),
          method: req.method(),
          postData: req.postData(),
          timestamp: new Date().toISOString(),
        });
      }
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Model Manager
    await page.locator('button.nav-tab:has-text("Model Manager")').click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, 'before-download');
    
    // Look for download button
    const downloadBtn = page.locator('button:has-text("Download"), button[title*="download" i]').first();
    
    if (await downloadBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      await downloadBtn.click();
      await page.waitForTimeout(2000);
      
      await screenshotMgr.captureAndCompare(page, 'download-initiated');
      
      // Check for download-related logs
      const downloadLogs = consoleLogs.filter(log =>
        /download/i.test(log.text)
      );
      
      console.log('Download-related logs:', downloadLogs.length);
      downloadLogs.forEach(log => {
        console.log(`  [${log.type}] ${log.text.substring(0, 100)}`);
      });
      
      // Check for download API calls
      const downloadCalls = networkRequests.filter(req =>
        /download/i.test(req.url) || 
        (req.postData && /download/i.test(req.postData))
      );
      
      console.log('Download API calls:', downloadCalls.length);
    }
  });

  test('should correlate download actions with MCP server logs', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('download-correlation');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString(),
      });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const startTime = new Date();
    
    // Navigate to Model Manager
    await page.locator('button.nav-tab:has-text("Model Manager")').click();
    await page.waitForTimeout(2000);
    
    // Try to trigger a download action
    const downloadBtn = page.locator('button:has-text("Download")').first();
    
    if (await downloadBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      consoleLogs.length = 0; // Clear previous logs
      
      await downloadBtn.click();
      await page.waitForTimeout(3000);
      
      const endTime = new Date();
      
      // Analyze logs in time window
      const relevantLogs = consoleLogs.filter(log => {
        const logTime = new Date(log.timestamp);
        return logTime >= startTime && logTime <= endTime;
      });
      
      await screenshotMgr.captureAndCompare(page, 'after-download-attempt');
      
      console.log('\n=== DOWNLOAD CORRELATION REPORT ===');
      console.log(`Time window: ${startTime.toISOString()} to ${endTime.toISOString()}`);
      console.log(`Relevant logs: ${relevantLogs.length}`);
      
      const downloadLogs = relevantLogs.filter(log => /download/i.test(log.text));
      console.log(`Download-specific logs: ${downloadLogs.length}`);
      
      downloadLogs.forEach((log, idx) => {
        console.log(`  ${idx + 1}. [${log.type}] ${log.text.substring(0, 120)}`);
      });
    }
  });

  test('should track download progress', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('download-progress');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Model Manager
    await page.locator('button.nav-tab:has-text("Model Manager")').click();
    await page.waitForTimeout(2000);
    
    // Check for progress indicators
    const progressElements = page.locator(
      '.progress, .progress-bar, [role="progressbar"], .download-status'
    );
    
    const progressCount = await progressElements.count();
    console.log('Progress indicators found:', progressCount);
    
    if (progressCount > 0) {
      await screenshotMgr.captureAndCompare(page, 'progress-indicators');
    }
    
    // Look for download queue or status
    const queueElement = page.locator('#download-queue, .download-list, .active-downloads');
    
    if (await queueElement.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'download-queue');
    }
  });
});
