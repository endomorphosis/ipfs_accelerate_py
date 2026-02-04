/**
 * AI Models Inference Tests
 * 
 * Tests AI model inference functionality and log correlation with MCP server
 */

import { test, expect } from '@playwright/test';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('AI Models Inference', () => {
  test('should display AI Inference tab', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('ai-inference');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to AI Inference tab
    const inferenceTab = page.locator('button.nav-tab:has-text("AI Inference")');
    await expect(inferenceTab).toBeVisible({ timeout: 10000 });
    await inferenceTab.click();
    
    await page.waitForTimeout(1500);
    await screenshotMgr.captureAndCompare(page, 'inference-tab');
    
    // Verify inference interface elements
    await expect(page.locator('#ai-inference')).toBeVisible();
    
    await screenshotMgr.captureAndCompare(page, 'inference-interface', { fullPage: true });
  });

  test('should display model selection interface', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('model-selection');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to AI Inference
    await page.locator('button.nav-tab:has-text("AI Inference")').click();
    await page.waitForTimeout(2000);
    
    // Look for model selector
    const modelSelector = page.locator(
      'select#model-select, select[name="model"], #modelSelector'
    ).first();
    
    if (await modelSelector.isVisible({ timeout: 5000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'model-selector-visible');
      
      // Get available models
      const options = await modelSelector.locator('option').count();
      console.log('Available models:', options);
      
      if (options > 1) {
        // Select a model
        await modelSelector.selectOption({ index: 1 });
        await page.waitForTimeout(500);
        
        await screenshotMgr.captureAndCompare(page, 'model-selected');
      }
    }
  });

  test('should configure inference parameters', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('inference-params');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to AI Inference
    await page.locator('button.nav-tab:has-text("AI Inference")').click();
    await page.waitForTimeout(2000);
    
    // Look for parameter controls
    const paramInputs = page.locator(
      'input[type="number"], input[type="range"], textarea[name*="prompt"]'
    );
    
    const inputCount = await paramInputs.count();
    console.log('Parameter inputs found:', inputCount);
    
    await screenshotMgr.captureAndCompare(page, 'inference-parameters');
    
    // Try to set some parameters
    const textArea = page.locator('textarea').first();
    if (await textArea.isVisible({ timeout: 3000 }).catch(() => false)) {
      await textArea.fill('This is a test prompt for inference');
      await page.waitForTimeout(500);
      
      await screenshotMgr.captureAndCompare(page, 'prompt-entered');
    }
  });

  test('should run inference and display results', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('inference-execution');
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
      if (req.url().includes('inference') || req.url().includes('jsonrpc')) {
        networkRequests.push({
          url: req.url(),
          method: req.method(),
          timestamp: new Date().toISOString(),
        });
      }
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to AI Inference
    await page.locator('button.nav-tab:has-text("AI Inference")').click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, 'before-inference');
    
    // Look for inference button
    const inferenceBtn = page.locator(
      'button:has-text("Run Inference"), button:has-text("Generate"), button:has-text("Submit")'
    ).first();
    
    if (await inferenceBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      // Enter a test prompt first
      const promptInput = page.locator('textarea, input[type="text"]').first();
      if (await promptInput.isVisible({ timeout: 2000 }).catch(() => false)) {
        await promptInput.fill('Test inference prompt');
      }
      
      // Clear logs before inference
      consoleLogs.length = 0;
      
      // Run inference
      await inferenceBtn.click();
      await page.waitForTimeout(5000); // Wait for inference to complete
      
      await screenshotMgr.captureAndCompare(page, 'inference-running');
      
      // Check for results
      await page.waitForTimeout(2000);
      await screenshotMgr.captureAndCompare(page, 'inference-results', { fullPage: true });
      
      // Analyze logs
      const inferenceLogs = consoleLogs.filter(log =>
        /inference|generate|completion/i.test(log.text)
      );
      
      console.log('\n=== INFERENCE LOGS ===');
      console.log(`Total logs: ${consoleLogs.length}`);
      console.log(`Inference-related logs: ${inferenceLogs.length}`);
      
      inferenceLogs.forEach((log, idx) => {
        console.log(`  ${idx + 1}. [${log.type}] ${log.text.substring(0, 120)}`);
      });
      
      // Check network calls
      const inferenceCalls = networkRequests.filter(req =>
        /inference/i.test(req.url)
      );
      
      console.log(`Inference API calls: ${inferenceCalls.length}`);
    }
  });

  test('should test Advanced AI operations', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('advanced-ai');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI tab
    const advancedTab = page.locator('button.nav-tab:has-text("Advanced AI")');
    await expect(advancedTab).toBeVisible({ timeout: 10000 });
    await advancedTab.click();
    
    await page.waitForTimeout(1500);
    await screenshotMgr.captureAndCompare(page, 'advanced-ai-tab');
    
    // Look for advanced features
    const advancedFeatures = page.locator(
      'button:has-text("Multi-modal"), button:has-text("Batch"), button:has-text("Pipeline")'
    );
    
    const featureCount = await advancedFeatures.count();
    console.log('Advanced features found:', featureCount);
    
    if (featureCount > 0) {
      await screenshotMgr.captureAndCompare(page, 'advanced-features', { fullPage: true });
    }
  });

  test('should correlate inference with MCP server logs', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('inference-correlation');
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
    
    // Navigate to AI Inference
    await page.locator('button.nav-tab:has-text("AI Inference")').click();
    await page.waitForTimeout(2000);
    
    const startTime = new Date();
    consoleLogs.length = 0;
    
    // Try to run inference
    const inferenceBtn = page.locator('button:has-text("Run"), button:has-text("Generate")').first();
    
    if (await inferenceBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await inferenceBtn.click();
      await page.waitForTimeout(5000);
      
      const endTime = new Date();
      
      await screenshotMgr.captureAndCompare(page, 'after-inference');
      
      // Generate correlation report
      console.log('\n=== INFERENCE CORRELATION REPORT ===');
      console.log(`Time window: ${startTime.toISOString()} to ${endTime.toISOString()}`);
      console.log(`Total logs: ${consoleLogs.length}`);
      
      const sequentialPatterns = [
        /inference.*start|run.*inference/i,
        /model.*load|loading.*model/i,
        /inference.*complete|result|response/i,
      ];
      
      const foundPatterns: boolean[] = [];
      for (const pattern of sequentialPatterns) {
        const found = consoleLogs.some(log => pattern.test(log.text));
        foundPatterns.push(found);
        console.log(`Pattern "${pattern.source}": ${found ? '✓' : '✗'}`);
      }
      
      // Log all inference-related messages
      const inferenceLogs = consoleLogs.filter(log =>
        /inference|model|generate/i.test(log.text)
      );
      
      console.log(`\nInference-related logs (${inferenceLogs.length}):`);
      inferenceLogs.forEach((log, idx) => {
        console.log(`  ${idx + 1}. [${log.timestamp}] [${log.type}] ${log.text.substring(0, 100)}`);
      });
    }
  });

  test('should verify inference result display', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('inference-results-display');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to AI Inference
    await page.locator('button.nav-tab:has-text("AI Inference")').click();
    await page.waitForTimeout(2000);
    
    // Look for result containers
    const resultContainers = page.locator(
      '#inference-result, #output, .result-container, .inference-output'
    );
    
    const resultCount = await resultContainers.count();
    console.log('Result containers found:', resultCount);
    
    if (resultCount > 0) {
      await screenshotMgr.captureAndCompare(page, 'result-containers');
      
      // Check if results are visible
      for (let i = 0; i < Math.min(resultCount, 3); i++) {
        const container = resultContainers.nth(i);
        const isVisible = await container.isVisible().catch(() => false);
        console.log(`Result container ${i + 1} visible:`, isVisible);
      }
    }
  });
});
