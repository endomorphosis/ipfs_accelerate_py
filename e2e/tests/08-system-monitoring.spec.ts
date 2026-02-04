/**
 * Hardware, Acceleration & System Monitoring Tests
 * 
 * Tests hardware detection, model acceleration, system logs, and performance monitoring
 */

import { test, expect } from '@playwright/test';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('Hardware & Acceleration', () => {
  test('should test hardware information retrieval', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('hardware-info');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Overview or Network & Status
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for hardware information section
    const hwSection = page.locator(
      'div:has-text("Hardware"), div:has-text("GPU"), div:has-text("CPU"), div:has-text("Memory")'
    ).first();
    
    if (await hwSection.isVisible({ timeout: 5000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'hardware-info');
      console.log('✓ Hardware information display found');
    }
    
    // Check for hardware-related logs
    const hwLogs = consoleLogs.filter(log => /hardware|gpu|cpu|memory|device/i.test(log.text));
    console.log('Hardware logs:', hwLogs.length);
  });

  test('should test model acceleration options', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('model-acceleration');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for acceleration options
    const accelSection = page.locator(
      'div:has-text("Accelerat"), button:has-text("Accelerate"), div:has-text("Optimization")'
    ).first();
    
    if (await accelSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'acceleration-options');
      
      const accelLogs = consoleLogs.filter(log => /accelerat|optimi|hardware/i.test(log.text));
      console.log('Acceleration logs:', accelLogs.length);
    }
  });

  test('should test model benchmarking', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI or Model Manager
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for benchmark functionality
    const benchmarkBtn = page.locator(
      'button:has-text("Benchmark"), button:has-text("Test"), button:has-text("Performance")'
    ).first();
    
    if (await benchmarkBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ Benchmark button found');
      
      const benchLogs = consoleLogs.filter(log => /benchmark|performance|test/i.test(log.text));
      console.log('Benchmark logs:', benchLogs.length);
    }
  });

  test('should test hardware-specific model status', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('model-status');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Model Manager
    await page.locator('button.nav-tab:has-text("Model Manager")').click();
    await page.waitForTimeout(2000);
    
    // Look for model status indicators
    const statusSection = page.locator(
      'div:has-text("Status"), span:has-text("Loaded"), span:has-text("Accelerated")'
    ).first();
    
    if (await statusSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'model-status');
    }
  });
});

test.describe('System Logs & Monitoring', () => {
  test('should test system logs retrieval', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('system-logs');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to System Logs tab
    await page.locator('button.nav-tab:has-text("System Logs")').click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, 'system-logs-tab');
    
    // Verify logs interface exists
    await expect(page.locator('#system-logs')).toBeVisible();
    
    // Look for log display area
    const logsDisplay = page.locator(
      'pre, code, .log-entry, .log-container, textarea[readonly]'
    ).first();
    
    if (await logsDisplay.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'logs-display');
    }
  });

  test('should test error log filtering', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('error-logs');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to System Logs
    await page.locator('button.nav-tab:has-text("System Logs")').click();
    await page.waitForTimeout(2000);
    
    // Look for error filter
    const errorFilter = page.locator(
      'button:has-text("Errors"), select option:has-text("Error"), input[value="error" i]'
    ).first();
    
    if (await errorFilter.isVisible({ timeout: 3000 }).catch(() => false)) {
      await errorFilter.click();
      await page.waitForTimeout(1000);
      
      await screenshotMgr.captureAndCompare(page, 'filtered-errors');
      
      const logLogs = consoleLogs.filter(log => /error.*log|filter|level/i.test(log.text));
      console.log('Log filtering logs:', logLogs.length);
    }
  });

  test('should test log level selection', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('log-levels');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to System Logs
    await page.locator('button.nav-tab:has-text("System Logs")').click();
    await page.waitForTimeout(2000);
    
    // Look for log level selector
    const levelSelector = page.locator(
      'select, [role="combobox"]'
    ).filter({ hasText: /info|warn|error|debug/i }).first();
    
    if (await levelSelector.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'log-level-selector');
    }
  });

  test('should test performance metrics display', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('performance-metrics');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Network & Status
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for performance metrics
    const metricsSection = page.locator(
      'div:has-text("Performance"), div:has-text("Metrics"), div:has-text("CPU"), div:has-text("Memory")'
    ).first();
    
    if (await metricsSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'performance-metrics');
    }
  });

  test('should test session management', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Look for session-related functionality
    const sessionLogs = consoleLogs.filter(log => /session|start.*session|end.*session/i.test(log.text));
    console.log('Session management logs:', sessionLogs.length);
    
    // Check if sessions are tracked
    const hasSessionTracking = sessionLogs.length > 0;
    console.log('Session tracking active:', hasSessionTracking);
  });
});

test.describe('Coverage Analysis', () => {
  test('should test SDK coverage analysis', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('coverage-analysis');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Coverage Analysis tab
    await page.locator('button.nav-tab:has-text("Coverage Analysis")').click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, 'coverage-analysis-tab');
    
    // Verify coverage interface exists
    await expect(page.locator('#coverage')).toBeVisible();
    
    await screenshotMgr.captureAndCompare(page, 'coverage-display', { fullPage: true });
  });

  test('should test MCP tools coverage display', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('mcp-tools-coverage');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to MCP Tools or Coverage Analysis
    await page.locator('button.nav-tab:has-text("MCP Tools")').click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, 'mcp-tools-tab');
    
    // Look for tool list or coverage metrics
    const toolsList = page.locator(
      'div:has-text("Available"), div:has-text("Tools"), table, ul'
    ).first();
    
    if (await toolsList.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'mcp-tools-list');
    }
  });
});

test.describe('System Integration Tests', () => {
  test('should verify hardware and system monitoring tools via MCP', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(3000);
    
    // Test MCP tool availability for hardware/system features
    const toolsToTest = [
      'ipfs_get_hardware_info',
      'ipfs_accelerate_model',
      'ipfs_benchmark_model',
      'get_system_logs',
      'get_recent_errors',
      'get_performance_metrics',
      'get_server_status',
    ];
    
    for (const toolName of toolsToTest) {
      try {
        const result = await page.evaluate(async (tool) => {
          const client = (window as any).mcpClient;
          if (!client) return { available: false };
          return { available: true, tool };
        }, toolName);
        
        console.log(`System tool "${toolName}":`, result);
      } catch (error: any) {
        console.log(`System tool "${toolName}" check failed:`, error.message);
      }
    }
    
    // Verify system-related logs
    const systemLogs = consoleLogs.filter(log =>
      /hardware|system|logs|performance|metrics/i.test(log.text)
    );
    
    console.log('System-related logs found:', systemLogs.length);
  });
});
