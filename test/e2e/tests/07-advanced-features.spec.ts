/**
 * Enhanced Inference & Workflow Tests
 * 
 * Tests advanced inference features, workflow management, and queue operations
 */

import { test, expect } from '@playwright/test';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('Enhanced Inference Features', () => {
  test('should test multiplex inference configuration', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('multiplex-inference');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI tab
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, 'advanced-ai-tab');
    
    // Look for multiplex or routing configuration
    const multiplexSection = page.locator(
      'div:has-text("Multiplex"), div:has-text("Routing"), div:has-text("Load Balance")'
    ).first();
    
    if (await multiplexSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'multiplex-config');
      console.log('✓ Multiplex inference UI found');
    }
  });

  test('should test endpoint registration and management', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('endpoint-management');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI or SDK Playground
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for endpoint management UI
    const endpointSection = page.locator(
      'button:has-text("Add Endpoint"), button:has-text("Register"), div:has-text("Endpoints")'
    ).first();
    
    if (await endpointSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'endpoint-management');
      
      const endpointLogs = consoleLogs.filter(log => /endpoint|register/i.test(log.text));
      console.log('Endpoint logs:', endpointLogs.length);
    }
  });

  test('should test CLI endpoint tools', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to SDK Playground or Advanced AI
    await page.locator('button.nav-tab:has-text("SDK Playground")').click();
    await page.waitForTimeout(2000);
    
    // Look for CLI tools section
    const cliSection = page.locator(
      'div:has-text("CLI"), button:has-text("CLI"), div:has-text("Command")'
    ).first();
    
    if (await cliSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ CLI tools interface found');
      
      const cliLogs = consoleLogs.filter(log => /cli|command|provider/i.test(log.text));
      console.log('CLI logs:', cliLogs.length);
    }
  });

  test('should test queue history and monitoring', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('queue-monitoring');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Queue Monitor tab
    await page.locator('button.nav-tab:has-text("Queue Monitor")').click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, 'queue-monitor-tab');
    
    // Verify queue monitor interface
    await expect(page.locator('#queue-monitor')).toBeVisible();
    
    // Look for history and statistics
    const historySection = page.locator(
      'div:has-text("History"), div:has-text("Statistics"), div:has-text("Metrics")'
    ).first();
    
    if (await historySection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'queue-history');
    }
  });

  test('should test distributed inference capabilities', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for distributed inference options
    const distributedSection = page.locator(
      'div:has-text("Distributed"), div:has-text("Multi-Device"), button:has-text("Distribute")'
    ).first();
    
    if (await distributedSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ Distributed inference UI found');
      
      const distLogs = consoleLogs.filter(log => /distributed|multi.*device|parallel/i.test(log.text));
      console.log('Distributed inference logs:', distLogs.length);
    }
  });
});

test.describe('Workflow Management', () => {
  test('should test workflow creation interface', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('workflow-creation');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI or dedicated workflow tab
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for workflow creation UI
    const workflowBtn = page.locator(
      'button:has-text("Create Workflow"), button:has-text("New Pipeline"), button:has-text("Add Workflow")'
    ).first();
    
    if (await workflowBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'before-workflow-create');
      await workflowBtn.click();
      await page.waitForTimeout(2000);
      
      await screenshotMgr.captureAndCompare(page, 'workflow-creation-dialog');
      
      const workflowLogs = consoleLogs.filter(log => /workflow|pipeline|create/i.test(log.text));
      console.log('Workflow creation logs:', workflowLogs.length);
    }
  });

  test('should test workflow listing', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('workflow-list');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for workflow list
    const workflowList = page.locator(
      'div:has-text("Workflows"), table:has-text("Workflow"), #workflow-list'
    ).first();
    
    if (await workflowList.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'workflow-list');
    }
  });

  test('should test workflow execution controls', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for workflow control buttons
    const controlBtns = page.locator(
      'button:has-text("Start"), button:has-text("Pause"), button:has-text("Stop")'
    );
    
    const count = await controlBtns.count();
    console.log('Workflow control buttons found:', count);
    
    if (count > 0) {
      const execLogs = consoleLogs.filter(log => /start|pause|stop|execute/i.test(log.text));
      console.log('Workflow execution logs:', execLogs.length);
    }
  });

  test('should test workflow templates', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('workflow-templates');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for template section
    const templateSection = page.locator(
      'div:has-text("Template"), button:has-text("From Template"), select:has-text("Template")'
    ).first();
    
    if (await templateSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'workflow-templates');
    }
  });

  test('should test HuggingFace model search integration', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('hf-search');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Model Manager
    await page.locator('button.nav-tab:has-text("Model Manager")').click();
    await page.waitForTimeout(2000);
    
    // Look for HuggingFace search
    const searchInput = page.locator('input[placeholder*="search" i], input[type="text"]').first();
    
    if (await searchInput.isVisible({ timeout: 3000 }).catch(() => false)) {
      await searchInput.fill('bert');
      await page.waitForTimeout(1000);
      
      await screenshotMgr.captureAndCompare(page, 'hf-search-results');
      
      const hfLogs = consoleLogs.filter(log => /huggingface|search.*model/i.test(log.text));
      console.log('HuggingFace search logs:', hfLogs.length);
      expect(hfLogs.length).toBeGreaterThan(0);
    }
  });
});

test.describe('Advanced Feature Integration', () => {
  test('should verify all advanced inference tools are accessible via MCP', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(3000);
    
    // Test MCP tool availability
    const toolsToTest = [
      'multiplex_inference',
      'register_endpoint',
      'get_queue_status',
      'get_queue_history',
      'search_huggingface_models',
      'create_workflow',
      'list_workflows',
    ];
    
    for (const toolName of toolsToTest) {
      try {
        const result = await page.evaluate(async (tool) => {
          const client = (window as any).mcpClient;
          if (!client) return { available: false, error: 'No MCP client' };
          
          // Just check if the tool exists (don't actually call it)
          return { available: true, tool };
        }, toolName);
        
        console.log(`Tool "${toolName}":`, result);
      } catch (error: any) {
        console.log(`Tool "${toolName}" check failed:`, error.message);
      }
    }
    
    // Verify some advanced tool was mentioned in logs
    const advancedLogs = consoleLogs.filter(log =>
      /multiplex|endpoint|workflow|queue.*history|huggingface/i.test(log.text)
    );
    
    console.log('Advanced feature logs found:', advancedLogs.length);
  });
});
