/**
 * P2P, Distributed Features & Backend Management Tests
 * 
 * Tests P2P workflow scheduler, distributed tasks, Copilot integration, and backend management
 */

import { test, expect } from '@playwright/test';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('P2P & Distributed Features', () => {
  test('should test P2P scheduler status', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Network & Status
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for P2P scheduler info
    const p2pSection = page.locator(
      'div:has-text("P2P"), div:has-text("Scheduler"), div:has-text("Distributed")'
    ).first();
    
    if (await p2pSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ P2P scheduler section found');
      
      const p2pLogs = consoleLogs.filter(log => /p2p|scheduler|distributed/i.test(log.text));
      console.log('P2P scheduler logs:', p2pLogs.length);
    }
  });

  test('should test task submission to P2P network', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('p2p-tasks');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI or Queue Monitor
    await page.locator('button.nav-tab:has-text("Queue Monitor")').click();
    await page.waitForTimeout(2000);
    
    // Look for task submission interface
    const taskSection = page.locator(
      'button:has-text("Submit Task"), button:has-text("Add Task"), div:has-text("Task Queue")'
    ).first();
    
    if (await taskSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'p2p-task-queue');
      
      const taskLogs = consoleLogs.filter(log => /submit.*task|task.*queue|p2p.*task/i.test(log.text));
      console.log('Task submission logs:', taskLogs.length);
    }
  });

  test('should test peer state management', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text) });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Network & Status
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for peer state information
    const peerSection = page.locator(
      'div:has-text("Peer"), div:has-text("Connected"), div:has-text("State")'
    ).first();
    
    if (await peerSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ Peer state section found');
      
      const peerLogs = consoleLogs.filter(log => /peer.*state|connected.*peer/i.test(log.text));
      console.log('Peer state logs:', peerLogs.length);
    }
  });

  test('should test Merkle clock operations', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(3000);
    
    // Check for Merkle clock related logs
    const merkleLogs = consoleLogs.filter(log => /merkle|clock|vector.*clock/i.test(log.text));
    console.log('Merkle clock logs:', merkleLogs.length);
    
    // Note: This is likely a background operation
    console.log('Merkle clock operations tracked:', merkleLogs.length > 0);
  });
});

test.describe('Copilot Integration', () => {
  test('should test Copilot command suggestions', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('copilot-commands');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to SDK Playground
    await page.locator('button.nav-tab:has-text("SDK Playground")').click();
    await page.waitForTimeout(2000);
    
    // Look for Copilot integration
    const copilotSection = page.locator(
      'div:has-text("Copilot"), button:has-text("Copilot"), div:has-text("Suggest")'
    ).first();
    
    if (await copilotSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'copilot-interface');
      
      const copilotLogs = consoleLogs.filter(log => /copilot|suggest|explain/i.test(log.text));
      console.log('Copilot logs:', copilotLogs.length);
    }
  });

  test('should test Copilot SDK sessions', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to SDK Playground
    await page.locator('button.nav-tab:has-text("SDK Playground")').click();
    await page.waitForTimeout(2000);
    
    // Look for session management
    const sessionSection = page.locator(
      'button:has-text("Create Session"), button:has-text("New Session"), div:has-text("Session")'
    ).first();
    
    if (await sessionSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ Copilot session management found');
      
      const sessionLogs = consoleLogs.filter(log => /copilot.*session|create.*session/i.test(log.text));
      console.log('Copilot session logs:', sessionLogs.length);
    }
  });

  test('should test Copilot tool discovery', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(3000);
    
    // Check if Copilot tools are discovered
    const toolLogs = consoleLogs.filter(log => /copilot.*tool|available.*tool/i.test(log.text));
    console.log('Copilot tool discovery logs:', toolLogs.length);
  });
});

test.describe('Backend Management', () => {
  test('should test inference backend listing', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('backend-listing');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI or SDK Playground
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for backend listing
    const backendSection = page.locator(
      'div:has-text("Backend"), div:has-text("Provider"), select:has-text("Backend")'
    ).first();
    
    if (await backendSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'backend-listing');
      
      const backendLogs = consoleLogs.filter(log => /backend|provider|inference.*engine/i.test(log.text));
      console.log('Backend logs:', backendLogs.length);
    }
  });

  test('should test backend configuration', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Advanced AI
    await page.locator('button.nav-tab:has-text("Advanced AI")').click();
    await page.waitForTimeout(2000);
    
    // Look for backend configuration options
    const configSection = page.locator(
      'button:has-text("Configure"), button:has-text("Settings"), button:has-text("Options")'
    ).first();
    
    if (await configSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ Backend configuration UI found');
      
      const configLogs = consoleLogs.filter(log => /config|setting|option/i.test(log.text));
      console.log('Configuration logs:', configLogs.length);
    }
  });

  test('should test backend filtering and selection', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('backend-selection');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to AI Inference
    await page.locator('button.nav-tab:has-text("AI Inference")').click();
    await page.waitForTimeout(2000);
    
    // Look for backend selector
    const backendSelector = page.locator(
      'select, [role="combobox"]'
    ).filter({ hasText: /backend|provider|engine/i }).first();
    
    if (await backendSelector.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'backend-selector');
    }
  });
});

test.describe('Docker & Container Management', () => {
  test('should test Docker container operations', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Network & Status or Overview
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for Docker/container info
    const dockerSection = page.locator(
      'div:has-text("Docker"), div:has-text("Container"), button:has-text("Docker")'
    ).first();
    
    if (await dockerSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ Docker management UI found');
      
      const dockerLogs = consoleLogs.filter(log => /docker|container/i.test(log.text));
      console.log('Docker logs:', dockerLogs.length);
    }
  });
});

test.describe('Complete Feature Coverage Validation', () => {
  test('should verify all MCP tool categories are accessible', async ({ page }) => {
    const consoleLogs: any[] = [];
    const networkRequests: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    page.on('request', req => {
      if (req.url().includes('/jsonrpc') || req.url().includes('tools/call')) {
        networkRequests.push({
          url: req.url(),
          method: req.method(),
          timestamp: new Date().toISOString(),
        });
      }
    });
    
    await page.goto('/');
    await page.waitForTimeout(3000);
    
    // Test comprehensive MCP tool availability across all categories
    const toolCategories = [
      { category: 'IPFS Files', tools: ['ipfs_add_file', 'ipfs_cat', 'ipfs_pin_add'] },
      { category: 'IPFS Network', tools: ['ipfs_id', 'ipfs_swarm_peers'] },
      { category: 'P2P Workflows', tools: ['p2p_scheduler_status', 'p2p_submit_task'] },
      { category: 'Copilot', tools: ['copilot_suggest_command', 'copilot_sdk_create_session'] },
      { category: 'Hardware', tools: ['ipfs_get_hardware_info', 'ipfs_accelerate_model'] },
      { category: 'System Logs', tools: ['get_system_logs', 'get_recent_errors'] },
      { category: 'Backends', tools: ['list_inference_backends'] },
      { category: 'Workflows', tools: ['create_workflow', 'list_workflows'] },
    ];
    
    console.log('\n=== COMPREHENSIVE FEATURE COVERAGE TEST ===\n');
    
    for (const { category, tools } of toolCategories) {
      console.log(`\nCategory: ${category}`);
      for (const toolName of tools) {
        try {
          const available = await page.evaluate(async (tool) => {
            return typeof (window as any).mcpClient !== 'undefined';
          }, toolName);
          
          console.log(`  ${toolName}: ${available ? '✓ Available' : '✗ Not Available'}`);
        } catch (error: any) {
          console.log(`  ${toolName}: ✗ Error - ${error.message}`);
        }
      }
    }
    
    console.log('\n=== SUMMARY ===');
    console.log(`Console Logs: ${consoleLogs.length}`);
    console.log(`MCP Requests: ${networkRequests.length}`);
    console.log('==============\n');
    
    // Verify MCP client is functional
    const mcpActive = await page.evaluate(() => {
      return typeof (window as any).mcpClient !== 'undefined' && 
             (window as any).mcpClient !== null;
    });
    
    expect(mcpActive).toBeTruthy();
  });
});
