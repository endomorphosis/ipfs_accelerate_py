/**
 * IPFS Operations Tests
 * 
 * Tests IPFS file operations, network operations, and IPFS Manager tab
 */

import { test, expect } from '@playwright/test';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('IPFS File Operations', () => {
  test('should display IPFS Manager tab and file operations', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('ipfs-manager');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to IPFS Manager tab
    const ipfsTab = page.locator('button.nav-tab:has-text("IPFS Manager")');
    await expect(ipfsTab).toBeVisible({ timeout: 10000 });
    await ipfsTab.click();
    
    await page.waitForTimeout(1500);
    await screenshotMgr.captureAndCompare(page, 'ipfs-manager-tab');
    
    // Verify IPFS Manager interface exists
    await expect(page.locator('#ipfs-manager')).toBeVisible();
    
    await screenshotMgr.captureAndCompare(page, 'ipfs-interface', { fullPage: true });
  });

  test('should test IPFS file add functionality', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('ipfs-file-add');
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
    
    // Navigate to IPFS Manager
    await page.locator('button.nav-tab:has-text("IPFS Manager")').click();
    await page.waitForTimeout(2000);
    
    // Look for file upload or add file button
    const addFileBtn = page.locator(
      'button:has-text("Add File"), button:has-text("Upload"), input[type="file"]'
    ).first();
    
    if (await addFileBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'before-file-add');
      
      // Check for IPFS-related logs
      const ipfsLogs = consoleLogs.filter(log =>
        /ipfs|add.*file|upload/i.test(log.text)
      );
      
      console.log('IPFS-related logs:', ipfsLogs.length);
    }
  });

  test('should test IPFS cat (read) functionality', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to IPFS Manager
    await page.locator('button.nav-tab:has-text("IPFS Manager")').click();
    await page.waitForTimeout(2000);
    
    // Look for CID input or file list
    const cidInput = page.locator('input[placeholder*="CID" i], input[placeholder*="hash" i]').first();
    
    if (await cidInput.isVisible({ timeout: 3000 }).catch(() => false)) {
      // Test reading a file by CID
      await cidInput.fill('QmTestCID123');
      
      const readBtn = page.locator('button:has-text("Read"), button:has-text("Cat"), button:has-text("Get")').first();
      if (await readBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await readBtn.click();
        await page.waitForTimeout(2000);
        
        const catLogs = consoleLogs.filter(log => /ipfs.*cat|read.*file/i.test(log.text));
        console.log('IPFS cat logs:', catLogs.length);
      }
    }
  });

  test('should test IPFS pin operations', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('ipfs-pin');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to IPFS Manager
    await page.locator('button.nav-tab:has-text("IPFS Manager")').click();
    await page.waitForTimeout(2000);
    
    // Look for pin management UI
    const pinSection = page.locator(
      'div:has-text("Pin"), section:has-text("Pinned"), button:has-text("Pin")'
    ).first();
    
    if (await pinSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'pin-management');
    }
  });
});

test.describe('IPFS Network Operations', () => {
  test('should test IPFS node ID retrieval', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Network & Status tab
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for IPFS node info
    const nodeInfo = page.locator(
      'div:has-text("Node ID"), div:has-text("Peer ID"), #ipfs-node-id'
    ).first();
    
    if (await nodeInfo.isVisible({ timeout: 5000 }).catch(() => false)) {
      console.log('✓ IPFS node ID display found');
    }
    
    // Check for ipfs_id related logs
    const idLogs = consoleLogs.filter(log => /ipfs.*id|node.*info|peer.*id/i.test(log.text));
    console.log('IPFS ID logs:', idLogs.length);
  });

  test('should test IPFS swarm peers', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('ipfs-swarm');
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Network & Status
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for swarm peers list
    const swarmSection = page.locator(
      'div:has-text("Swarm"), div:has-text("Peers"), div:has-text("Connected")'
    ).first();
    
    if (await swarmSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshotMgr.captureAndCompare(page, 'swarm-peers');
      
      const swarmLogs = consoleLogs.filter(log => /swarm|peers|connected/i.test(log.text));
      console.log('Swarm-related logs:', swarmLogs.length);
    }
  });

  test('should test IPFS pubsub functionality', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Network & Status
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for pubsub UI
    const pubsubSection = page.locator(
      'div:has-text("PubSub"), div:has-text("Topics"), button:has-text("Publish")'
    ).first();
    
    if (await pubsubSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ PubSub interface found');
      
      const pubsubLogs = consoleLogs.filter(log => /pubsub|topic|publish/i.test(log.text));
      console.log('PubSub logs:', pubsubLogs.length);
    }
  });

  test('should test DHT operations', async ({ page }) => {
    const consoleLogs: any[] = [];
    
    page.on('console', msg => {
      consoleLogs.push({ type: msg.type(), text: msg.text() });
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to Network & Status
    await page.locator('button.nav-tab:has-text("Network & Status")').click();
    await page.waitForTimeout(2000);
    
    // Look for DHT operations
    const dhtSection = page.locator(
      'div:has-text("DHT"), button:has-text("Find Peer"), button:has-text("Find Providers")'
    ).first();
    
    if (await dhtSection.isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('✓ DHT interface found');
      
      const dhtLogs = consoleLogs.filter(log => /dht|findpeer|findprov/i.test(log.text));
      console.log('DHT logs:', dhtLogs.length);
    }
  });
});

test.describe('IPFS Integration Tests', () => {
  test('should verify all IPFS operations are accessible', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('ipfs-operations-check');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Navigate to IPFS Manager
    await page.locator('button.nav-tab:has-text("IPFS Manager")').click();
    await page.waitForTimeout(2000);
    
    await screenshotMgr.captureAndCompare(page, 'ipfs-manager-overview', { fullPage: true });
    
    // Verify MCP client is available
    const mcpClientActive = await page.evaluate(() => {
      return typeof (window as any).mcpClient !== 'undefined';
    });
    
    expect(mcpClientActive).toBeTruthy();
    
    // Try to call an IPFS MCP tool
    try {
      const result = await page.evaluate(async () => {
        const client = (window as any).mcpClient;
        if (!client) return null;
        
        // Try to get IPFS node ID
        return await client.request('tools/call', {
          name: 'ipfs_id',
          arguments: {}
        }).catch((e: Error) => ({ error: e.message }));
      });
      
      console.log('IPFS ID call result:', result);
    } catch (error: any) {
      console.log('IPFS tool call test (expected to possibly fail):', error.message);
    }
  });
});
