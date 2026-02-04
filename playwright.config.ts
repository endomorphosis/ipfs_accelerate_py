import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright Configuration for IPFS Accelerate Dashboard E2E Tests
 * 
 * This configuration supports comprehensive end-to-end testing including:
 * - Screenshot capture
 * - Console log validation
 * - Video recording
 * - Log correlation with MCP server
 */
export default defineConfig({
  testDir: './e2e',
  
  // Maximum time one test can run
  timeout: 120 * 1000,
  
  // Test execution settings
  fullyParallel: false, // Run tests sequentially to avoid port conflicts
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : 1,
  
  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'test-results/html-report' }],
    ['json', { outputFile: 'test-results/test-results.json' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
    ['list'],
  ],
  
  // Shared settings for all projects
  use: {
    // Base URL for the dashboard
    baseURL: process.env.DASHBOARD_URL || 'http://localhost:3001',
    
    // Collect trace on failure
    trace: 'on-first-retry',
    
    // Screenshot settings
    screenshot: 'only-on-failure',
    
    // Video settings
    video: 'retain-on-failure',
    
    // Action timeout
    actionTimeout: 15 * 1000,
    
    // Navigation timeout
    navigationTimeout: 30 * 1000,
  },
  
  // Configure projects for different browsers
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        // Capture console logs
        launchOptions: {
          args: [
            '--enable-logging',
            '--v=1',
          ],
        },
      },
    },
    
    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        viewport: { width: 1920, height: 1080 },
      },
    },
    
    {
      name: 'webkit',
      use: {
        ...devices['Desktop Safari'],
        viewport: { width: 1920, height: 1080 },
      },
    },
    
    // Mobile viewports for responsive testing
    {
      name: 'mobile-chrome',
      use: {
        ...devices['Pixel 5'],
      },
    },
    
    {
      name: 'mobile-safari',
      use: {
        ...devices['iPhone 12'],
      },
    },
  ],
  
  // Web server configuration for local testing
  webServer: {
    command: 'python -m ipfs_accelerate_py.mcp_dashboard --port 3001',
    url: 'http://localhost:3001',
    timeout: 120 * 1000,
    reuseExistingServer: !process.env.CI,
    stdout: 'pipe',
    stderr: 'pipe',
    env: {
      PYTHONUNBUFFERED: '1',
      MCP_SERVER_PORT: '3001',
      MCP_SERVER_HOST: 'localhost',
    },
  },
  
  // Output directories
  outputDir: 'test-results',
});
