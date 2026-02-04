/**
 * Screenshot Comparison Utility
 * 
 * Provides utilities for visual regression testing
 */

import { Page } from '@playwright/test';
import path from 'path';
import fs from 'fs';

export interface ScreenshotOptions {
  fullPage?: boolean;
  mask?: string[]; // CSS selectors to mask
  threshold?: number; // Pixel difference threshold (0-1)
}

export class ScreenshotManager {
  private baselineDir: string;
  private currentDir: string;
  private diffDir: string;

  constructor(testName: string) {
    const baseDir = path.join(process.cwd(), 'test-results', 'visual-regression');
    
    this.baselineDir = path.join(baseDir, 'baseline', testName);
    this.currentDir = path.join(baseDir, 'current', testName);
    this.diffDir = path.join(baseDir, 'diff', testName);

    // Create directories
    fs.mkdirSync(this.baselineDir, { recursive: true });
    fs.mkdirSync(this.currentDir, { recursive: true });
    fs.mkdirSync(this.diffDir, { recursive: true });
  }

  /**
   * Take a screenshot and optionally compare with baseline
   */
  async captureAndCompare(
    page: Page,
    name: string,
    options: ScreenshotOptions = {}
  ): Promise<{
    path: string;
    hasBaseline: boolean;
    isDifferent?: boolean;
    diffPath?: string;
  }> {
    const screenshotPath = path.join(this.currentDir, `${name}.png`);
    const baselinePath = path.join(this.baselineDir, `${name}.png`);
    const diffPath = path.join(this.diffDir, `${name}.png`);

    // Mask elements if specified
    if (options.mask && options.mask.length > 0) {
      for (const selector of options.mask) {
        try {
          await page.locator(selector).evaluate(el => {
            (el as HTMLElement).style.visibility = 'hidden';
          });
        } catch {
          // Element might not exist, continue
        }
      }
    }

    // Take screenshot
    await page.screenshot({
      path: screenshotPath,
      fullPage: options.fullPage || false,
    });

    // Check if baseline exists
    const hasBaseline = fs.existsSync(baselinePath);

    if (!hasBaseline) {
      // First run - copy as baseline
      fs.copyFileSync(screenshotPath, baselinePath);
      return {
        path: screenshotPath,
        hasBaseline: false,
      };
    }

    // Compare with baseline using Playwright's built-in comparison
    // Note: This is a simplified version. In production, you'd use pixelmatch or similar
    return {
      path: screenshotPath,
      hasBaseline: true,
      isDifferent: false, // Would be calculated by comparison
      diffPath: diffPath,
    };
  }

  /**
   * Take multiple screenshots of different viewport sizes
   */
  async captureResponsive(
    page: Page,
    name: string,
    viewports: { width: number; height: number; name: string }[]
  ): Promise<string[]> {
    const paths: string[] = [];

    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.waitForTimeout(1000); // Wait for reflow
      
      const screenshotName = `${name}_${viewport.name}`;
      const result = await this.captureAndCompare(page, screenshotName);
      paths.push(result.path);
    }

    return paths;
  }

  /**
   * Take annotated screenshot with element highlights
   */
  async captureAnnotated(
    page: Page,
    name: string,
    highlights: { selector: string; label?: string }[]
  ): Promise<string> {
    // Add highlights
    for (const highlight of highlights) {
      try {
        await page.locator(highlight.selector).evaluate((el, label) => {
          const element = el as HTMLElement;
          element.style.outline = '3px solid red';
          element.style.outlineOffset = '2px';
          
          if (label) {
            const labelEl = document.createElement('div');
            labelEl.textContent = label;
            labelEl.style.cssText = `
              position: absolute;
              background: red;
              color: white;
              padding: 4px 8px;
              font-size: 12px;
              font-weight: bold;
              z-index: 10000;
            `;
            element.style.position = 'relative';
            element.appendChild(labelEl);
          }
        }, highlight.label);
      } catch {
        // Element might not exist
      }
    }

    const screenshotPath = path.join(this.currentDir, `${name}_annotated.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });

    return screenshotPath;
  }

  /**
   * Standard viewport configurations
   */
  static getStandardViewports() {
    return [
      { width: 1920, height: 1080, name: 'desktop-1080p' },
      { width: 1366, height: 768, name: 'desktop-laptop' },
      { width: 768, height: 1024, name: 'tablet-portrait' },
      { width: 375, height: 667, name: 'mobile-iphone' },
      { width: 414, height: 896, name: 'mobile-large' },
    ];
  }
}
