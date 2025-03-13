/**
 * GPU detection utilities
 */

export interface GPUCapabilities {
  browserName?: string;
  browserVersion?: string;
  isMobile?: boolean;
  platform?: string;
  osVersion?: string;
  metalApiSupported?: boolean;
  metalApiVersion?: string;
  gpu: {
    vendor: string;
    model: string;
    capabilities: Record<string, boolean>;
  };
  webgpu: {
    supported: boolean;
    features: string[];
  };
  wasm: {
    supported: boolean;
    simd: boolean;
    threads: boolean;
    features?: string[];
  };
}

export async function detectGPUCapabilities(): Promise<GPUCapabilities> {
  // Placeholder implementation
  const isWebGPUSupported = !!navigator.gpu;
  
  return {
    browserName: getBrowserName(),
    browserVersion: getBrowserVersion(),
    isMobile: isMobileDevice(),
    platform: getPlatform(),
    osVersion: getOSVersion(),
    gpu: {
      vendor: 'unknown',
      model: 'unknown',
      capabilities: {
        computeShaders: isWebGPUSupported,
        parallelCompilation: true
      }
    },
    webgpu: {
      supported: isWebGPUSupported,
      features: isWebGPUSupported ? ['basic', 'compute'] : []
    },
    wasm: {
      supported: typeof WebAssembly !== 'undefined',
      simd: false,
      threads: false
    }
  };
}

// Helper functions to detect browser info
function getBrowserName(): string {
  const userAgent = navigator.userAgent;
  
  if (userAgent.indexOf('Firefox') > -1) {
    return 'Firefox';
  } else if (userAgent.indexOf('Edge') > -1 || userAgent.indexOf('Edg/') > -1) {
    return 'Edge';
  } else if (userAgent.indexOf('Chrome') > -1) {
    return 'Chrome';
  } else if (userAgent.indexOf('Safari') > -1) {
    return 'Safari';
  }
  
  return 'Unknown';
}

function getBrowserVersion(): string {
  const userAgent = navigator.userAgent;
  let match;
  
  if ((match = userAgent.match(/(Firefox|Chrome|Safari|Edge|Edg)\/(\d+\.\d+)/))) {
    return match[2];
  }
  
  return '0';
}

function isMobileDevice(): boolean {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function getPlatform(): string {
  const userAgent = navigator.userAgent;
  
  if (userAgent.indexOf('Win') > -1) {
    return 'Windows';
  } else if (userAgent.indexOf('Mac') > -1) {
    return 'macOS';
  } else if (userAgent.indexOf('Linux') > -1) {
    return 'Linux';
  } else if (userAgent.indexOf('Android') > -1) {
    return 'Android';
  } else if (userAgent.indexOf('iPhone') > -1 || userAgent.indexOf('iPad') > -1) {
    return 'iOS';
  }
  
  return 'Unknown';
}

function getOSVersion(): string {
  const userAgent = navigator.userAgent;
  let match;
  
  if ((match = userAgent.match(/(Windows NT|Mac OS X|Android|iOS) ([\d\.]+)/))) {
    return match[2];
  }
  
  return '0';
}
