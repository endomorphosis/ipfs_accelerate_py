/**
 * Jest configuration for WebGPU optimizer tests
 */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: [
    '**/webgpu_optimizer/test_*.ts'
  ],
  globals: {
    'ts-jest': {
      tsconfig: '../../../tsconfig.json'
    }
  },
  moduleNameMapper: {
    '^src/(.*)$': '<rootDir>/../../../src/$1'
  },
  testTimeout: 60000, // Longer timeout for performance tests
  verbose: true,
  collectCoverage: false,
  setupFilesAfterEnv: ['<rootDir>/test_setup.js']
};