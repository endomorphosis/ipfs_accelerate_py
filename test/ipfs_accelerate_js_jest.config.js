/** @type {import('jest').Config} */
module.exports = {
  // The root directory that Jest should scan for tests and modules
  rootDir: '.',
  
  // A list of paths to directories that Jest should use to search for files in
  roots: ['<rootDir>/src/', '<rootDir>/test/'],
  
  // The glob patterns Jest uses to detect test files
  testMatch: [
    '**/__tests__/**/*.ts?(x)',
    '**/?(*.)+(spec|test).ts?(x)'
  ],
  
  // An array of regexp pattern strings that are matched against all test paths, matched tests are skipped
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  
  // An array of regexp pattern strings that are matched against all source file paths
  // matched files will be covered by the transpiler
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest'
  },
  
  // An array of regexp pattern strings that are matched against all source file paths
  // matched files will not be covered by the transpiler
  transformIgnorePatterns: ['/node_modules/'],
  
  // The directory where Jest should output its coverage files
  coverageDirectory: 'coverage',
  
  // An array of regexp pattern strings used to skip coverage collection
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/test/',
    '/dist/'
  ],
  
  // Indicates which provider should be used to instrument code for coverage
  coverageProvider: 'v8',
  
  // Make calling deprecated APIs throw helpful error messages
  errorOnDeprecated: true,
  
  // Default timeout of a test in milliseconds
  testTimeout: 5000,
  
  // A preset that is used as a base for Jest's configuration
  preset: 'ts-jest',
  
  // The test environment that will be used for testing
  testEnvironment: 'jsdom',
  
  // Setup files run before each test
  setupFilesAfterEnv: ['<rootDir>/test/setup.ts'],
  
  // Custom resolvers
  moduleNameMapper: {
    '^src/(.*)$': '<rootDir>/src/$1',
    '^test/(.*)$': '<rootDir>/test/$1'
  },
  
  // Configure Jest to use Babel
  globals: {
    'ts-jest': {
      tsconfig: '<rootDir>/tsconfig.json',
      isolatedModules: true
    }
  },
  
  // Generate coverage report in text, lcov, and html formats
  coverageReporters: ['text', 'lcov', 'html'],
  
  // Configure which files Jest should collect coverage from
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/index.ts',
    '!src/**/*.stories.{ts,tsx}',
    '!src/types/**'
  ],
  
  // Minimum threshold enforcement for coverage results
  coverageThreshold: {
    global: {
      statements: 70,
      branches: 70,
      functions: 70,
      lines: 70
    }
  },
  
  // A list of reporter names that Jest uses when writing coverage reports
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: './reports/junit',
      outputName: 'jest-junit.xml'
    }]
  ]
};