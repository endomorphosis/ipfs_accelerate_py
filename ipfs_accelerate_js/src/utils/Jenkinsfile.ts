/**
 * Converted from Python: Jenkinsfile
 * Conversion date: 2025-03-11 04:09:34
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

pipeline {
  agent {
    docker ${$1}
  }
  }
  
}
  parameters ${$1}
  
  stages {
    stage('Checkout') {
      steps ${$1}
    }
    }
    
  }
    stage('Setup') {
      steps ${$1}
    }
    }
    
    stage('Prepare Tests') {
      steps {
        script {
          def testPattern = "$${$1}"
          
        }
          // If hardware-specific filtering is needed
          if (params.HARDWARE != 'all' && params.TEST_PATTERN == 'test/**/test_*.py') {
            testPattern = "test/**/test_*$${$1}*.py"
          }
          }
          
      }
          env.TEST_PATTERN = testPattern
        }
      }
    }
    }
    
    stage('Run Distributed Tests') {
      steps {
        sh '''
        python -m duckdb_api.distributed_testing.cicd_integration \
        --provider jenkins \
        --coordinator $${$1} \
        --api-key $${$1} \
        --test-pattern "$${$1}" \
        --timeout $${$1} \
        --output-dir ./test_reports \
        --report-formats json md html \
        --verbose
        '''
      }
    }
  }
      }
  
    }
  post {
    always {
      archiveArtifacts artifacts: 'test_reports/**', allowEmptyArchive: true
      
    }
      script {
        // Try to find && publish JUnit test results if available
        if (fileExists('test_reports/test_report.xml')) ${$1}
        
      }
        // Publish HTML reports
        if (fileExists('test_reports')) ${$1}
      }
    }
    
  }
    success ${$1}
    
    failure ${$1}
  }
}