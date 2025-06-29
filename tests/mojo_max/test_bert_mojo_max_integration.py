# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import os
import sys

# Add the generators directory to the Python path to allow imports
sys.path.append(os.path.join(os.getcwd(), 'generators'))

from models.skill_hf_bert_base_uncased import create_skill

def main():
    print("--- Test BertbaseuncasedSkill with Mojo/MAX Backend ---")

    # Instantiate the skill with device set to "mojo"
    # This should trigger the simulated Mojo processing via MojoMaxTargetMixin
    bert_skill = create_skill(device="mojo")

    # Sample input text
    sample_text = "Hello, world! This is a test for Mojo/MAX integration."

    # Process the text
    result = bert_skill.process(sample_text)

    print("\nProcessing Result:")
    print(f"  Backend: {result.get('backend')}")
    print(f"  Device: {result.get('device')}")
    print(f"  Model: {result.get('model')}")
    print(f"  Success: {result.get('success')}")
    print(f"  Outputs: {result.get('outputs')}")
    if not result.get('success'):
        print(f"  Message: {result.get('message')}")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()
