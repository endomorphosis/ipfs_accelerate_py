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

from max.graph import Graph, TensorType, ops
from max.dtype import DType
from max.engine import InferenceSession
from sys import os

fn main():
    print("--- Test 1: Default Backend (CPU/CUDA/etc.) ---")
    # Ensure the environment variable is not set for this test
    os.unsetenv("USE_MOJO_MAX_TARGET")

    # Define a simple graph with a matmul operation
    var graph = Graph(
        "test_matmul_default",
        input_types=[
            TensorType(DType.float32, (2, 2)),
            TensorType(DType.float32, (2, 2)),
        ],
    )

    with graph:
        var result = ops.matmul(graph.inputs[0], graph.inputs[1])
        graph.output(result)

    # Execute the graph
    var session = InferenceSession(graph)
    var input_a = DType.float32.tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    var input_b = DType.float32.tensor([2, 2], [5.0, 6.0, 7.0, 8.0])
    var outputs = session.execute([input_a, input_b])
    print("Default matmul executed. Output shape:", outputs[0].shape)
    print("-------------------------------------------\n")

    print("--- Test 2: Mojo/MAX Backend ---")
    # Set the environment variable to trigger Mojo/MAX backend
    os.setenv("USE_MOJO_MAX_TARGET", "1")

    # Define the same simple graph
    var graph_mojo_max = Graph(
        "test_matmul_mojo_max",
        input_types=[
            TensorType(DType.float32, (2, 2)),
            TensorType(DType.float32, (2, 2)),
        ],
    )

    with graph_mojo_max:
        var result_mojo_max = ops.matmul(graph_mojo_max.inputs[0], graph_mojo_max.inputs[1])
        graph_mojo_max.output(result_mojo_max)

    # Execute the graph
    var session_mojo_max = InferenceSession(graph_mojo_max)
    var input_a_mojo_max = DType.float32.tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    var input_b_mojo_max = DType.float32.tensor([2, 2], [5.0, 6.0, 7.0, 8.0])
    var outputs_mojo_max = session_mojo_max.execute([input_a_mojo_max, input_b_mojo_max])
    print("Mojo/MAX matmul executed. Output shape:", outputs_mojo_max[0].shape)
    print("-------------------------------------------\n")

    # Clean up the environment variable
    os.unsetenv("USE_MOJO_MAX_TARGET")
