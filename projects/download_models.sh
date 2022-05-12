#!/bin/bash
#
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Downloads TensorFlow Lite models for the projects.

set -e

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TEST_DATA_URL="https://github.com/google-coral/test_data/raw/master/"

echo "Downloading model files..."
(
  cd "${SCRIPT_DIR}" &&
    curl \
      -OL "${TEST_DATA_URL}/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" \
      -OL "${TEST_DATA_URL}/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
)
echo "Done."
