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

# Installs all the TensorFlow Lite models used by the examples.

set -e

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TEST_DATA_URL="https://github.com/google-coral/test_data/raw/master/"
readonly MODEL_DIR="${SCRIPT_DIR}/models"

if [[ -d "${MODEL_DIR}" ]]; then
  echo "Models directory exists. Skipping downloads."
  exit 1
fi

mkdir -p "${MODEL_DIR}"

echo "Downloading model files..."
(
  cd "${MODEL_DIR}" &&
    curl -OL "${TEST_DATA_URL}/tf2_mobilenet_v2_1.0_224_ptq_edgetpu.tflite" \
      -OL "${TEST_DATA_URL}/imagenet_labels.txt" \
      -OL "${TEST_DATA_URL}/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" \
      -OL "${TEST_DATA_URL}/coco_labels.txt" \
      -OL "${TEST_DATA_URL}/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite" \
      -OL "${TEST_DATA_URL}/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite" \
      -OL "${TEST_DATA_URL}/movenet_single_pose_lightning_ptq_edgetpu.tflite"
)
echo "Done."
