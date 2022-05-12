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

"""
Performs continuous face detection with the camera.

Simply run the script and it will draw boxes around detected faces:

    python3 detect_faces.py

For more instructions, see g.co/aiy/maker
"""

from aiymakerkit import vision
import models

detector = vision.Detector(models.FACE_DETECTION_MODEL)

for frame in vision.get_frames():
    faces = detector.get_objects(frame, threshold=0.1)
    vision.draw_objects(frame, faces)
