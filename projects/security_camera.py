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
This is an example project that detects when a person walks into a specific
region of the camera view.

As is, this code simply changes the color of the bounding-box drawn around the
person when they enter the fenced area. The code also makes some arbitrary
guesses about what proportion of the person's body must be in the fencee area
to be considered inside it. So you probably need to adjust these parameters to
suit your situation.

The TensorFlow model required is already downloaded if you flashed the
AIY Maker Kit system image for Raspberry Pi. Otherwise, you must download it
by running the download_models.sh script in this directory.

For more details about this project, see:
https://aiyprojects.withgoogle.com/maker/#guides--build-a-person-detecting-security-camera
"""

import os.path
from aiymakerkit import vision
from aiymakerkit import utils
from pycoral.adapters.detect import BBox
from pycoral.utils.dataset import read_label_file

# BGR (not RGB) colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def path(name):
    """ Creates an absolute path to a file in the same directory as this script."""
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, name)


# Load the TensorFlow Lite model (compiled for the Edge TPU)
OBJECT_DETECTION_MODEL = path('ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
detector = vision.Detector(OBJECT_DETECTION_MODEL)
labels = utils.read_labels_from_metadata(OBJECT_DETECTION_MODEL)

# Define the protected fence region
width, height = vision.VIDEO_SIZE
xmin = 0
ymin = 0
xmax = int(width * 0.5)
ymax = int(height * 0.5)
fence_box = BBox(xmin, ymin, xmax, ymax)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
    # Draw the fenced region
    vision.draw_rect(frame, fence_box, color=BLUE, thickness=3)

    # Detect only objects with at least 50% confidence
    objects = detector.get_objects(frame, threshold=0.5)

    # Review all detected objects and check for
    # objects that look like a person (ignore the rest)
    for obj in objects:
        label = labels.get(obj.id)
        if 'person' in label:
            # Get size of the person, the fence, and the overlap between them
            person_area = obj.bbox.area
            fence_area = fence_box.area
            overlap_area = BBox.intersect(obj.bbox, fence_box).area

            # Check if more than 30% of the person is in the fence,
            # OR if more than 50% of the fence is obscured by the object
            # (such as if the person is very close to the camera)
            if (overlap_area / person_area) > 0.3 or (
                    overlap_area / fence_area) > 0.5:
                # They are in the fence; draw their box red
                vision.draw_rect(frame, obj.bbox, color=RED)
            else:
                # They are outside the fence; draw them green
                vision.draw_rect(frame, obj.bbox, color=GREEN)
