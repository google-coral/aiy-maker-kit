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
Performs one image classification at a time, either from a camera or file.

To capture an image from the camera and classify it using our MobileNet model,
simply run the script (and then press the Spacebar to capture an image):

    python3 classify_image.py

Or classify from the camera using your own model:

    python3 classify_image.py -m my_model.tflite

And to classify an existing image file, just pass your image:

    python3 classify_image.py -i my_image.jpg

For information about the script options, run:

    python3 classify_image.py --help

For more instructions, see g.co/aiy/maker
"""

import argparse
import contextlib
import select
import sys
import termios
import tty
from cv2 import imread
from pycoral.utils.dataset import read_label_file
from aiymakerkit import vision
from aiymakerkit.utils import read_labels_from_metadata
import models


@contextlib.contextmanager
def nonblocking(f):
    """Context manager to listen for key events."""
    def get_char():
        if select.select([f], [], [], 0) == ([f], [], []):
            return sys.stdin.read(1)
        return None

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(f.fileno())
        yield get_char
    finally:
        termios.tcsetattr(f, termios.TCSADRAIN, old_settings)


def classify_image(classifier, labels, frame):
    """
    Classify an image and print the top result.

    Args:
      classifier: A ``vision.Classifier`` object.
      labels: The labels file for the Classifier model.
      frame: The image to classify.
    Returns:
      A list of all class predictions, ordered by score.
    """
    classes = classifier.get_classes(frame)
    label_id = classes[0].id
    score = classes[0].score
    label = labels.get(label_id)
    print(label, score)
    return classes


def classify_live(classifier, labels):
    """
    Wait for the Spacebar key event, and then capture an image from the
    camera and classify it with ``classify_image()``.
    """
    with nonblocking(sys.stdin) as get_char:
        # Handle key events from GUI window.
        def handle_key(key, frame):
            if key == 32:  # Spacebar
                classify_image(classifier, labels, frame)
            if key == ord('q') or key == ord('Q'):
                return False  # Quit the program
            return True  # Keep the camera alive, wait for keys

        first_pass = True
        for frame in vision.get_frames(handle_key=handle_key):
            if first_pass:
                print(
                    'Press the spacebar to capture and classify an image from your camera.')
                first_pass = False
            # Handle key events from console.
            ch = get_char()
            if ch is not None and not handle_key(ord(ch), frame):
                break


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default=models.CLASSIFICATION_MODEL,
                        help='File path of .tflite file. Default is vision.CLASSIFICATION_MODEL')
    parser.add_argument('-l', '--labels', default=None,
                        help='File path of labels file. If not specified, ' \
                        'we get the labels from the model metadata.')
    parser.add_argument('-i', '--input',
                        help='Image to be classified. If not given, use spacebar to capture and classify an image.')
    args = parser.parse_args()

    classifier = vision.Classifier(args.model)
    if args.labels is not None:
        labels = read_label_file(args.labels)
    else:
        labels = read_labels_from_metadata(args.model)

    if args.input:
        frame = imread(args.input)
        classify_image(classifier, labels, frame)
    else:
        classify_live(classifier, labels)


if __name__ == '__main__':
    main()
