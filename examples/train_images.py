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
Trains a new image classification model using local images.

This script is designed for a three-step workflow with two other scripts:

1. Capture some images to train your model:

    python3 collect_images.py -l my-labels.txt

2. Train the model with those images, using this script (by default, this script
looks for images in the "captures" directory, which is where the
collect_images.py script saves them by default):

    python3 train_images.py -l my-labels.txt

3. And then run the model (my-model.tflite is the default name given to the
retrained model but you can change it by passing the `--out_model` flag to
the train_models.py script):

    python3 classify_video.py -m my-model.tflite

NOTE: The retrained model is specifically built to run with acceleration on the
Coral Edge TPU, so the model cannot run if your system does not have the
Coral USB Accelerator or another Edge TPU attached.

For information about the script options, run:

    python3 train_images.py --help

For more instructions, see g.co/aiy/maker
"""

import argparse
import os

from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.learn.imprinting.engine import ImprintingEngine
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
import models


def read_image(path, shape):
    with Image.open(path) as img:
        return img.convert('RGB').resize(shape, Image.NEAREST)


def train(capture_dir, labels, model, out_model):
    engine = ImprintingEngine(model, keep_classes=False)
    extractor = make_interpreter(engine.serialize_extractor_model(),
                                 device=':0')
    extractor.allocate_tensors()

    for class_id in sorted(labels):
        class_name = labels[class_id]
        print('\nClass: %s (id=%d)' % (class_name, class_id))
        class_capture_dir = os.path.join(capture_dir, class_name)
        for img in os.listdir(class_capture_dir):
            imgpath = os.path.join(class_capture_dir, img)
            common.set_input(extractor,
                             read_image(imgpath, common.input_size(extractor)))
            extractor.invoke()
            embedding = classify.get_scores(extractor)
            print('  %s => %s' % (imgpath, embedding))
            engine.train(embedding, class_id)

    with open(out_model, 'wb') as f:
        f.write(engine.serialize_model())
    print('\nTrained model was saved to %s' % out_model)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--labels', '-l', type=str, required=True,
                        help='Labels file from your training dataset')
    parser.add_argument('--capture_dir', '-d', type=str,
                        default='captures',
                        help='Captures directory with your training images')
    parser.add_argument('--model', '-m', type=str,
                        default=models.CLASSIFICATION_IMPRINTING_MODEL,
                        help='Base model that will be retrained')
    parser.add_argument('--out_model', '-om', type=str,
                        default='my-model.tflite',
                        help='Output filename for the retrained model')
    args = parser.parse_args()

    labels = read_label_file(args.labels)
    train(args.capture_dir, labels, args.model, args.out_model)


if __name__ == '__main__':
    main()
