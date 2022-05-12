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

import os
import subprocess
import sys
import time
import traceback
from aiymakerkit import vision
from examples import models

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def usb_accelerator_connected():
    if subprocess.run(['lsusb', '-d', '18d1:9302'],
                      capture_output=True).returncode == 0:
        return True
    if subprocess.run(['lsusb', '-d', '1a6e:089a'],
                      capture_output=True).returncode == 0:
        return True
    return False


def main():
    print('--- Checking display ---')
    if not 'DISPLAY' in os.environ:
        print('No display detected. You must have a display connected or ' \
              'enabled so you can see the desktop. If you have a display ' \
              'enabled but you are logged in via SSH, you need to specify ' \
              'the display with `export DISPLAY=:0`, then retry this script.')
        return 1
    print('Found a display.\n')

    print('--- Checking required files ---')
    if not os.path.isfile(models.CLASSIFICATION_MODEL):
        print('Downloading files...')
        subprocess.call(['bash', os.path.join(
            SCRIPT_DIR, 'examples', 'download_models.sh')])
        subprocess.call(['bash', os.path.join(
            SCRIPT_DIR, 'projects', 'download_models.sh')])
    print('Found the required files.\n')

    print('--- Testing camera ---')
    TIME_LIMIT = 4
    start = time.monotonic()
    for frame in vision.get_frames():
        elapsed = int(time.monotonic() - start)
        print('Closing video in...', TIME_LIMIT - elapsed, end='\r')
        if (elapsed >= TIME_LIMIT):
            print('\nCamera is working.\n')
            break

    print('--- Testing USB Accelerator ---')
    if not usb_accelerator_connected():
        print('Coral USB Accelerator NOT found!')
        print('Make sure it\'s connected to the Raspberry Pi.')
        return 1

    print('Loading a model...')
    try:
        classifier = vision.Classifier(models.CLASSIFICATION_MODEL)
        classes = classifier.get_classes(frame)
        if classes:
            print('USB Accelerator is working.')
    except ValueError:
        traceback.print_exc()
        print('Something went wrong with the USB Accelerator.')
        print('Try unplugging the USB Accelerator, then plug it back in and ' \
              'run the script again.')
        return 1

    print('\nEverything look good!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
