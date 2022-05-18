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
Captures images from the camera and saves them into labeled folders.
This is designed to create an image dataset you can use to train a model.

To capture individual images:

1. Create a text file with a different class label on each line. For example:

    Background
    Apple
    Banana

It's important that the first item be "Background" to indicate when there are
no objects in view. And beware that for each labels file, this script supports
a total of 10 labels (limited by the number of digit keys on the keyboard).

2. Run this script and pass your labels file:

    python3 collect_images.py -l my-labels.txt

3. Capture several pictures of the "Background" by pressing 0 (the zero key)
at least 10 times. More images is always better for training, but if you plan
on using these images with the train_images.py script, then 10 images is usually
enough for a basic image classifier.

4. Place the first object ("Apple") into the camera view and press 1 (the one
key). Adjust the position of the object, and press 1 again. Repeat many times.

5. Repeat step 4 for object number 2 (press 2 while showing "Banana") and so on.

The images are saved in a local "captures" directory, using subdirectory
names that correspond to names in your labels file. If you do not provide a
labels file, the subdirectories are simply named as the numbers you pressed.

That's it. Now you have a labeled image dataset.

Alternatively, you can perform continuous image capture so you don't need to
repeatedly press the same number over and over. Just add the `--continuous`
flag with the number of images you want to capture, and this code will
rapid-fire capture that many photos. For example:

    python3 collect_images.py -l my-labels.txt --continuous 20

Then, when you press a given number, it will capture 20 photos for that label
(it does so after a short delay so you can get in position, which is necessary
if you're capturing photos for pose classification).

For information about all the script options, run:

    python3 collect_images.py --help

For more instructions, see g.co/aiy/maker
"""

import argparse
import contextlib
import queue
import os.path
import select
import sys
import termios
import threading
import tty
from datetime import datetime
from time import time
from pycoral.utils.dataset import read_label_file
from aiymakerkit import vision


@contextlib.contextmanager
def nonblocking(f):
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


@contextlib.contextmanager
def worker(process):
    requests = queue.Queue()

    def run():
        while True:
            request = requests.get()
            if request is None:
                break
            process(request)
            requests.task_done()

    def submit(request):
        requests.put(request)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        yield submit
    finally:
        requests.put(None)
        thread.join()


def save_frame(request):
    filename, frame = request
    vision.save_frame(filename, frame)
    print('Saved: %s' % filename)


def print_help(labels):
    print("Press buttons '0' .. '9' to save images from the camera.")
    if labels:
        for key in sorted(labels):
            print(key, '-', labels[key])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--labels', '-l', type=str, default=None,
                        help='Labels file')
    parser.add_argument('--continuous', '-c', type=int, default=0,
                        help='Continuously capture the number of specified images')
    parser.add_argument('--capture_dir', '-d', type=str, default='captures',
                        help='Directory for image captures')
    parser.add_argument('--capture_device_index', '-i', type=int, default=0,
                        help='Hardware capture device index')
    args = parser.parse_args()

    labels = {}
    if args.labels:
        labels = read_label_file(args.labels)
    print_help(labels)

    with nonblocking(sys.stdin) as get_char, worker(save_frame) as submit:
        def generate_filename(label_id):
            class_dir = labels.get(label_id, str(label_id))
            timestamp = datetime.now()
            filename = "PI_CAM_" + timestamp.strftime(
                "%Y%m%d_%H%M%S%f") + '.png'
            return os.path.join(args.capture_dir, class_dir, filename)

        # Handle key events from GUI window.
        def handle_key(key, frame):
            if key == ord('q') or key == ord('Q'):
                return False  # Stop processing frames.
            if key == ord('h') or key == ord('H'):
                print_help(labels)
                return True
            if args.continuous:
                return True
            if ord('0') <= key <= ord('9'):
                label_id = key - ord('0')
                filename = generate_filename(label_id)
                submit((filename, frame.copy()))
            return True  # Keep processing frames.

        START_DELAY_SECS = 3
        SNAP_DELAY_SECS = 1 / 3
        snap_time = int()
        snap_count = int()
        continuous_active = False

        for frame, key in vision.get_frames(handle_key=handle_key,
                                            capture_device_index=args.capture_device_index,
                                            return_key=True):
            # Handle continous capture mode
            if args.continuous:
                if key is not None and (ord('0') <= key <= ord('9')):
                    label_id = key - ord('0')
                    continuous_active = True
                    start_time = time()
                if continuous_active:
                    countdown = START_DELAY_SECS - int(time() - start_time)
                    # Snap the specified number of pics
                    if snap_count < args.continuous:
                        if countdown > 0:
                            vision.draw_label(frame,
                                              'GET READY IN: ' + str(countdown))
                            print('Get ready in: ', countdown, end='\r')
                        else:
                            # Wait a little between frames
                            if time() - snap_time > SNAP_DELAY_SECS:
                                filename = generate_filename(label_id)
                                submit((filename, frame.copy()))
                                snap_time = time()
                                snap_count += 1
                    elif time() - snap_time > 1:  # Artificial delay to let the last save finish
                        label_name = str(label_id)
                        label_name += ' (' + labels[
                            label_id] + ')' if labels else ''
                        print('Captured', snap_count,
                              'photos for label ' + label_name)
                        snap_count = 0
                        continuous_active = False
            # Handle key events from console.
            ch = get_char()
            if ch is not None and not handle_key(ord(ch), frame):
                break


if __name__ == '__main__':
    main()
