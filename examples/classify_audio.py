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
Performs live speech recognition with a microphone.

To run the script, pass a speech model as the only argument:

    python3 classify_audio.py soundclassifier_with_metadata.tflite

Specifically, the model must be based on BrowserFFT, which you can train
yourself at https://teachablemachine.withgoogle.com/train/audio

For more instructions, see g.co/aiy/maker
"""

from aiymakerkit import audio
import argparse


def handle_results(label, score):
    print('CALLBACK: ', label, '=>', score)
    return True  # keep listening

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    args = parser.parse_args()

    audio.classify_audio(model=args.model_file, callback=handle_results)

if __name__ == '__main__':
    main()
