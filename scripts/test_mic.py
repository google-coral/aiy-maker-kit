# Copyright 2022 Google LLC
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

import pyaudio
import wave
import time
import os
import subprocess

FORMAT = pyaudio.paInt32
CHUNK_SIZE = 1024
CHANNELS = 1
RATE = 44100
RECORD_SECS = 3
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
FILENAME = os.path.join(SCRIPT_DIR, 'test.wav')


def find_input_devices():
    devices = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
    cards = []
    for line in devices.stdout.split('\n'):
        if 'card' in line:
            cards.append(line)
    if cards:
        print('Detected microphones:')
        for card in cards:
            print('--->', card)
        return True
    else:
        print('No microphones detected.')
        return False


def main():
    audio = pyaudio.PyAudio()

    print('\n------- Check microphone -------\n')
    if find_input_devices():
        # Record a few seconds of audio
        stream = audio.open(rate=RATE, channels=CHANNELS, format=FORMAT,
                            input=True, frames_per_buffer=CHUNK_SIZE)
        frames = []
        start = time.monotonic()
        print('\nNow recording.')
        for i in range(int(RATE / CHUNK_SIZE * RECORD_SECS)):
            data = stream.read(CHUNK_SIZE)
            frames.append(data)
            elapsed = int(time.monotonic() - start)
            print('Recording for...', RECORD_SECS - elapsed, end='\r')
        print('\nDone.')
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save to a WAV file
        with wave.open(FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        print('Recording saved at', FILENAME)


if __name__ == '__main__':
    main()
