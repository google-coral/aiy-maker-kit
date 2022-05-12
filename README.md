# AIY Maker Kit Python API and examples

The aiymakerkit API greatly simplifies the amount of code needed to
perform common operations with TensorFlow Lite models, such as performing image
classification, object detection, pose estimation, and speech recognition
(usually in combination with the Coral Edge TPU).

This repo also includes
scripts to collect training images and perform transfer learning with an image
classification model, directly on your device (such as a Raspberry Pi).

This project was designed specifically for the
[AIY Maker Kit](https://aiyprojects.withgoogle.com/maker/), which uses a
Raspberry Pi with a Coral USB Accelerator, camera, and microphone.

## Learn more

To get started, see the [AIY Maker Kit documentation](https://aiyprojects.withgoogle.com/maker/).
It includes complete setup instructions with a Raspberry Pi, project tutorials,
and the **[aiymakerkit API reference](https://aiyprojects.withgoogle.com/maker/#reference)**.


## Install on Raspberry Pi OS

If you're on a Raspberry Pi, we recommend you flash our custom Raspberry Pi OS
system image before installing this library, as documented at
https://aiyprojects.withgoogle.com/maker/. That way, you're sure to have
all the required software installed and there should be no trouble.

But if you want to do things differently and can tolerate some extra steps and risk troubleshooting,
you can build our system image yourself and/or install the required libraries
on an existing RPI OS system as documented at
https://github.com/google-coral/aiy-maker-kit-tools (but we
do not recommend it).


## Install manually

For other situations where you want to install only the `aiymakerkit` library,
**you must manually install the `libedgetpu` and `pycoral` libraries first**.
Assuming that you are also using the Coral USB
Accelerator, you can get these libraries by following the [Coral USB Accelerator
setup guide at coral.ai](https://coral.ai/docs/accelerator/get-started/).

Then you can clone this repo and install the library as follows:

```
git clone https://github.com/google-coral/aiy-maker-kit.git

cd aiymakerkit

python3 -m pip install .
```
