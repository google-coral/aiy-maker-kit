
Image classification
--------------------

.. autoclass:: aiymakerkit.vision.Classifier
    :members:


Object detection
----------------

.. autoclass:: aiymakerkit.vision.Detector
    :members:


Pose detection
--------------

.. autoclass:: aiymakerkit.vision.PoseDetector
    :members:
    :undoc-members:

.. autofunction:: aiymakerkit.vision.get_keypoint_types

.. autoclass:: aiymakerkit.vision.KeypointType
    :members:
    :undoc-members:


Pose classification
-------------------

.. autoclass:: aiymakerkit.vision.PoseClassifier
    :members:
    :undoc-members:


Camera & drawing
----------------

.. autofunction:: aiymakerkit.vision.get_frames

.. autofunction:: aiymakerkit.vision.save_frame

.. automodule:: aiymakerkit.vision
    :members: draw_classes, draw_objects, draw_pose, draw_label, draw_rect, draw_circle
    :member-order: bysource


.. Until we can put objects.inv on coral.ai
.. |Class| replace:: ``Class``
.. _Class: https://coral.ai/docs/reference/py/pycoral.adapters/#pycoral.adapters.classify.Class
.. |Object| replace:: ``Object``
.. _Object: https://coral.ai/docs/reference/py/pycoral.adapters/#pycoral.adapters.detect.Object
.. |BBox| replace:: ``BBox``
.. _BBox: https://coral.ai/docs/reference/py/pycoral.adapters/#pycoral.adapters.detect.BBox