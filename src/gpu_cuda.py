import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())


import cv2

print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())

import sys

print(sys.path)


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    except RuntimeError as e:
        print(e)
