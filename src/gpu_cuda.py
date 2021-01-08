import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())
print(tf.__version__)


import cv2

print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())

import sys

print(sys.path)
