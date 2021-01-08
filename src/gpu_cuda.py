import tensorflow
from tensorflow.python.client import device_lib

tf = tensorflow.compat.v1
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())
print(tf.__version__)


import cv2

print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())

import sys

print(sys.path)
