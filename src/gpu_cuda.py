from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow
from tensorflow.python.client import device_lib
from tensorflow import keras

import cv2
import sys


tf = tensorflow.compat.v1
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())

print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())

print(sys.path)


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()