import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(device_lib.list_local_devices())
