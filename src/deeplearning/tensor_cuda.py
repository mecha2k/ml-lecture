# import torch
# import tensorflow
# import sys
# import platform
# import warnings
#
#
# warnings.filterwarnings("ignore", category=UserWarning)
# print(torch.__version__)
# print(tensorflow.__version__)

# tf = tensorflow.compat.v1
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(tf.__version__)
#
#
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if gpus:
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], "GPU")
#     except RuntimeError as e:
#         print(e)

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # oneDNN 최적화 비활성화

import tensorflow as tf

print(tf.__version__)
# import tensorflow as tf


#
# print(tf.version.VERSION)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
    except RuntimeError as e:
        print(e)


# CUDA 지원 여부 확인

print("CUDA 지원 여부:", tf.test.is_built_with_cuda())

# 사용 가능한 GPU 목록 확인
print("사용 가능한 GPU:", tf.config.list_physical_devices("GPU"))

# GPU 사용 가능 여부 확인
print("GPU 사용 가능 여부:", tf.test.is_gpu_available())

# TensorFlow 빌드 정보 확인
print("TensorFlow 빌드 정보:", tf.sysconfig.get_build_info())
