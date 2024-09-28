import tensorflow as tf
import os
import sys
import warnings


# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
    except RuntimeError as e:
        print(e)

print("tensorflow version : ", tf.__version__)
print("tensorflow build : ", tf.sysconfig.get_build_info())
print("cuda available : ", tf.test.is_built_with_cuda())
print("gpu available : ", tf.config.list_physical_devices("GPU"))
print("gpus : ", tf.config.list_physical_devices("GPU"))
print("CUDA version : ", tf.sysconfig.get_build_info()["cuda_version"], "\n\n")

print("os path : ")
for path in os.environ["PATH"].split(os.pathsep):
    print(path)
