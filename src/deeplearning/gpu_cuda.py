import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import cv2
import sys
import platform
import warnings
import os
from dotenv import load_dotenv


load_dotenv(verbose=True)
warnings.filterwarnings("ignore", category=UserWarning)
print(os.getenv("TELEGRAM"))

print("opencv version : ", cv2.__version__)
print("opencv cuda count : ", cv2.cuda.getCudaEnabledDeviceCount())
try:
    print(cv2.cuda.printCudaDeviceInfo(0))
except Exception as e:
    print(e)
print(sys.path)

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


print(torch.__version__)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch")

print(sys.version)
print(platform.platform())

sample = torch.randn(256, 256).to(device)
print(sample.shape)
