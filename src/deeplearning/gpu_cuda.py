import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tensorflow
import cv2
import sys
import platform
import warnings
import os

from tensorflow.python.client import device_lib
from dotenv import load_dotenv


load_dotenv(verbose=True)
warnings.filterwarnings("ignore", category=UserWarning)
print(os.getenv("TELEGRAM"))

tf = tensorflow.compat.v1
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(tf.__version__)
print(device_lib.list_local_devices())


print("opencv version : ", cv2.__version__)
print("opencv cuda count : ", cv2.cuda.getCudaEnabledDeviceCount())
try:
    print(cv2.cuda.printCudaDeviceInfo(0))
except Exception as e:
    print(e)
print(sys.path)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    except RuntimeError as e:
        print(e)

print(torch.__version__)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"{device} is available in torch")

print(sys.version)
print(platform.platform())

sample = torch.randn(256, 256).to(device)
print(sample.shape)
