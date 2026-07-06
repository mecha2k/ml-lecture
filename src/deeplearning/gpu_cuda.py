import tensorflow as tf
import torch
import cv2
import sys
import platform
import warnings
import os
from dotenv import load_dotenv


load_dotenv(verbose=True)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
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
device = torch.device("cuda:0")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
    )
elif device.type == "mps":
    print("Apple Silicon MPS 사용 중")
    print(f"MPS 사용 가능 여부: {torch.backends.mps.is_available()}")
    print(f"MPS 빌드 포함 여부: {torch.backends.mps.is_built()}")
print("CUDA version : ", torch.version.cuda)

print(sys.version)
print(platform.platform())

sample = torch.randn(256, 256).to(device)
print(sample.shape)
