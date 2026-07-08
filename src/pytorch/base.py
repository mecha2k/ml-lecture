# -*- coding: utf-8 -*-
"""
공통 초기화 모듈 — 모든 노트북/스크립트에서 import하여 사용
"""

# ── 표준 라이브러리 ──────────────────────────────────────────────────────────
import os
import random
import warnings
import requests
from PIL import Image
from pathlib import Path

# ── 데이터 처리 ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── 시각화 ───────────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ── 머신러닝 ─────────────────────────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.datasets import load_digits

# ── 분류 ─────────────────────────────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
)
from sklearn.naive_bayes import GaussianNB

# ── 부스팅 (외부 라이브러리) ───────────────────────────────────────────────────
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# ── 차원 축소 / 군집화 ────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# ── 딥러닝 (PyTorch) ──────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device ───────────────────────────────────────────────────────────────────
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

# ── 한글 폰트 (Windows) ───────────────────────────────────────────────────────
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지

# ── 기타 설정 ─────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)
