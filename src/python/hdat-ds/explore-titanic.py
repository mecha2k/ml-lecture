import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


datapath = Path("./data") / "train.csv"
train = pd.read_csv(datapath, index_col="PassengerId")
train.info()
print(train.shape)
print(train.head())

# **1. 타이타닉의 train 데이터에서 1) 전체 생존률과 2) 생존자의 총 인원수, 사망자의 총 인원수를 출력해주세요.**
# 1번(생존률)의 경우 약 38.4%가 나와야 하며, 2번(인원수)의 경우 생존자의 총 인원수는 342명, 사망자의 총 인원수는 549명이 나와야 합니다.
survived_rate = train["Survived"].mean() * 100
print(f"생존률 = {survived_rate:.1f}%")
print(f"생존자의 총 인원수(1) = {train['Survived'].value_counts()[1]}")
print(f"사망자의 총 인원수(0) = {train['Survived'].value_counts()[0]}")

# **2. Survived 컬럼에 들어가 있는 값을 쉬운 표현으로 바꿔주세요.**
# Survived 컬럼에는 0(사망)이라는 값과 1(생존)이라는 값이 있습니다. 이 표현은 직관적이지 않기 때문에, 데이터 분석을 원활하게 하기 위해서는 사람이 읽기 쉬운 표현을 쓰는 것이 좋습니다.
train.loc[train["Survived"] == 0, "Survived(humanized)"] = "Died"
train.loc[train["Survived"] == 1, "Survived(humanized)"] = "Survived"
# Survived 컬럼이 0인 값을 Perish로, 1인 값을 Survived로 대체(replace)합니다.
train["Survived(humanized)"] = train["Survived"].replace(0, "Died").replace(1, "Survived")
print(train[["Survived", "Survived(humanized)"]].head())

# 또한 이번에는 Survived 컬럼이 아닌 아닌 새롭게 만든 Survived(humanized) 컬럼으로 생존자의 총 인원수와 사망자의 총 인원수를 출력해 주세요.
# 앞서 사용한 ```value_counts```를 그대로 사용하면 될 것 같습니다.
# 생존자의 총 인원수(Survived)은 342명, 사망자의 총 인원수(Perish)는 549명이 나와야 합니다.
print(train["Survived(humanized)"].value_counts()["Survived"])


# **3. Pclass 컬럼에 들어가 있는 값을 읽기 쉬운 표현으로 바꿔주세요.**
# Pclass도 마찬가지로 1, 2, 3이라는 표현은 직관적이지 않기 때문에, 사람이 이해하기 쉬운 표현으로 바꿔주고 싶습니다.
# pandas의 pivot_table을 활용하여 Pclass별 생존률을 출력합니다.
# 여기서 Pclass값이 1, 2, 3이 나오는데, Pclass 컬럼에 대한 사전 설명을 듣지 않으면 이해하기 어렵습니다.
# 그러므로 Pclass값을 조금 더 직관적으로 바꿔준다면 pivot_table로 분석하기 편할 것입니다.
pclass_pivot = pd.pivot_table(data=train, index="Pclass", values="Survived", aggfunc="mean")
print(pclass_pivot)

# 이번에는 **Pclass(humanized)**라는 새로운 컬럼을 만들어주세요.
# 이 컬럼에는 1, 2, 3이 아닌 First Class, Business, Economy 라는 값이 들어가 있다면 좋겠습니다.
# 또한 위 내용을 바탕으로 **Pclass(humanized)**별 생존자와 사망자의 차이를 시각화해주세요. 최종적으로는 다음의 결과가 나와야 합니다.
train.loc[train["Pclass"] == 1, "Pclass(humanized)"] = "First Class"
train.loc[train["Pclass"] == 2, "Pclass(humanized)"] = "Business"
train.loc[train["Pclass"] == 3, "Pclass(humanized)"] = "Economy"
# 마찬가지로 아래의 코드도 동일한 역할을 수행합니다.
# Survived 컬럼이 0인 값을 Perish로, 1인 값을 Survived로 대체(replace)합니다.
train["Pclass(humanized)"] = (
    train["Pclass"].replace(1, "First Class").replace(2, "Business").replace(3, "Economy")
)
print(train[["Pclass", "Pclass(humanized)"]].head())

# pandas의 pivot_table을 활용하여 Pclass별 생존률을 출력합니다.
# 하지만 이번에는 Pclass 컬럼이 아닌 Pclass(humanized) 컬럼을 사용합니다.
# 이전에 비해서 훨씬 더 직관적으로 생존률을 확인할 수 있습니다.
print(pd.pivot_table(data=train, index="Pclass(humanized)", values="Survived", aggfunc="mean"))

# 그리고 이를 활용해 seaborn의 countplot으로 시각화 할 수 있습니다.
# 시본(seaborn)의 countplot으로 Pclass별 생존자와 사망자의 차이를 시각화합니다.
sns.countplot(data=train, x="Pclass(humanized)", hue="Survived(humanized)")
plt.savefig("explore-titanic-pclass.png", bbox_inches="tight")

# **4. Embarked 컬럼에 들어가 있는 값을 읽기 쉬운 표현으로 바꿔주세요.**
# Embarked 컬럼도 마찬가지로 C, S, Q라는 표현은 직관적이지 않습니다.
# C는 Cherbourg 라는 표현으로, S는 Southampton 이라는 표현으로, 그리고 Q는 Queenstown 이라는 표현으로 바꾸겠습니다.
# pandas의 pivot_table을 활용하여 Embarked 별 생존률을 출력합니다.
# 여기서도 Embarked 컬럼이 C, S, Q라는 다소 직관적이지 않은 값이 나옵니다.
# 그러므로 Embarked 컬럼의 값도 Pclass 처럼 직관적으로 바꿔주고 싶습니다.
print(pd.pivot_table(data=train, index="Embarked", values="Survived", aggfunc="mean"))


# 먼저 Embarked 컬럼이 C인 승객을 색인합니다. 이후 Embarked(humanized)라는 이름의
train.loc[train["Embarked"] == "C", "Embarked(humanized)"] = "Cherbourg"
train.loc[train["Embarked"] == "S", "Embarked(humanized)"] = "Southampton"
train.loc[train["Embarked"] == "Q", "Embarked(humanized)"] = "Queenstown"
print(train[["Embarked", "Embarked(humanized)"]].head())

print(pd.pivot_table(data=train, index="Embarked(humanized)", values="Survived", aggfunc="mean"))
sns.countplot(data=train, x="Embarked(humanized)", hue="Survived(humanized)")
plt.savefig("explore-titanic-embarked.png", bbox_inches="tight")


# # **5. 나이(Age) 컬럼에서 다음의 정보를 출력해주세요.**
# #
# #   * 평균(mean)
# #   * 가장 나이가 많은 사람. (max)
# #   * 가장 나이가 적은 사람. (min)
# #
# # 가령 평균은 약 29.7세, 가장 어린 사람은 0.42세(약 생후 4개월), 가장 나이가 많은 사람은 80세가 나와야 합니다.
# #%%
# # 나이(Age) 컬럼에서 mean 함수를 통해 평균 나이를 구합니다.
# # 평균 나이가 약 29.7세라는 것을 알 수 있습니다.
# train["Age"].mean()
# #%%
# # 나이(Age) 컬럼에서 min 함수를 통해 나이의 최소치를 구합니다.
# # 타이타닉호에 탑승한 가장 어린 승객은 약 0.42세(생후 4개월 정도)라는 것을 알 수 있습니다.
# train["Age"].min()
# #%%
# # 나이(Age) 컬럼에서 max 함수를 통해 나이의 최대치 구합니다.
# # 타이타닉호에 탑승한 가장 나이가 많은 승객은 80세라는 것을 알 수 있습니다.
# train["Age"].max()
# #%% md
# # 또는 판다스의 [describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)를 활용하면 한 줄의 코드로 평균, 분산, 최소치, 최대치를 볼 수 있습니다.
# #%%
# # 나이(Age) 컬럼에 대해 describe 함수를 사용합니다.
# # 이 함수는 특정 컬럼의 평균, 분산, 최대치, 최소치와 같은 기초적인 통계치를 보여줍니다.
# train["Age"].describe()
# #%% md
# # **6. 객실 등급별 나이(Age) 컬럼의 평균을 보여주세요.**
# #
# # 이번에는 전체 평균이 아닌 객실 등급(Pclass)별 평균을 보고 싶습니다.
# #
# # 가령 전체 승객의 평균 나이는 약 29.7세이지만, 1등급 승객의 평균 나이는 약 38.2세가 나와야 합니다. 비슷한 방식으로 2등급과 3등급 승객의 평균 나이를 알 수 있다면 좋겠습니다.
# #%%
# # Pclass가 1등급인 승객만 색인해서 가져온 뒤, 이를 Pclass1이라는 변수에 할당합니다.
# pclass1 = train[train["Pclass"] == 1]
#
# # 1등급 승객의 평균 나이를 구합니다.
# pclass1["Age"].mean()
# #%%
# # Pclass가 2등급인 승객만 색인해서 가져온 뒤, 이를 Pclass2이라는 변수에 할당합니다.
# pclass2 = train[train["Pclass"] == 2]
#
# # 2등급 승객의 평균 나이를 구합니다.
# pclass2["Age"].mean()
# #%%
# # Pclass가 3등급인 승객만 색인해서 가져온 뒤, 이를 Pclass3이라는 변수에 할당합니다.
# pclass3 = train[train["Pclass"] == 3]
#
# # 3등급 승객의 평균 나이를 구합니다.
# pclass3["Age"].mean()
# #%% md
# # 또는 판다스의 [groupby](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html)를 활용하면 한 줄의 코드로 1, 2, 3등급 승객의 평균 나이를 가져올 수 있습니다.
# #%%
# # 타이타닉 데이터를 Pclass 기준으로 그룹화합니다.
# # 이렇게 하면 Pclass의 세 종류(1등급, 2등급, 3등급)마다 따로따로 연산을 할 수 있습니다.
# # 이후 나이(Age) 컬럼의 평균(mean)을 구하면 1, 2, 3등급마다의 평균 나이가 나옵니다.
# train.groupby("Pclass")["Age"].mean()
# #%% md
# # 비슷한 기능을 pivot_table로도 할 수 있습니다.
# #%%
# # pandas의 pivot_table을 활용하여 Pclass별 평균 나이(Age)를 출력합니다.
# # 이전에 비해서 훨씬 더 깔끔하게 출력할 수 있습니다.
# pd.pivot_table(data=train, index="Pclass", values="Age")
# #%% md
# # **7. 나이를 일정 구역으로 나눠서, 구역마다의 생존률을 보여주세요.**
# #
# # 이번에는 나이(Age)별 생존률을 확인하고 싶습니다. 다만 나이 컬럼은 숫자이기 때문에, 그대로 쓰지 않고 일정 구역으로 나눈 뒤 생존률의 통계를 내는 것이 보기 편할 것입니다. 그러므로 나이 컬럼을 다음의 세 구역으로 나눕니다.
# #
# #   1. 나이가 15세 미만인 승객.
# #   2. 나이가 15세 이상이고 30세 미만인 승객.
# #   3. 나이가 30세 이상인 승객.
# #
# # 최종적으로는 다음의 결과가 나와야 합니다.


# # 또한, 위 조건에서 1번, 2번, 3번 구역에 해당하는 승객의 평균 생존률을 구하고 싶습니다.
# #
# # 가령 1번 구역(나이가 15세 미만)에 해당하는 승객의 평균 생존률은 약 57.7%가 나와야 합니다.
# #%% md
# # 마지막으로 이를 활용해 1) 구역별 생존자와 사망자의 차이, 2) 구역별 평균 나이를 시각화 해주세요. 최종적으로는 다음의 결과가 나와야 합니다.
# #
# # ![quiz-7-1](https://drive.google.com/uc?export=view&id=15otOk_H9yHbARzoSBSbWIzMcswpQ7olE)
# #%% md
# # ![quiz-7-2](https://drive.google.com/uc?export=view&id=1n49DVaxgf2au5eXgnzSV0wY3PoHw02gL)
# #%%
# # 나이가 15세 미만인 승객을 색인한 뒤, AgeType이라는 새로운 컬럼에 "Young"이라는 값을 넣습니다.
# train.loc[train["Age"] < 15, "AgeType"] = "Young"
#
# # 비슷하게 나이가 15세 이상 30세 미만인 승객의 AgeType에는 "Medium"이라는 값을 넣습니다.
# train.loc[(train["Age"] >= 15) & (train["Age"] < 30), "AgeType"] = "Medium"
#
# # 비슷하겍 30세 이상인 승객의 AgeType에는 "Old"이라는 값을 넣습니다.
# train.loc[train["Age"] >= 30, "AgeType"] = "Old"
#
# # train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# # 출력은 (row, column) 으로 표시됩니다.
# print(train.shape)
#
# # 나이(Age) 컬럼과 AgeType 컬럼을 출력하여 비교합니다.
# train[["Age", "AgeType"]].head(10)
# #%%
# # 타이타닉 데이터를 AgeType 기준으로 그룹화합니다.
# # 이후 생존 여부(Survived) 컬럼의 평균(mean)을 구하면 Young, Medium, Old 마다의 평균 생존률이 나옵니다.
# train.groupby("AgeType")["Survived"].mean()
# #%% md
# # 마찬가지로 countplot을 이용해 시각화 할 수 있습니다.
# #%%
# # 시본(seaborn)의 countplot으로 AgeType별 생존자와 사망자의 차이를 시각화합니다.
# sns.countplot(data=train, x="AgeType", hue="Survived(humanized)")
# #%%
# # 시본(seaborn)의 barplot으로 AgeType별 평균 나이(Age)를 시각화합니다.
# sns.barplot(data=train, x="AgeType", y="Age")
# #%% md
# # **8. 나이가 비어있는 승객과 비어있지 않은 승객의 생존률 차이를 보여주세요.**
# #
# # 이번에는 다른 방식으로 생존률의 차이를 보겠습니다. 타이타닉 데이터의 나이(Age) 컬럼을 자세히 보면 나이가 비어있는 데이터가 있습니다. 판다스에서는 이를 NaN(Not a Number의 약자)으로 표현합니다.
# #
# # 타이타닉 데이터에서 나이 컬럼이 비어있는 승객과 비어있지 않은 승객의 생존률을 각각 찾아서 출력해주세요. 또한 이를 시각화로 비교해주세요. 최종적으로 다음의 결과가 나와야합니다.
# #%% md
# # ![quiz-8](https://drive.google.com/uc?export=view&id=18d5DhEuPU4N99avIwrl2FpPGdyQUI7Ty)
# #%%
# # isnull 함수를 활용해 나이 컬럼이 비어있는 승객만 색인합니다.
# # 이 데이터에서 AgeBlank라는 새로운 컬럼을 만든 뒤, 여기에 "Blank"라는 값을 넣습니다.
# train.loc[train["Age"].isnull(), "AgeBlank"] = "Blank"
#
# # 비슷한 방식으로 notnull 함수를 활용하여 AgeBlank 컬럼에 "Not Blank"라는 값을 넣습니다.
# train.loc[train["Age"].notnull(), "AgeBlank"] = "Not Blank"
#
# # train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# # 출력은 (row, column) 으로 표시됩니다.
# print(train.shape)
#
# # 나이(Age) 컬럼과 AgeBlank 컬럼을 출력하여 비교합니다.
# train[["Age", "AgeBlank"]].head(10)
# #%%
# # 타이타닉 데이터를 AgeBlank 기준으로 그룹화합니다.
# # 이후 생존 여부(Survived) 컬럼의 평균(mean)을 구하면 Blank, Not Blank 마다의 평균 생존률이 나옵니다.
# train.groupby("AgeBlank")["Survived"].mean()
# #%%
# # 비슷한 기능을 pivot table로도 할 수 있습니다.
# pd.pivot_table(data=train, index="AgeBlank", values="Survived")
# #%% md
# # 마찬가지로 이를 countplot으로 시각화할 수 있습니다.
# #%%
# # 시본(seaborn)의 countplot으로 AgeBlank별 생존자와 사망자의 차이를 시각화합니다.
# sns.countplot(data=train, x="AgeBlank", hue="Survived(humanized)")
# #%% md
# # **9. Pclass별 나이(Age)의 평균을 구한 뒤 빈 값에 채워주세요.**
# #
# # 이번에는 나이(Age) 컬럼의 빈 값을 채우고 싶습니다. 일반적으로 가장 많이 하는 방식은 나이의 평균(mean)값을 구한 뒤 이를 빈 값에 채워넣는 것입니다. 하지만 이번에는 다른 방식으로 빈 값을 채우고 싶은데, 바로 객실 등급(Pclass)에 따라 다르게 나이의 빈 값을 채워주고 싶습니다. 가령
# #
# #   1. 객실 등급(Pclass)이 1등급인 승객의 평균 나이를 구해서, 해당 승객 중 나이(Age)컬럼값이 비어있는 승객을 찾아 빈 나이 값을 채워줍니다.
# #   2. 객실 등급(Pclass)이 2등급인 승객의 평균 나이를 구해서, 해당 승객 중 나이(Age)컬럼값이 비어있는 승객을 찾아 빈 나이 값을 채워줍니다.
# #   3. 객실 등급(Pclass)이 3등급인 승객의 평균 나이를 구해서, 해당 승객 중 나이(Age)컬럼값이 비어있는 승객을 찾아 빈 나이 값을 채워줍니다.
# #
# # 위와 같은 방식을 사용하면, 단순히 전체 평균을 사용하는 것 보다 조금 더 원래 값에 근접하게 평균을 채워줄 수 있을 것 같습니다. 최종적으로는 다음의 결과가 나와야 합니다.


# # 타이타닉 데이터를 Pclass 기준으로 그룹화한 뒤, 나이(Age) 컬럼의 평균을 구합니다.
# # 이 결과를 mean_age_by_pclass 라는 변수에 할당합니다.
# mean_age_by_pclass = train.groupby("Pclass")["Age"].mean()
# mean_age_by_pclass
# #%%
# # Age 컬럼에 바로 값을 채워주는 것도 좋지만, 가능한 원본은 유지한 채 사본에다가 작업하는 것을 추천합니다.
# # 그러므로 Age(fill) 이라는 새로운 컬럼을 만든 뒤, 이 컬럼의 빈 값을 채워줄 것입니다.
# train["Age(fill)"] = train["Age"]
#
# # train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# # 출력은 (row, column) 으로 표시됩니다.
# print(train.shape)
#
# # 객실 등급(Pclass), 나이(Age), 그리고 Age(fill) 컬럼을 출력하여 비교합니다.
# train[["Pclass","Age", "Age(fill)"]].head(30)
# #%%
# # 객실 등급(Pclass)이 1등급이고 나이(Age) 컬럼값이 비어있는 승객을 색인합니다.
# # 이 승객의 Age(fill)에 평균 1등급 승객의 평균 나이를 채워넣습니다.
# train.loc[(train["Pclass"] == 1) & (train["Age"].isnull()), "Age(fill)"] = mean_age_by_pclass.loc[1]
#
# # 비슷한 원리로 객실 등급(Pclass)이 2등급인 승객도 비슷한 방식으로 빈 나이값을 채워넣습니다.
# train.loc[(train["Pclass"] == 2) & (train["Age"].isnull()), "Age(fill)"] = mean_age_by_pclass.loc[2]
#
# # 객실 등급(Pclass)이 3등급인 승객도 비슷한 방식으로 빈 나이값을 채워넣습니다.
# train.loc[(train["Pclass"] == 3) & (train["Age"].isnull()), "Age(fill)"] = mean_age_by_pclass.loc[3]
#
# # train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# # 출력은 (row, column) 으로 표시됩니다.
# print(train.shape)
#
# # 객실 등급(Pclass), 나이(Age), 그리고 Age(fill) 컬럼을 출력하여 비교합니다.
# train[["Pclass","Age", "Age(fill)"]].head(30)
# #%%
# # 나이(Age) 컬럼값이 비어있는 승객만 가져온 뒤,
# # 이 승객의 객실 등급(Pclass), 나이(Age), 그리고 Age(fill) 컬럼을 출력하여 비교합니다.
# train.loc[train["Age"].isnull(), ["Pclass","Age", "Age(fill)"]].head(10)
# #%% md
# # ### SibSp, Parch 컬럼 분석
# #%% md
# # **10. 타이타닉호에 동승한 형제, 자매, 배우자(SibSp)도 없고, 부모와 자식(Parch)도 없는 사람을 구해주세요.**
# #
# # 해당 사용자를 싱글(Single)이라고 가정하겠습니다. 최종적으로는 다음의 결과가 나와야 합니다.

# # 또한 싱글(Single)인 사람과 그렇지 않은 사람간의 생존률의 차이도 알고 싶습니다. 최종적으로는 다음의 결과가 나와야 합니다.

# # 마지막으로 이를 시각화를 통해 비교해주세요. 최종적으로는 다음의 결과가 나와야합니다.
# #%% md
# # ![quiz-10](https://drive.google.com/uc?export=view&id=1WakO3v3BHr3aunjcg170AKLjxZ3Xi1eQ)
# #%%
# # SibSp가 0이고 Parch가 0이면 True, 아니면 False인 리스트를 생성합니다.
# # 이 리스트를 Single이라는 이름의 새로운 컬럼에 집어넣습니다.
# train["Single"] = (train["SibSp"] == 0) & (train["Parch"] == 0)
#
# # train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# # 출력은 (row, column) 으로 표시됩니다.
# print(train.shape)
#
# # SibSp, Parch, 그리고 Single을 출력하여 비교합니다.
# train[["SibSp", "Parch", "Single"]].head()
# #%%
# # pandas의 pivot_table을 활용하여 Single 여부에 따른 생존률을 출력합니다.
# # Single 컬럼의 값이 True일 경우의 생존률과, False일 경우의 생존률을 비교할 수 있습니다.
# pd.pivot_table(train, index="Single", values="Survived")
# #%% md
# # 마찬가지로 countplot으로 시각화할 수 있습니다.
# #%%
# # 시본(seaborn)의 countplot으로 Single별 생존자와 사망자의 차이를 시각화합니다.
# sns.countplot(data=train, x="Single", hue="Survived(humanized)")
# #%% md
# # **11. SibSp 컬럼과  Parch 컬럼을 활용하여 가족 수(FamilySize)라는 새로운 컬럼을 만들어주세요.**
# #
# # 형제, 자매, 배우자(SibSp) 컬럼과 부모 자식(Parch) 컬럼은 얼핏 달라 보이지만 실은 가족 관계를 나타내는 것이라고 볼 수 있습니다. 그러므로 두 컬럼을 하나로 합쳐서 **가족 수(FamilySize)**라는 새로운 컬럼을 만들면 승객의 가족관계를 더 편리하게 분석할 수 있을 것입니다.
# #
# # 형제, 자매, 배우자(SibSp) 컬럼과 부모 자식(Parch) 컬럼을 더해서 가족 수(FamilySize) 컬럼을 만들어주세요. 단 가족 수를 계산할때는 언제나 나 자신을 포함해서 계산하는데, 나 자신은 SibSp 컬럼에도 Parch 컬럼에도 들어가있지 않습니다. 그러므로 가족 수(FamilySize) 컬럼은 언제나 SibSp 컬럼과 Parch 컬럼을 더한 값에서 하나가 더 많아야 합니다.
# #
# # 그러므로 최종적으로 다음의 결과가 나와야 합니다.

# # 또한 가족 수(FamilySize) 컬럼을 구한 뒤, 가족 수 별 생존률의 차이도 알고 싶습니다. 가족 수(ex: 1명 ~ 11명) 마다의 생존률을 구해서 출력해주세요. 최종적으로 다음의 결과가 나와야 합니다.


# # ![quiz-11](https://drive.google.com/uc?export=view&id=1vjGvKBVWM1SsSlKz6Aji1ENo6MZV5ses)
# #%%
# # 형제, 자매, 배우자(SibSp) 컬럼과 부모 자식(Parch) 컬럼을 더해서 가족 수(FamilySize) 컬럼을 만듭니다.
# # 또한 가족 수에 나 자신을 포함하기 위해서 언제나 +1을 해줍니다.
# train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
#
# # train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# # 출력은 (row, column) 으로 표시됩니다.
# print(train.shape)
#
# # SibSp, Parch, FamilySize를 출력하여 비교합니다.
# train[["SibSp", "Parch", "FamilySize"]].head(10)
# #%%
# # pandas의 pivot_table을 활용하여 FamilySize에 따른 생존률을 출력합니다.
# # 가족 수가 1명부터 11명까지 각각의 생존률을 비교할 수 있습니다.
# pd.pivot_table(train, index="FamilySize", values="Survived")
# #%% md
# # 마찬가지로 이를 countplot으로 시각화할 수 있습니다.
# #%%
# # 시본(seaborn)의 countplot으로 FamilySize별 생존자와 사망자의 차이를 시각화합니다.
# sns.countplot(data=train, x="FamilySize", hue="Survived(humanized)")
# #%% md
# # **12. 가족 수(FamilySize) 컬럼의 구역을 나눠주세요.**
# #
# # 가족 수(FamilySize) 컬럼을 기준으로 pivot_table로 분석을 해본 결과, 경우의 수가 너무 많아서(가족 수가 1명일 때 ~ 11명일 때) 분석 결과가 너무 잘게 쪼개지는 것 같습니다.
# #
# # 그러므로 가족 수(FamilySize) 컬럼을 세 구역으로 나누고 싶습니다. 구체적으로는 다음과 같습니다.
# #
# #   * **싱글(Single)** - 동승한 가족이 아무도 없고, 나 혼자 탑승한 경우입니다.
# #   * **핵가족(Nuclear)** - 동승한 가족이 나 자신을 포함해 2명 이상 5명 미만인 경우입니다.
# #   * **대가족(Big)** - 동승한 가족이 나 자신을 포함 5명 이상인 경우입니다.
# #
# # 위의 정보를 활용하여, 가족 형태(FamilyType)라는 새로운 컬럼을 만들어 주세요. 이 컬럼에는 앞서 설명한 Single, Nuclear, 그리고 Big이 들어갑니다. 최종적으로는 다음의 결과가 나와야 합니다.


# # 또한 가족 수(FamilySize)와 마찬가지로 가족 형태(FamilyType) 별 생존률의 차이도 구해주세요. 최종적으로 다음의 결과가 나와야 합니다.


# # 마지막으로 이를 시각화를 통해 비교해주세요. 최종적으로는 다음의 결과가 나와야합니다.
# #%% md
# # ![quiz-12](https://drive.google.com/uc?export=view&id=1xYTDLeZ_6d11DNdzW12RrcCtf4ZEa2nt)
# #%%
# # 가족 수(FamilSize)가 1인 승객을 가져와서, FamilyType 컬럼에 Single 이라는 값을 넣어줍니다.
# train.loc[train["FamilySize"] == 1, "FamilyType"] = "Single"
#
# # 가족 수(FamilSize)가 2 이상 5 미만인 승객을 가져와서, FamilyType 컬럼에 Nuclear(핵가족) 이라는 값을 넣어줍니다.
# train.loc[(train["FamilySize"] > 1) & (train["FamilySize"] < 5), "FamilyType"] = "Nuclear"
#
# # 가족 수(FamilSize)가 5 이상인 승객을 가져와서, FamilyType 컬럼에 Big(대가족) 이라는 값을 넣어줍니다.
# train.loc[train["FamilySize"] >= 5, "FamilyType"] = "Big"
#
# # train 변수에 할당된 데이터의 행렬 사이즈를 출력합니다.
# # 출력은 (row, column) 으로 표시됩니다.
# print(train.shape)
#
# # train 데이터의 상위 10개를 띄우되, FamilySize와 FamilyType 컬럼만 출력합니다.
# train[["FamilySize", "FamilyType"]].head(10)
# #%%
# # pivot_table을 통해 가족 형태(FamilyType)의 변화에 따른 생존률을 출력합니다.
# pd.pivot_table(data=train, index="FamilyType", values="Survived")
# #%% md
# # 마지막으로 countplot을 활용해 이를 비교할 수 있습니다.
# #%%
# # 시본(seaborn)의 countplot으로 FamilyType별 생존자와 사망자의 차이를 시각화합니다.
# sns.countplot(data=train, x="FamilyType", hue="Survived(humanized)")
# #%% md
# # ## 마무리하며
# #
# # 지금까지 프로그래밍 언어 파이썬([Python](https://python.org/))과 파이썬의 데이터 분석 패키지 판다스([Pandas](https://pandas.pydata.org/)), 데이터 시각화 패키지 씨본([Seaborn](https://seaborn.pydata.org))과 [matplotlib](https://matplotlib.org)를 활용한 실전 예제를 살펴보았습니다. 앞서 말씀드린대로, 위 문제를 실전에서 반나절(3~4시간) 안에 해결할 수 있다면 현업에서 데이터 사이언티스트로서 일 할 수 있는 충분한 판다스 스킬을 보유했다고 볼 수 있습니다.
# #
# # 반면 1) 앞으로 데이터 분석을 업무에 활용하고자 하는 분들, 또는 2) 앞으로 데이터 사이언티스트로 취업이나 이직, 전직을 노리는 분 중, 위 문제를 반나절 안에 풀지 못한 분들은 판다스를 추가 학습해야 할 필요가 있다고 생각하시면 됩니다. 그런 분들에게는 다음의 자료를 추천합니다.
# #
# #   * [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/10min.html)
# #   * [Pandas Cookbook](http://github.com/jvns/pandas-cookbook)
# #   * [Python for Data Science](http://wavedatalab.github.io/datawithpython/)
# #   * [Modern Pandas](http://tomaugspurger.github.io/modern-1-intro.html)
# #   * [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
# #
# # 또는 위 자료를 참고하지 않고 빠른 기간 안에 판다스와 시각화를 습득하고 싶은 분들에게는 DS School의 [실전 데이터분석 과정](http://dsschool.co.kr/suggestions)을 추천해 드립니다.
# #
# # 기타 수업 관련 문의 사항은 슬랙의 전담 튜터에게 Direct Messages로 연락주세요^^ 수료증, 영수증 발급이나 기수 변경 등은 support@dsschool.co.kr 로 문의 주시면 됩니다!
# #%%
