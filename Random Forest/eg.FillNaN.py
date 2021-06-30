import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

dataset = load_boston()
# print(dataset.data.shape)
x_full, y_full = dataset.data, dataset.target
n_samples = x_full.shape[0]
n_features = x_full.shape[1]

# 首先确定希望放入的缺失数据比例，假设50%
rng = np.random.RandomState(0)
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))  # np.floor向下取整，返回.0格式的浮点数

# 创造一个数组，包含3289个分布在0~506的行索引，和3289个分布在0~13的列索引，即可为数据中任意3289个位置赋空值(实际上有重复)
missing_features = rng.randint(0, n_features, n_missing_samples)
missing_samples = rng.randint(0, n_samples, n_missing_samples)
x_missing = x_full.copy()
y_missing = y_full.copy()
x_missing[missing_samples, missing_features] = np.nan
x_missing = pd.DataFrame(x_missing)
print(x_missing, '\n', x_missing.isnull().sum().sum())

# 均值填补
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x_missing_mean = imp_mean.fit_transform(x_missing)  # fit_transform()训练+导出
print(pd.DataFrame(x_missing_mean))

# 0值填补
imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
x_missing_0 = imp_0.fit_transform(x_missing)
print(pd.DataFrame(x_missing_0))