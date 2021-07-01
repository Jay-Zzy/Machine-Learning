import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

dataset = load_boston()
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
x_missing[missing_samples, missing_features] = np.nan
x_missing = pd.DataFrame(x_missing)
print(x_missing, '\n缺失值共有：', x_missing.isnull().sum().sum())

# 均值填补
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x_missing_mean = imp_mean.fit_transform(x_missing)  # fit_transform()训练+导出
# print(pd.DataFrame(x_missing_mean))

# 0值填补
imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
x_missing_0 = imp_0.fit_transform(x_missing)
# print(pd.DataFrame(x_missing_0))

# 随机森林填值
sortindex = np.argsort(x_missing.isnull().sum(axis=0)).values  # 保留索引排序，返回索引
# print(sortindex)
for i in sortindex:
    df = x_missing
    fillc = df.iloc[:, i]  # 从缺失值最少的一列开始
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)  # 构建新特征矩阵
    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
    Ytrain = fillc[fillc.notnull()]
    Xtrain = df_0[Ytrain.index, :]
    Ytest = fillc[fillc.isnull()]  # 这里只是为了知道要预测的值的位置(索引)
    Xtest = df_0[Ytest.index, :]

    rfc = RandomForestRegressor(n_estimators=100).fit(Xtrain, Ytrain)
    Ypredict = rfc.predict(Xtest)
    # 对列填值(loc是根据index来索引，而iloc是根据行号来索引)
    x_missing.loc[x_missing.iloc[:, i].isnull(), i] = Ypredict
# print(x_missing, '\n缺失值共有：', x_missing.isnull().sum().sum())

# 多方法对比
X = [x_full, x_missing_mean, x_missing_0, x_missing]
mse = []
for x in X:
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    # 5折验证并求平均
    scores = cross_val_score(estimator, x, y_full, scoring='neg_mean_squared_error', cv=5).mean()
    mse.append(scores * -1)
print('\n', *zip(['x_full', 'x_missing_mean', 'x_missing_0', 'x_missing'], mse))