import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2  # 卡方检验chi2
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"Digit data/train.csv")
# print(data.head())
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
print(X.shape)

# 方差过滤：消除方差≤中位数的特征（特征中所有值相同方差为0，也被过滤掉）
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
print(X_fsvar.shape)

# 卡方过滤，k为需要的特征数。目的是选取卡方值很大，p值≤0.05的特征，这类特征与标签相关
# 方法一：暴力尝试k值组合，运算速度极慢
# score = []
# for i in range(390, 200, -10):
#     X_fschi = SelectKBest(chi2, k=i).fit_transform(X_fsvar, Y)
#     score_test = cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, Y, cv=5).mean()  # 返回预测准确度
#     score.append(score_test)
# plt.plot(range(390, 200, -10), score, label='k')
# plt.show()

# 方法二：对每个特征计算卡方值与p值，k=所有特征数-p值大于0.05的特征数（该例392个特征的数据均与label数据有关）
chivalue, pvalues = chi2(X_fsvar, Y)
k = chivalue.shape[0] - (pvalues > 0.05).sum()
X_fschi = SelectKBest(chi2, k).fit_transform(X_fsvar, Y)
print(cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, Y, cv=5).mean())
