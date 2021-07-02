from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = load_breast_cancer()
rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
print(score_pre)

# scorel = []
# for i in range(60, 81):  # 之前已经从1~200(step=10)发现峰值在60~80之间
#     rfc = RandomForestClassifier(n_estimators=i,
#                                  n_jobs=-1,
#                                  random_state=90)  # n_jobs设定工作的core数量,等于-1的时候，表示cpu里的所有core进行工作
#     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
#     scorel.append(score)
# print(max(scorel), ([*range(60, 81)][scorel.index(max(scorel))]))  # 0.9666353383458647 73
# plt.figure(figsize=[20, 5])
# plt.plot(range(60, 81), scorel)
# plt.show()

# 网格搜索(该例没啥帮助)
rfc = RandomForestClassifier(n_estimators=73, random_state=90)

param_grid = {'max_depth': np.arange(1, 20, 1)}
GS = GridSearchCV(rfc, param_grid, cv=10).fit(data.data, data.target)
print(GS.best_params_, GS.best_score_)

param_grid = {'min_samples_leaf': np.arange(1, 11, 1)}
GS = GridSearchCV(rfc, param_grid, cv=10).fit(data.data, data.target)
print(GS.best_params_, GS.best_score_)