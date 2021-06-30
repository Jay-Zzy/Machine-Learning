import sklearn.metrics
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()  # 波士顿数据集target是连续型变量，所以不是分类了

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
score = cross_val_score(regressor, boston.data, boston.target, cv=10, scoring='neg_mean_squared_error')  # 不写scoring默认返回R方
# sorted(sklearn.metrics.SCORERS.keys())  # sklearn中所有模型评估指标列表
print(score)