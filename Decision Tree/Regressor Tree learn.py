from sklearn import tree
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt


# 交叉验证
boston = load_boston()
print(boston.keys())
regressor = tree.DecisionTreeRegressor(random_state=0)
# 交叉验证10次训练集测试集的划分效果（R2越接近1越好）
score = cross_val_score(regressor, boston.data, boston.target, cv=10
                # , scoring='neg_mean_squared_error'
                )
print(score)


# 自建数据，小试牛刀
rng = np.random.RandomState(1)           # 限制范围0~1
X = np.sort(5*rng.rand(80, 1), axis=0)   # 生成80行1列的2维随机数集合
Y = np.sin(X).ravel()                    # X是二维的但Y必须是一维，用ravel()降维
Y[::5] += 3 * (0.5 - rng.rand(16))       # 加入噪声：每5个数据取一个加上随机数

regr_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_2 = tree.DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, Y)
regr_2.fit(X, Y)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]  # 另一种增维方法
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

plt.figure()
plt.scatter(X, Y, s=20, edgecolors='darkorange', c='orange')
plt.plot(X_test, y_1, label='depth_2', color='skyblue')
plt.plot(X_test, y_2, label='depth_5', color='yellowgreen')
plt.xlabel('Titanic data')
plt.ylabel('target')
plt.legend()
plt.show()