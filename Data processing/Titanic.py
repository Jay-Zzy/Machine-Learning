from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 数据预处理
data = pd.read_csv(r"Titanic data/train.csv")
data.drop(['Cabin', 'Name', 'Ticket'], inplace=True, axis=1)
data['Age'].fillna(data['Age'].mean(), inplace=True)
data.dropna(inplace=True)
print(data.info())

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# 字符变量转换为模型能理解的数值变量
# data_ = Titanic data.copy()
# 传统写法
# labels = data_['Embarked'].unique().tolist()                          # array转list
# Titanic data['Embarked'] = Titanic data['Embarked'].apply(lambda x: labels.index(x))  # 分类的字符变数字
# data_['Sex'] = (data_['Sex'] == 'male').astype(int)                   # 性别变数字
# 调用编码库
# data_['Embarked'] = LabelEncoder().fit_transform(data_['Embarked'])
# data_['Sex'] = LabelEncoder().fit_transform(data_['Sex'])
# print(data_.head())

# 转换为哑变量，消除数值间的数学关系
X = data.loc[:, ['Sex', 'Embarked']]
enc = OneHotEncoder(categories='auto').fit(X)
result = enc.transform(X).toarray()  # 稀疏矩阵转换为数组
newdata = pd.concat([data, pd.DataFrame(result)], axis=1)
newdata.drop(['Sex', 'Embarked'], inplace=True, axis=1)
newdata.columns = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Female', 'Male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
print(newdata.head())    # 将编码后的特征添加进数据集并删除以前的


# x = data_.iloc[:, data_.columns != 'Survived']
# y = data_.iloc[:, data_.columns == 'Survived']
# # print(x.head(), '\n', y.head())
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3, random_state=3)
# # print(Xtrain.index)  # 可以发现索引由于随机划分被打乱了
# # 重新附一个index
# for i in [Xtrain, Ytrain, Xtest, Ytest]:
#     i.index = range(i.shape[0])