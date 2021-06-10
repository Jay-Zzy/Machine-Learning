from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
print(wine.keys())

# import pandas as pd
# wine_pd = pd.concat([pd.DataFrame(wine.Titanic data), pd.DataFrame(wine.target)], axis=1)
# print(wine_pd)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3, random_state=30)
# random_state设置分支中随机模式的参数（使结果固定）；splitter设置为'random'可以防止过拟合
clf = tree.DecisionTreeClassifier(criterion='entropy'
                                  ,random_state=11
                                  ,splitter='random'
                                  ,max_depth=5
                                  # ,min_samples_split=5
                                  # ,min_samples_leaf=5
                                  )

clf = clf.fit(Xtrain, Ytrain)
score_test = clf.score(Xtest, Ytest)  # 返回预测准确度
score_train = clf.score(Xtrain, Ytrain)
print(score_test, score_train)
print([*zip(wine.feature_names, clf.feature_importances_)])  # 查看每个特征对应的重要度


# # 绘制决策树
# import graphviz
# list1 = ['1', '2', '3']
# graph_data = tree.export_graphviz(clf
#                              ,feature_names=wine.feature_names
#                              ,class_names=list1
#                              ,filled=True
#                              ,rounded=True
#                                   )
# graph = graphviz.Source(graph_data)
# graph.render(r'E:\python\iris')


# # 通过绘制参数结果曲线观察最佳参数
# import matplotlib.pyplot as plt
# test = []
# for i in range(10):
#     clf = tree.DecisionTreeClassifier(max_depth=i+1
#                                       ,criterion='entropy'
#                                       ,random_state=11
#                                       ,splitter='random'
#                                       )
#     clf = clf.fit(Xtrain, Ytrain)
#     score_test = clf.score(Xtest, Ytest)  # 返回预测准确度
#     test.append(score_test)
# plt.plot(range(1, 11), test, label='max_depth')
# plt.legend()
# plt.show()