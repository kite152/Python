import numpy as np
from numpy import tile
from numpy import sum
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2)
model = KNN(n_neighbors=5)
model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
accuracy_score_score = accuracy_score(y_test, y_pred)
print("准确率为：", accuracy_score_score)
# print(x_test)
# x1 = np.array([[1.5,3,5.8,2.2], [6.2, 2.9, 4.3, 1.3]])
# x1_pred = model.predict(x1)
# print(x1_pred)

def createDataSet():
    group = iris.data # 样本点数据
    labels = iris.target
    return group, labels


# 使用KNN进行分类
def KNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] 表示行数

    # 计算欧氏距离
    diff = tile(newInput, (numSamples, 1)) - dataSet  # 计算元素属性值的差
    squaredDiff = diff ** 2  # 对差值取平方
    squaredDist = sum(squaredDiff, axis=1)  # 按行求和
    distance = squaredDist ** 0.5

    # 对距离进行排序
    # argsort() 返回按照升序排列的数组的索引
    sortedDistIndices = np.argsort(distance)

    classCount = {}  # 定义字典
    for i in range(k):
        # 选择前k个最短距离
        voteLabel = labels[sortedDistIndices[i]]

        # 累计标签出现的次数
        # 如果在标签在字典中没有出现的话, get()会返回0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        # 返回得到的投票数最多的分类
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex


dataSet, labels = createDataSet()

testX = np.array([1.5,3,5.8,2.2])
k = 5
outputLabel = KNNClassify(testX, dataSet, labels, k)
print("Your input is:", testX, "and classified to class: ", outputLabel)

testX = np.array([6.2, 2.9, 4.3, 1.3])
outputLabel = KNNClassify(testX, dataSet, labels, k)
print("Your input is:", testX, "and classified to class: ", outputLabel)