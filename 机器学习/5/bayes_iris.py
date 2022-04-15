# coding=UTF-8
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
datas = load_iris()
X = datas.data
Y = datas.target
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=2)
pca = PCA(n_components=2)
train_x = pca.fit_transform(train_x, train_y)
test_x = pca.fit_transform(test_x, test_y)
gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
test_y_hat = gaussian.predict(test_x)
min_train_x1, min_train_x2 = np.min(train_x, axis=0)
max_train_x1, max_train_x2 = np.max(train_x, axis=0)
min_test_x1, min_test_x2 = np.min(test_x, axis=0)
max_test_x1, max_test_x2 = np.max(test_x, axis=0)
N = 500
x1min = min(min_train_x1, min_test_x1)
x1max = max(max_train_x1, max_test_x1)
x1 = np.linspace(x1min, x1max, N)

x2min = min(min_train_x2, min_test_x2)
x2max = max(max_train_x2, max_test_x2)
x2 = np.linspace(x2min, x2max, N)

x1, x2 = np.meshgrid(x1, x2)

X = np.c_[x1.ravel(), x2.ravel()]

y_hat = gaussian.predict(X)

plt.figure()
plt.subplot(2, 1, 1)
import matplotlib.colors as colors

cmp = colors.ListedColormap([u'indigo', u'gold', u'hotpink', u'firebrick', u'indianred'])

plt.pcolormesh(x1, x2, y_hat.reshape(x1.shape), cmap=cmp)

plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y)
plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y)

plt.title("高斯贝叶斯" + str(gaussian.score(test_x, test_y)))
plt.subplot(2, 1, 2)
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize

label = label_binarize(test_y, classes=(0, 1, 2))

fpr, tpr, threofor = metrics.roc_curve(label.ravel(), gaussian.predict_proba(test_x).ravel())
plt.plot(fpr, tpr, c="red", label="ROC")
plt.title("AUC" + str(metrics.auc(fpr, tpr)))
plt.legend()
plt.tight_layout(1)
plt.show()
