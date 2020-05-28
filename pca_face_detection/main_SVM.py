# -*- coding: utf-8 -*-
from time import time
from PIL import Image
import glob
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import math
import os
# 配置AR数据集路径
PICTURE_PATH = u"./dataset/AR_svm/"
# PICTURE_PATH_TEST = u"./dataset/test/"
# 读取所有图片并一维化
def get_picture():
    label = 1
    TrainFiles = os.listdir(PICTURE_PATH)
    for file in TrainFiles:
        label_name = str(file[3:5])
        all_data_label.append(label_name)
        img_path = PICTURE_PATH + file
        img = Image.open(img_path)
        all_data_set.append(list(img.getdata()))


# 读取所有图片并一维化
def get_picture_test():
    label = 1
    TrainFiles = os.listdir(PICTURE_PATH_TEST)
    for file in TrainFiles:
        label_name = str(file[3:5])
        all_data_label_test.append(label_name)
        img_path = PICTURE_PATH_TEST + file
        img = Image.open(img_path)
        all_data_set_test.append(list(img.getdata()))

# 输入核函数名称和参数gamma值，返回SVM训练十折交叉验证的准确率
def SVM(kernel_name, param):
    # 十折交叉验证计算出平均准确率
    # n_splits交叉验证，随机取
    kf = KFold(n_splits=10, shuffle=True)
    precision_average = 0.0
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}  # 自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced', gamma=param),
                       param_grid)
    j = 0
    for train, test in kf.split(X):
        j = j+1
        print("kfsplit ---> ", j)
        # print(len(train))
        # print(len(test))
        # print(len(X))
        # print(len(y))
        clf = clf.fit(X[train], y[train])
        # print(clf.best_estimator_)
        test_pred = clf.predict(X[test])
        # print classification_report(y[test], test_pred)
        # 计算平均准确率
        precision = 0
        for i in range(0, len(y[test])):
            if (y[test][i] == test_pred[i]):
                precision = precision + 1
        precision_average = precision_average + float(precision) / len(y[test])
    precision_average = precision_average / 10

    return precision_average



all_data_set = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = []  # 总数据对应的类标签
all_data_set_test = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label_test = []  # 总数据对应的类标签
get_picture()
# get_picture_test()
# train_length = len(all_data_label)
# test_length = len(all_data_label_test)
# all_data_set.extend(all_data_set_test)


'''
# 输出核函数与gamma测试图
t0 = time()
n_components = 80
pca = PCA(n_components=n_components, svd_solver='auto',
              whiten=True).fit(all_data_set)
# PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
# X为降维后的数据，y是对应类标签
X = np.array(all_data_pca)
y = np.array(all_data_label)
kernel_to_test = ['rbf', 'poly', 'sigmoid']
# rint SVM(kernel_to_test[0], 0.1)
plt.figure(1)

for kernel_name in kernel_to_test:
    print("kearnel_name ---> ", kernel_name)
    x_label = np.linspace(0.0001, 1, 100)
    y_label = []
    ppp = 0
    for i in x_label:
        ppp = ppp + 1
        print("x_label --------> ", ppp)
        y_label.append(SVM(kernel_name, i))
    plt.plot(x_label, y_label, label=kernel_name)

print("done in %0.3fs" % (time() - t0))
plt.xlabel("Gamma")
plt.ylabel("Precision")
plt.title('Different Kernels Contrust')
plt.legend()
plt.show()
'''



# n_components测试(3-1)
plt.figure(4)
x_label = range(50, 58)
y_label = []
n_components_test_result={}

x_label = []

for n_components in range(70,71):
    # PCA降维
    pca = PCA(n_components=n_components, svd_solver='auto',
              whiten=True).fit(all_data_set)
    # PCA降维后的总数据集
    all_data_pca = pca.transform(all_data_set)
    eigenfaces = pca.components_.reshape((n_components, 100, 80))
    # X为降维后的数据，y是对应类标签
    X = np.array(all_data_pca)
    y = np.array(all_data_label)

    '''
    # 输出Eigenfaces
    plt.figure("Eigenfaces")
    for i in range(1, 81):
        plt.subplot(8, 10, i).imshow(eigenfaces[i-1], cmap="gray")
        plt.xticks(())
        plt.yticks(())

    plt.show()
    '''
    kf = KFold(n_splits=10, shuffle=True)
    precision_average = 0.0
    t0 = time()
    param_grid = {'C': [100],
                   'gamma': [0.01], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    print("\nn_components: " + str(n_components))
    j = 0
    for train, test in kf.split(X):
        j = j+1
        print("kfsplit ---> ", j)
        # print(len(train))
        # print(len(test))
        # print(len(X))
        # print(len(y))
        clf = clf.fit(X[train], y[train])
        # print(clf.best_estimator_)
        test_pred = clf.predict(X[test])
        # print classification_report(y[test], test_pred)
        # 计算平均准确率
        precision = 0
        for i in range(0, len(y[test])):
            if (y[test][i] == test_pred[i]):
                precision = precision + 1
        precision_average = precision_average + float(precision) / len(y[test])
    precision_average = precision_average / 10
    print(">>>>>>>>>>>>>>>",precision_average)


    # n_components_test_result[str(n_components)] = str(precision_average)
    # x_label.append(n_components)
    # y_label.append(precision_average)
    print



# print(n_components_test_result)
# plt.plot(x_label, y_label)

# plt.xlabel("n_components")
# plt.ylabel("Precision")
# plt.title('n_components_test_result')
# plt.legend()
# plt.show()

