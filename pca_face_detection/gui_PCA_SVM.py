# coding=utf-8
'''
5/15 模式识别大作业 
     基于PCA-SVM的人脸识别(GUI)
     葛健男
     gejn@mail.ustc.edu.cn
'''
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os

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


import qdarkstyle
from PyQt5.QtPrintSupport import QPageSetupDialog, QPrintDialog, QPrinter


# 配置AR数据集路径
PICTURE_PATH = u"./dataset/AR/"
PICTURE_PATH_TEST = u"./dataset/AR_test/"


# 读取所有图片并一维化
def get_picture():
    label = 1
    all_data_set = []
    all_data_label = []
    all_names_train = []
    TrainFiles = os.listdir(PICTURE_PATH)
    for file in TrainFiles:
        label_name = str(file[3:5])
        all_data_label.append(label_name)
        img_path = PICTURE_PATH + file
        img = Image.open(img_path)
        all_data_set.append(list(img.getdata()))
        all_names_train.append(file)
    return all_data_set, all_data_label, all_names_train


# 读取所有图片并一维化
def get_picture_test():
    label = 1
    all_data_set_test = []
    all_data_label_test = []
    all_names = []
    TrainFiles = os.listdir(PICTURE_PATH_TEST)
    for file in TrainFiles:
        label_name = str(file[3:5])
        all_data_label_test.append(label_name)
        img_path = PICTURE_PATH_TEST + file
        img = Image.open(img_path)
        all_data_set_test.append(list(img.getdata()))
        all_names.append(file)

    return all_data_set_test, all_data_label_test, all_names


# 关于
class AboutUsDialog(QDialog):
    def __init__(self, parent=None):
        super(AboutUsDialog, self).__init__(parent)
        self.setWindowTitle(u'关于我们')
        self.resize(520, 320)
        self.setFixedSize(self.width(), self.height())
        self.setStyleSheet("QDialog{border-image:url(./beijing.jpg);}")
        self.version_label = QLabel(self)
        self.version_label.setStyleSheet("color:white")
        self.mummy_label = QLabel(self)
        self.mummy_label.setStyleSheet("color:white")
        self.copyright_label = QLabel(self)
        self.copyright_label.setStyleSheet("color:white")
        # self.ok_button = QPushButton(self)
        # self.ok_button = QPushButton('选择图片', self, clicked="hide()")
        # self.ok_button.setFixedSize(75, 25)
        # self.ok_button.setStyleSheet("QPushButton{border:1px solid lightgray;background:rgb(230,230,230)}"
        #                              "QPushButton:hover{border-color:green;background:transparent}")
        self.setWindowFlags(Qt.Dialog)
        self.h_layout = QHBoxLayout()
        self.v2_layout = QVBoxLayout()
        self.v2_layout.addWidget(self.version_label)
        self.v2_layout.addWidget(self.mummy_label)
        self.v2_layout.addWidget(self.copyright_label)
        self.v2_layout.addStretch()
        self.v2_layout.setSpacing(10)
        self.v2_layout.setContentsMargins(40, 0, 20, 10)
        self.h_layout.addLayout(self.v2_layout)
        # self.h_layout.addLayout(self.v3_layout)
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addStretch()
        # self.bottom_layout.addWidget(self.ok_button)
        self.bottom_layout.setSpacing(0)
        self.bottom_layout.setContentsMargins(0, 0, 30, 20)
        self.main_layout = QVBoxLayout(self)
        # self.main_layout.addLayout(self.title_layout)
        self.main_layout.addStretch()
        self.main_layout.addLayout(self.h_layout)
        self.main_layout.addLayout(self.bottom_layout)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)
        self.translateLanguage()
        self.all_names = []

    def translateLanguage(self):
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.version_label.setFont(font)
        self.mummy_label.setFont(font)
        self.copyright_label.setFont(font)
        # self.ok_button.setFont(font)
        self.version_label.setText(u"基于SVM-PCA的人脸识别 (模式识别大作业)")
        self.mummy_label.setText(u"中国科学技术大学 葛健男")
        self.copyright_label.setText(u"Copyrigfault_waringht(c) Rights Reserved.")


    def isInTitle(self, xPos, yPos):
        return yPos < 30 and xPos < 456

class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowOpacity(0.95)  # 设置窗口透明度
        # 美化风格
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowTitle("模式识别大作业--人脸识别—PCA_SVM--葛健男")
        self.resize(500, 900)
        layout = QVBoxLayout(self)
        layout_imgs = QHBoxLayout(self)
        # self.imageLabel = QLabel(self)
        # self.imageLabel.setAlignment(Qt.AlignCenter)
        # self.imageLabel.setFixedSize(400, 559)
        # layout_imgs.addWidget(self.imageLabel)
        # self.imageLabel_result = QLabel(self)
        # self.imageLabel_result.setAlignment(Qt.AlignCenter)
        # self.imageLabel_result.setFixedSize(348, 559)


        self.image_test_1 = QLabel(self)
        self.image_test_1.setAlignment(Qt.AlignCenter)
        self.image_test_1.setFixedSize(200, 280)
        self.image_test_2 = QLabel(self)
        self.image_test_2.setAlignment(Qt.AlignCenter)
        self.image_test_2.setFixedSize(200, 280)
        self.image_test_3 = QLabel(self)
        self.image_test_3.setAlignment(Qt.AlignCenter)
        self.image_test_3.setFixedSize(200, 280)
        self.image_test_4 = QLabel(self)
        self.image_test_4.setAlignment(Qt.AlignCenter)
        self.image_test_4.setFixedSize(200, 280)


        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setFixedSize(400, 559)


        test_1 = QHBoxLayout(self)
        test_1.addWidget(self.image_test_1)
        test_1.addWidget(self.image_test_2)
        test_1.setSpacing(2)
        test_2 = QHBoxLayout(self)
        test_2.addWidget(self.image_test_3)
        test_2.addWidget(self.image_test_4)
        test_2.setSpacing(2)
        test_show = QVBoxLayout(self)
        test_show.addLayout(test_1)
        test_show.addLayout(test_2)

        source_show = QVBoxLayout(self)
        source_show.addWidget(self.imageLabel)

        layout_imgs.addLayout(source_show)
        layout_imgs.addLayout(test_show)
        layout_imgs.setSpacing(3)



        # layout_imgs.addWidget(self.imageLabel_result)
        # layout_imgs.setSpacing(10)
        layout.addItem(layout_imgs)
        clayout = QHBoxLayout()
        layout.addItem(clayout)


        self.button1 = QPushButton('开始训练', self, clicked=self.train_svm)
        self.button2 = QPushButton('测试图片', self, clicked=self.select_pic)
        self.button3 = QPushButton('显示结果', self, clicked=self.test_svm)
        self.button4 = QPushButton('关于我们', self, clicked=self.about_us)
        self.button1.setStyleSheet("QPushButton{color:black}"
                                  "QPushButton:hover{color:red}"
                                  "QPushButton{background-color:rgb(255,128,128)}"
                                  "QPushButton{border:2px}"
                                  "QPushButton{border-radius:10px}"
                                  "QPushButton{padding:2px 4px}")
        self.button1.setMinimumHeight(30)
        self.button2.setStyleSheet("QPushButton{color:black}"
                                  "QPushButton:hover{color:red}"
                                  "QPushButton{background-color:rgb(255,128,128)}"
                                  "QPushButton{border:2px}"
                                  "QPushButton{border-radius:10px}"
                                  "QPushButton{padding:2px 4px}")
        self.button2.setMinimumHeight(30)
        self.button3.setStyleSheet("QPushButton{color:black}"
                                  "QPushButton:hover{color:red}"
                                  "QPushButton{background-color:rgb(255,128,128)}"
                                  "QPushButton{border:2px}"
                                  "QPushButton{border-radius:10px}"
                                  "QPushButton{padding:2px 4px}")
        self.button3.setMinimumHeight(30)
        self.button4.setStyleSheet("QPushButton{color:black}"
                                  "QPushButton:hover{color:red}"
                                  "QPushButton{background-color:rgb(255,128,128)}"
                                  "QPushButton{border:2px}"
                                  "QPushButton{border-radius:10px}"
                                  "QPushButton{padding:2px 4px}")
        self.button4.setMinimumHeight(30)
        clayout.addWidget(self.button1)
        clayout.addWidget(self.button2)
        clayout.addWidget(self.button3)
        clayout.addWidget(self.button4)
        clayout.setSpacing(10)
        # 设置文本显示框
        text = QHBoxLayout(self)
        self.tx = QTextEdit(self)
        text.addWidget(self.tx)
        layout.addItem(text)
        self.tx.setFontPointSize(15)
        self.tx.setPlainText("算法运行结果显示")

    def train_svm(self):
        # all_data_set = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
        # all_data_label = []  # 总数据对应的类标签
        # all_data_set_test = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
        # all_data_label_test = []  # 总数据对应的类标签
        print_txt = "正在训练...  路径为:" + PICTURE_PATH
        self.tx.setPlainText(print_txt)
        all_data_set, all_data_label,all_names_train  =  get_picture()
        all_data_set_test, all_data_label_test,all_names = get_picture_test()

        self.all_names_train = all_names_train
        self.all_names = all_names


        train_length = len(all_data_label)
        test_length = len(all_data_label_test)
        all_data_set.extend(all_data_set_test)
        n_components = 80
        pca = PCA(n_components=n_components, svd_solver='auto',
                  whiten=True).fit(all_data_set)
        # PCA降维后的总数据集
        all_data_pca = pca.transform(all_data_set)
        eigenfaces = pca.components_.reshape((n_components, 100, 80))
        # X为降维后的数据，y是对应类标签
        X = np.array(all_data_pca)
        y = np.array(all_data_label)
        self.X_test = X[train_length:-1]
        self.y_test = np.array(all_data_label_test)

        y_test = np.array(all_data_label_test)
        t0 = time()
        param_grid = {'C': [100],
                           'gamma': [0.01], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        self.clf = clf.fit(X[0:train_length], y)
        

        print_txt = "训练完成！  训练集路径:" + PICTURE_PATH
        self.tx.setPlainText(print_txt)


    def select_pic(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.tif;;*.png;;All Files(*)")
        [self.source_path, self.imgName] = os.path.split(imgName)
        jpg = QtGui.QPixmap(imgName).scaled(self.imageLabel.width(), self.imageLabel.height())
        self.imageLabel.setPixmap(jpg)
        self.src_img = imgName
        self.src_img_name = self.imgName

        print_txt = "打开图片为" + self.imgName + "  label = " + str(self.imgName[3:5])
        self.tx.setPlainText(print_txt)
        print(self.source_path)



    def test_svm(self):
        test_index = self.all_names.index(self.imgName)

        X_test_cur_1 = self.X_test[test_index]
        X_test_cur = np.expand_dims(X_test_cur_1, axis=0)
        y_test_cur = self.y_test[test_index]
        print(X_test_cur.shape)
        print(X_test_cur)
        test_pred = self.clf.predict(X_test_cur)
        if y_test_cur == test_pred:
            print("correct!!!")
            print_txt = "预测类别:" + str(test_pred) +  "  分类正确 correct prediction! "
            test_show = y_test_cur
        else:
            print("incorrect!!!")
            print_txt = "预测类别:" + str(test_pred) +  "  实际类别:" + str(y_test_cur)
        self.tx.setPlainText(print_txt)
        img_path_show = []
        t = 0
        for name in self.all_names_train:
            if str(name[3:5]) == test_show:
                img_path = PICTURE_PATH + name
                img_path_show.append(img_path)
                t = t+1
            if t == 4:
                break

        self.srcImage = QImage(img_path_show[0])
        jpg = QtGui.QPixmap(img_path_show[0]).scaled(self.image_test_1.width(), self.image_test_1.height())
        self.image_test_1.setPixmap(jpg)
        self.srcImage = QImage(img_path_show[1])
        jpg = QtGui.QPixmap(img_path_show[1]).scaled(self.image_test_2.width(), self.image_test_2.height())
        self.image_test_2.setPixmap(jpg)
        self.srcImage = QImage(img_path_show[2])
        jpg = QtGui.QPixmap(img_path_show[2]).scaled(self.image_test_3.width(), self.image_test_3.height())
        self.image_test_3.setPixmap(jpg)
        self.srcImage = QImage(img_path_show[3])
        jpg = QtGui.QPixmap(img_path_show[3]).scaled(self.image_test_4.width(), self.image_test_4.height())
        self.image_test_4.setPixmap(jpg)

        # precision = 0
        # for i in range(0, len(y_test)):
        #     if (y_test[i] == test_pred[i]):
        #         precision = precision + 1

        # print(precision/test_length)

    def doVerFilp(self):
        # 垂直翻转
        self.srcImage = self.srcImage.mirrored(False, True)
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.srcImage))


    def about_us(self):
        self.about = AboutUsDialog()
        self.about.setModal(True)  # 设置模态
        self.about.show()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    # app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())