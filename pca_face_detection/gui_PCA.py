# coding=utf-8
'''
5/15 模式识别大作业 
     基于PCA的人脸识别(GUI)
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


import cv2
import numpy as np
from PIL import Image
window =50

def read_face(filename):
    f=open(filename,"r")
    im_list=[]
    im_temp = []
    for line in f.readlines():
        line_split=line.split('\n')
        print("读取 " + line_split[0])
        try:
            im_temp = Image.open(line_split[0])
        except IOError:
            print('fail to load image!')
        im_temp = np.array(im_temp)
        im_list.append(im_temp)
    return im_list

def Eigenface_PCA(x):
    print("Eigenface_PCA")
    x_reshape = x.reshape(x.shape[0], -1)
    # x_mean 平均脸
    x_mean = np.mean(x_reshape, axis=0)
    # 零均值化
    X = x_reshape - x_mean
    conv_x = X.dot(X.T)
    V, D = np.linalg.eig(conv_x)
    print(V)
    # 计算投影矩阵, 此即为一组特征脸
    V_k = np.dot(X.T,D) # 2500 x 21
    for i in range(D.shape[1]):
        V_k[:,i] /= np.linalg.norm(V_k[:,i]) # 特征向量归一化
    sorted_indices = np.argsort(V) # 特征值排序结果,从小到大排序
    sorted_indices = sorted_indices[::-1]
    print(sorted_indices)
    D_sort = V_k[:, sorted_indices]
    V_sort = V[sorted_indices]
    sum1 = np.sum(V_sort)
    Vect = []
    for i in range(len(V_sort)):
        temp_sum=np.sum(V_sort[:i+1])
        if temp_sum/sum1>0.99 :
            print("最后取前 K 维, K = " + str(i))
            Vect=D_sort[:,:i+1]
            break
    return X, x_mean, Vect

def reconstruct(x,x_mean,Vect):
    # 重建
    x_hat = x.dot(Vect).dot(Vect.T)
    x_result=x_hat+x_mean
    return x_result

def face_detection(im_test,x_mean,Vect,im_faces,candidate_num=5):
    n,m=im_test.shape
    face_candidate=np.zeros([candidate_num,window*window])
    im= cv2.GaussianBlur(im_test,(9,9),0)
    scores=np.zeros(im.shape)
    face_rectangle=np.zeros([candidate_num,4])
    scores[np.where(scores==0)]=10000
    for i in range(0,n-window,2):
        for j in range(0,m-window,2):
            im_grid=im[i:i+window,j:j+window].copy()
            im_faces.reshape(21,-1)
            im_grid_r=im_grid.reshape(-1)-x_mean
            im_grid_rr=reconstruct(im_grid.reshape(-1),x_mean,Vect)
            temp_scores=np.linalg.norm(im_grid_r-im_grid_rr)
            scores[i,j]=temp_scores
    print("face_detection ********")
    sub =int(window)
    # 非极大抑制
    for i in range(sub,n-sub,int(sub/2)):
        for j in range(sub,m-sub,int(sub/2)):
            min_s= np.min(scores[i-sub:i+sub,j-sub:j+sub].reshape(-1))
            scores[np.where(scores[i-sub:i+sub,j-sub:j+sub] > min_s)[0]+i-sub,\
            np.where(scores[i-sub:i+sub,j-sub:j+sub] > min_s)[1]+j-sub]=10000
    temp_scores = scores.reshape(-1)
    index=np.argsort(temp_scores) # 从小到大排序
    x_i=np.zeros((candidate_num,),dtype=np.int)
    for i in range(candidate_num):
        x_i[i]=int(index[i]/m) # 定位 x
    x_j=index[0:candidate_num]%m # 定位 y
    current_face_candidate=0
    for i in range(candidate_num):
        if temp_scores[index[i]] != 10000:
            im_tac = im[x_i[i]:x_i[i] + window, x_j[i]:x_j[i] + window].reshape(-1)
            face_candidate[current_face_candidate,:]=im_tac
            face_rectangle[i,:]=np.array([x_j[i]+window,x_i[i]+window,x_j[i],x_i[i]],dtype=np.int)
            current_face_candidate+=1
    return face_rectangle, face_candidate

def face_recognize(im,face_candidate,face_rectangle,im_blob,x_mean,Vect,jishu):
    face_candidate_r=reconstruct(face_candidate-x_mean,x_mean,Vect)
    im_blob_r=reconstruct(im_blob.reshape(21,-1)-x_mean,x_mean,Vect)
    num_face_can=face_candidate.shape[0]
    im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    plt.figure()
    plt.gray()
    index_all = []
    for i in range(num_face_can):
        scores=np.linalg.norm(im_blob_r-face_candidate_r[i,:],axis=1)
        index=np.argsort(scores)[0]
        print("识别的人为：" + str(index))
        index_all.append(index)
        xiangsi = im_blob_r[index, :].reshape(50,50)
        plt.subplot(1, 3, i + 1)
        plt.imshow(xiangsi)
        print(scores[index])
        cv2.rectangle(im_color, (int(face_rectangle[i,:][0]), int(face_rectangle[i,:][1])),
                      (int(face_rectangle[i,:][2]), int(face_rectangle[i,:][3])), (0, 0, 255), 2)
        cv2.putText(im_color, str(index+1), (int(face_rectangle[i,:][0]), int(face_rectangle[i,:][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255.0, 255.0, 255.0), 1, 1)
    return im_color, index_all
    # plt.show() # 显示检测出来的人
    # cv2.imshow("Result",im_color)
    # cv2.imwrite("result_{:d}.jpg".format(jishu+1),im_color)
    # # imsave(,im)
    # cv2.waitKey(0)


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
        self.setWindowTitle("模式识别大作业--人脸识别PCA--葛健男")
        self.resize(500, 900)
        layout = QVBoxLayout(self)
        layout_imgs = QHBoxLayout(self)

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
        self.imageLabel.setFixedSize(500, 559)


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


        self.button1 = QPushButton('开始训练', self, clicked=self.train_pca)
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

    def train_pca(self):
        im_list=read_face("./smiling_cropped/list.txt")
        im_resize_list=[]
        im_scale_factors=[]
        # 预处理
        for im in im_list:
            # im_gray = cv2.GaussianBlur(im, (9, 9), 0)
            im_resize_rgb=cv2.resize(im,(window,window),interpolation=cv2.INTER_LINEAR)
            im_resize = cv2.cvtColor(im_resize_rgb, cv2.COLOR_RGB2GRAY)
            im_resize_list.append(im_resize)
        im_faces = np.zeros([len(im_list),window,window])
        for i in range(len(im_resize_list)):
            im_faces[i,:,:]=im_resize_list[i]

        X,x_mean,V_k=Eigenface_PCA(im_faces)
        self.im_faces = im_faces
        import matplotlib.pyplot as plt
        self.V_k = V_k
        self.x_mean = x_mean
        # 平均脸
        x_mean_show = x_mean.reshape(50, 50)

        img_pil = Image.fromarray(np.uint8(x_mean_show))
        pix = img_pil.toqpixmap().scaled(self.imageLabel.width(), self.imageLabel.height()) #QPixmap
        # pix1 = img_pil.toqimage() #QImage
        self.imageLabel.setPixmap(pix)

        # 特征脸
        # feature_face1 = V_k[:,0].reshape(50, 50)
        # feature_face2 = V_k[:,1].reshape(50, 50)
        # feature_face3 = V_k[:,2].reshape(50, 50)
        # feature_face4 = V_k[:,3].reshape(50, 50)
        # img_pil1 = Image.fromarray(np.uint8(feature_face1))
        # pix1 = img_pil1.toqpixmap().scaled(self.image_test_1.width(), self.image_test_1.height()) #QPixmap
        # # pix1 = img_pil.toqimage() #QImage
        # self.image_test_1.setPixmap(pix1)
        # img_pil2 = Image.fromarray(np.uint8(feature_face2))
        # pix2 = img_pil2.toqpixmap().scaled(self.image_test_1.width(), self.image_test_1.height()) #QPixmap
        # # pix1 = img_pil.toqimage() #QImage
        # self.image_test_2.setPixmap(pix2)
        # img_pil3 = Image.fromarray(np.uint8(feature_face3))
        # pix3 = img_pil3.toqpixmap().scaled(self.image_test_1.width(), self.image_test_1.height()) #QPixmap
        # # pix1 = img_pil.toqimage() #QImage
        # self.image_test_3.setPixmap(pix3)
        # img_pil4 = Image.fromarray(np.uint8(feature_face4))
        # pix4 = img_pil4.toqpixmap().scaled(self.image_test_1.width(), self.image_test_1.height()) #QPixmap
        # # pix1 = img_pil.toqimage() #QImage
        # self.image_test_4.setPixmap(pix4)

        print_txt = "库数据读取完成！ 已显示平均脸(左)  库数据路径:" + "./smiling_cropped/"
        self.tx.setPlainText(print_txt)
    def select_pic(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.tga;;*.png;;All Files(*)")

        [self.source_path, self.imgName] = os.path.split(imgName)
        im_test = Image.open(imgName)
        pix = im_test.toqpixmap().scaled(self.imageLabel.width(), self.imageLabel.height()) #QPixmap
        # jpg = QtGui.QPixmap(imgName).scaled(self.imageLabel.width(), self.imageLabel.height())
        self.imageLabel.setPixmap(pix)


        self.src_img = imgName
        self.src_img_name = self.imgName
        print_txt = "打开图片为" + self.imgName + "  label = " + str(self.imgName[3:5])
        self.tx.setPlainText(print_txt)
        print(self.source_path)



    def test_svm(self):

        im_faces = self.im_faces
        x_mean = self.x_mean
        V_k = self.V_k
        candidate_num=3
        face_candidate=np.zeros([candidate_num,window*window])
        face_candidate_gt=np.zeros([candidate_num,window*window])
        face_rectangle=np.zeros([candidate_num,4])
        jishu=0
        groud_truth=im_faces.reshape(im_faces.shape[0],-1)
        # im_test =si.imread(line_split[0],True)
        try:
            im_test = Image.open(self.src_img)
        except IOError:
            print('fail to load image!')
            print_txt = 'fail to load image!'
            self.tx.setPlainText(print_txt)
        # im_temp=si.imread(line_split[0],True)
        im_test = np.array(im_test)
        im_test = cv2.cvtColor(im_test, cv2.COLOR_RGB2GRAY)
        face_rectangle,face_candidate = face_detection(im_test,x_mean,V_k,im_faces,candidate_num)
        print("##################################")
        print(face_candidate.shape) # (5, 2500)
        print(face_rectangle)
        im_color, index_all = face_recognize(im_test,face_candidate,face_rectangle,im_faces,x_mean,V_k, jishu)

        img_pil = Image.fromarray(np.uint8(im_color))
        pix = img_pil.toqpixmap().scaled(self.imageLabel.width(), self.imageLabel.height()) #QPixmap
        # pix1 = img_pil.toqimage() #QImage
        self.imageLabel.setPixmap(pix)
        index1 = index_all[0]
        index2 = index_all[1]
        index3 = index_all[2]
        if index1 > 9:
            file1 = './smiling_cropped/' + str(index1) + '.tga'
        else:
            file1 = './smiling_cropped/0' + str(index1) + '.tga'
        if index2 > 9:
            file2 = './smiling_cropped/' + str(index2) + '.tga'
        else:
            file2 = './smiling_cropped/0' + str(index2) + '.tga'
        if index3 > 9:
            file3 = './smiling_cropped/' + str(index3) + '.tga'
        else:
            file3 = './smiling_cropped/0' + str(index3) + '.tga'

        print(file1)
        im_test = Image.open(file1)
        pix = im_test.toqpixmap().scaled(self.image_test_1.width(), self.image_test_1.height())
        self.image_test_1.setPixmap(pix)
        im_test = Image.open(file2)
        pix = im_test.toqpixmap().scaled(self.image_test_1.width(), self.image_test_1.height())
        self.image_test_2.setPixmap(pix)
        im_test = Image.open(file3)
        pix = im_test.toqpixmap().scaled(self.image_test_1.width(), self.image_test_1.height())
        self.image_test_3.setPixmap(pix)

        print_txt = "检测完毕!!!  搜索结果见左图，识别结果见右图"
        self.tx.setPlainText(print_txt)
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