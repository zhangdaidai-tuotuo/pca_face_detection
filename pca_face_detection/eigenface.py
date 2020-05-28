'''
Python 3.6
# 2020/05/10
# 完成人脸检测和识别
# 修改：使用 eig 函数，对特征向量进行排序并取前占0.99的特征向量
# 修改：极大值抑制，边框颜色
# 葛健男
'''
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
    for i in range(num_face_can):
        scores=np.linalg.norm(im_blob_r-face_candidate_r[i,:],axis=1)
        index=np.argsort(scores)[0]
        print("识别的人为：" + str(index))
        xiangsi = im_blob_r[index, :].reshape(50,50)
        plt.subplot(1, 3, i + 1)
        plt.imshow(xiangsi)
        print(scores[index])
        cv2.rectangle(im_color, (int(face_rectangle[i,:][0]), int(face_rectangle[i,:][1])),
                      (int(face_rectangle[i,:][2]), int(face_rectangle[i,:][3])), (0, 0, 255), 2)
        cv2.putText(im_color, str(index+1), (int(face_rectangle[i,:][0]), int(face_rectangle[i,:][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255.0, 255.0, 255.0), 1, 1)
    plt.show() # 显示检测出来的人
    cv2.imshow("Result",im_color)
    cv2.imwrite("result_{:d}.jpg".format(jishu+1),im_color)
    # imsave(,im)
    cv2.waitKey(0)


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
import matplotlib.pyplot as plt

# 平均脸
plt.figure()
plt.gray()
plt.imshow(x_mean.reshape(50, 50))
cv2.imwrite("Average.jpg", (x_mean.reshape(50, 50)))

# 特征脸
plt.figure()
plt.gray()
# 以下选前面7个特征脸，按顺序分别显示到其余7个格子
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(V_k[:,i].reshape(50, 50))
plt.show()



x_result=reconstruct(X,x_mean,V_k)
plt.figure()
plt.gray()
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(im_faces[i, :].reshape(50, 50))
    plt.subplot(2, 4, i + 5)
    plt.imshow(x_result[i,:].reshape(50, 50))
plt.show()

# im_blob_feature = (im_faces.reshape(21,-1)-x_mean).dot(V_k)

f_test=open("group/smiling/list.txt","r")
candidate_num=3
face_candidate=np.zeros([candidate_num,window*window])
face_candidate_gt=np.zeros([candidate_num,window*window])
face_rectangle=np.zeros([candidate_num,4])
jishu=0
groud_truth=im_faces.reshape(im_faces.shape[0],-1)
for line in f_test.readlines():
    line_split=line.split('\n')
    # im_test =si.imread(line_split[0],True)
    try:
        im_test = Image.open(line_split[0])
    except IOError:
        print('fail to load image!')
    # im_temp=si.imread(line_split[0],True)
    im_test = np.array(im_test)
    im_test = cv2.cvtColor(im_test, cv2.COLOR_RGB2GRAY)
    face_rectangle,face_candidate = face_detection(im_test,x_mean,V_k,im_faces,candidate_num)
    print("##################################")
    print(face_candidate.shape) # (5, 2500)
    print(face_rectangle)
    face_recognize(im_test,face_candidate,face_rectangle,im_faces,x_mean,V_k, jishu)
    jishu+=1

