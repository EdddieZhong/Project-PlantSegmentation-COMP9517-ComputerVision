import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.cluster.vq import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics


def show_his(img):
    dic = dict.fromkeys(range(0, 255), 0)
    for column in range(img.shape[0]):
        for row in range(img.shape[1]):
            pixel_value = img[column, row]
            dic[pixel_value] += 1
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    return 0


def gray_threshold(img):
    # read the Q1 img as grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # get the Vmin and Vmax of img
    Vmin = img.min()
    Vmax = img.max()
    a, b = 0, 255
    c, d = Vmin, Vmax
    # output the transformed pic
    Tr1 = np.array((img - c) * ((b - a) / (d - c)), dtype='float32')

    # plt.imshow(Tr1, 'gray', vmin=0, vmax=255)
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    return Tr1


def meanBlur(img):
    MeanBlur = cv2.blur(img, (7, 7))
    M2 = cv2.subtract(img, MeanBlur)
    meanBlur_sharpen = cv2.addWeighted(img, 1, M2, 1.5, 0)
    return meanBlur_sharpen


def medianBlur(img):
    medianBlur = cv2.medianBlur(img, 5)
    M3 = cv2.subtract(img, medianBlur)
    medianBlur_sharpen = cv2.addWeighted(img, 1, M3, 3, 0)
    return medianBlur_sharpen


def Gaussian(img):
    Gaussian = cv2.GaussianBlur(img, (15, 15), 15)
    M = cv2.subtract(img, Gaussian)
    Gaussian_sharpen = cv2.addWeighted(img, 1, M, 2, 0)
    return Gaussian_sharpen


def get_feature_surf(img_list):
    fea_list = []
    for img in img_list:
        key, desc = cv2.xfeatures2d.SURF_create(1000).detectAndCompute(img, None)
        fea_list.append(desc)
    return fea_list


def feature_struct(des_list):
    res_decs = []
    for des in des_list:
        for fea in des:
            res_decs.append(fea)
    return res_decs


A_path = "./Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon"
T_path = "./Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Tobacco"
A_path_list = []
T_path_list = []

# get all path of each file
for r_d, d, files in os.walk(A_path):
    for file in files:
        if "_rgb.png" in file:
            img_path_temp_A = os.path.join(A_path, file)
            A_path_list.append(img_path_temp_A)

for r_d, d, files in os.walk(T_path):
    for file in files:
        if "_rgb.png" in file:
            img_path_temp_T = os.path.join(T_path, file)
            T_path_list.append(img_path_temp_T)
gaussian_img_all = []
mean_img_all = []
median_img_all = []
gray_threshold_img_all = []
all_img_list = []
all_label_list = []
for p in A_path_list:
    # print(p)
    temp_img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    temp_img_Gaussian = Gaussian(temp_img)
    temp_img_meanBlur = meanBlur(temp_img)
    temp_img_medianBlur = medianBlur(temp_img)
    temp_img_gray_threshold = gray_threshold(temp_img)
    all_img_list.append(cv2.resize(temp_img, (250, 250), interpolation=cv2.INTER_CUBIC))
    gaussian_img_all.append(cv2.resize(temp_img_Gaussian, (250, 250), interpolation=cv2.INTER_CUBIC))
    mean_img_all.append(cv2.resize(temp_img_meanBlur, (250, 250), interpolation=cv2.INTER_CUBIC))
    median_img_all.append(cv2.resize(temp_img_medianBlur, (250, 250), interpolation=cv2.INTER_CUBIC))
    gray_threshold_img_all.append(cv2.resize(temp_img_gray_threshold, (250, 250), interpolation=cv2.INTER_CUBIC))
    all_label_list.append(1)

for p in T_path_list:
    temp_img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    temp_img_Gaussian = Gaussian(temp_img)
    temp_img_meanBlur = meanBlur(temp_img)
    temp_img_medianBlur = medianBlur(temp_img)
    temp_img_gray_threshold = gray_threshold(temp_img)
    all_img_list.append(cv2.resize(temp_img, (250, 250), interpolation=cv2.INTER_CUBIC))
    gaussian_img_all.append(cv2.resize(temp_img_Gaussian, (250, 250), interpolation=cv2.INTER_CUBIC))
    mean_img_all.append(cv2.resize(temp_img_meanBlur, (250, 250), interpolation=cv2.INTER_CUBIC))
    median_img_all.append(cv2.resize(temp_img_medianBlur, (250, 250), interpolation=cv2.INTER_CUBIC))
    gray_threshold_img_all.append(cv2.resize(temp_img_gray_threshold, (250, 250), interpolation=cv2.INTER_CUBIC))
    all_label_list.append(0)

# split
x_train, x_test, y_train, y_test = train_test_split(gaussian_img_all, all_label_list, test_size=0.3, random_state=0)

# features
desc_in_train_list = get_feature_surf(x_train)
unzipped_desc_in_train_list = feature_struct(desc_in_train_list)
k = 80
VoC, Var = kmeans(unzipped_desc_in_train_list, k, 1)

flatten_train_features_list = np.zeros((len(desc_in_train_list), k), "float32")
for IDX in range(len(desc_in_train_list)):
    points, distance_D = vq(desc_in_train_list[IDX], VoC)
    for point in points:
        flatten_train_features_list[IDX][point] += 1

# create model
classes = ["Arabidopsis", "Tobacco"]
SVM_clf = LinearSVC()
SVM_clf.fit(flatten_train_features_list, y_train)

# predict
desc_in_test_list = get_feature_surf(x_test)
# test_des = HOG(x_test)
unzipped_desc_in_test_list = feature_struct(desc_in_test_list)

flatten_test_features_list = np.zeros((len(desc_in_test_list), k), "float32")
for IDX in range(len(desc_in_test_list)):
    points, distance_D = vq(desc_in_test_list[IDX], VoC)
    for point in points:
        flatten_test_features_list[IDX][point] += 1

predictions = SVM_clf.predict(flatten_test_features_list)
print("----SVM --------")
# show metrics
accuracy_score = metrics.accuracy_score(y_test, predictions)
recall_score = metrics.recall_score(y_test, predictions, average='macro')
precision_score = metrics.precision_score(y_test, predictions, average='macro')
roc_auc_score = metrics.roc_auc_score(y_test, predictions)
print("accuracy_score: %.6f \n precision_score: %.6f \nrecall_score: %.6f " %
      (accuracy_score, precision_score, recall_score))

# initialize the classifier model
b_n_n = 8
KNN_clf = KNeighborsClassifier(n_neighbors=b_n_n)
# fit the model
KNN_clf.fit(flatten_train_features_list, y_train)
# predict
predict_y_KNN_3 = KNN_clf.predict(flatten_test_features_list)

print("----KNN --------")
print(f"number of neighbors : {b_n_n}")
print("accuracy : %.7f" % metrics.accuracy_score(y_test, predict_y_KNN_3))
print("precision : %.7f" % metrics.precision_score(y_test, predict_y_KNN_3, average='macro'))
print("average-recall : %.7f" % metrics.recall_score(y_test, predict_y_KNN_3, average='macro'))
# print("confusion matrix :\n",metrics.confusion_matrix(Y_test,predict_y_KNN_3))


# a = {}
# p = {}
# r = {}
# for nn in range(2, 16):
#     KNN_clf = KNeighborsClassifier(n_neighbors=nn)
#     # fit the model
#     KNN_clf.fit(flatten_train_features_list, y_train)
#     # predict
#     predict_y_KNN = KNN_clf.predict(flatten_test_features_list)
#     a[nn] = metrics.accuracy_score(y_test, predict_y_KNN)-0.005
#     p[nn] = metrics.precision_score(y_test, predict_y_KNN, average='macro')-0.005
#     r[nn] = metrics.recall_score(y_test, predict_y_KNN, average='macro')-0.005
#
# plt.figure(figsize=(20, 10))
# plt.title('KNN with different K', fontsize=23)
# plt.xlabel(u'K', fontsize=20)
# plt.ylabel(u'metrics', fontsize=20)
#
# plt.plot(a.keys(), a.values(), color="deeppink", linewidth=2, linestyle=':', label='Accuracy', marker='o')
# plt.plot(p.keys(), p.values(), color="darkblue", linewidth=1, linestyle='--', label='precision_score', marker='+')
# plt.plot(r.keys(), r.values(), color="green", linewidth=1, linestyle='-.', label='Average_recall', marker='*')
# plt.legend(loc=2)
#
# plt.show()


# p = "ara2013_plant026_rgb.png"
# temp_img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
# temp_img_Gaussian = Gaussian(temp_img)
# temp_img_meanBlur = meanBlur(temp_img)
# temp_img_medianBlur = medianBlur(temp_img)
#
# # %%
#
# plt.subplot(121), plt.imshow(temp_img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(temp_img_Gaussian), plt.title('temp_img_Gaussian filter 5*5')
# plt.xticks([]), plt.yticks([])
#
# cv2.imwrite("a.png", temp_img_Gaussian)
# plt.show()
#
#
# def get_surf(img, hessianThreshold):
#     surf = cv2.xfeatures2d_SURF.create(hessianThreshold=hessianThreshold)
#     keypoints, descriptors = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
#     return keypoints, descriptors
#
#
# def show_surf(img):
#     empty_array = np.array([])
#     keypoints, descriptors = list(get_surf(img, 600))
#     surfed_image = cv2.drawKeypoints(img, keypoints, empty_array)
#     plt.imshow(surfed_image)
#     plt.xticks([]), plt.yticks([])
#     cv2.imwrite("x.png", surfed_image)
#
#
# show_surf(temp_img)
#
# show_surf(temp_img_Gaussian)
#
# show_surf(temp_img_meanBlur)
#
# show_surf(temp_img_medianBlur)
