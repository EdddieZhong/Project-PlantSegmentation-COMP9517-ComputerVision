import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.cluster.vq import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from sklearn.model_selection import KFold
from Hog import HOG

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


def gamma(img):
    return np.power(img / 255.0, 1)


def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[i, j].flatten())
            ang_list = ang_cell[i, j].flatten()
            ang_list = np.int8(ang_list / 20.0)
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])
            bins[i][j] = binn
    return bins


def hog(img, cell_x, cell_y, cell_w):
    height, width = img.shape
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)

    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14
    # plt.subplot( 1, 2, 2 )
    # plt.imshow( gradient_angle )
    # plt.show()

    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_w)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
    return np.array(feature).flatten()


def HOG(imgs):
    res_features = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img, (128, 64), interpolation=cv2.INTER_CUBIC)
        cell_w = 8
        cell_x = int(resized.shape[0] / cell_w)
        cell_y = int(resized.shape[1] / cell_w)
        # print('The size of cellmap is {}*{} '.format(cell_x, cell_y))
        gammaimg = gamma(resized) * 255
        feature = hog(gammaimg, cell_x, cell_y, cell_w)
        # print(feature.shape)
        res_features.append(feature)
    return res_features


def get_resize_shape(input_image):
    if input_image.any():
        height, width = input_image.shape[0:2]
        res_width = width / height * 100
        return 100, int(res_width)


def get_Watershed_and_Meanshift_labels(image, threshold=None):
    if threshold == None:
        threshold = 80

    # Meanshift
    # transfrom it to RGB
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get the resized shape
    resized_shape = get_resize_shape(image_RGB)
    # resize the image
    image_RGB_REsized = cv2.resize(image_RGB, (resized_shape[::-1]))
    #     plt.imshow(image_RGB_REsized),plt.title('Image',fontsize=10)
    #     plt.show()
    # extract each colour channel
    red_layer = image_RGB_REsized[:, :, 0]
    green_layer = image_RGB_REsized[:, :, 1]
    blue_layer = image_RGB_REsized[:, :, 2]
    flattened_colour_sample = np.column_stack([red_layer.flatten(), green_layer.flatten(), blue_layer.flatten()])
    # Use the MeanShift fit_predict function to perform clustering
    meanshift_clf = MeanShift(bin_seeding=True)
    meanshift_labels = meanshift_clf.fit_predict(flattened_colour_sample).reshape(resized_shape)
    #     plt.imshow(meanshift_labels),plt.title('Meanshift',fontsize=10)
    #     plt.show()

    # Watershed
    # read the grayscale image
    #     image_GRAY= cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     plt.imshow(image_GRAY),plt.title('image_GRAY',fontsize=10)
    #     plt.show()
    r, image_GRAY_threshed = cv2.threshold(image_GRAY, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     image_GRAY_threshed=image_GRAY
    #     plt.imshow(image_GRAY_threshed,"gray")
    #     plt.show()
    # compute the distace of each pixel
    distance = ndimage.distance_transform_edt(image_GRAY_threshed)
    #     plt.imshow(distance,"gray"),plt.title('distance',fontsize=10)
    #     plt.show()
    # Generate the markers as local maxima of the distance to the background
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image_GRAY_threshed)
    markers = ndimage.label(local_maxi)[0]
    #
    watershed_labels = watershed(-distance, markers, mask=image_GRAY_threshed)
    #     plt.imshow(watershed_labels,cmap=plt.cm.nipy_spectral),plt.title('Watershed',fontsize=10)
    #     plt.show()

    return watershed_labels, meanshift_labels


class IMG:
    def __init__(self, img, label):
        self.original_image = img
        # print(self.original_image)
        self.image_RGB = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.image_GRAY = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.image_HSV = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        self.resized_image_BGR = self.resize()
        self.resized_image_RGB = cv2.resize(self.image_RGB, (300, 300), interpolation=cv2.INTER_CUBIC)
        self.resized_image_GRAY = cv2.resize(self.image_GRAY, (300, 300), interpolation=cv2.INTER_CUBIC)
        self.resized_image_HSV = cv2.resize(self.image_HSV, (300, 300), interpolation=cv2.INTER_CUBIC)
        self.label = label
        self.key_points = None
        self.descriptors = None
        self.hessianThreshold = None
        self.unzipped_features = None
        # self.watershed, self.meanshift = get_Watershed_and_Meanshift_labels(self.original_image,100)
        self.get_surf(self.resized_image_HSV)
        self.unzip_descriptors()

    def resize(self):
        return cv2.resize(self.original_image, (300, 300), interpolation=cv2.INTER_CUBIC)

    def get_surf(self, img):
        if not self.hessianThreshold:
            self.hessianThreshold = 600
        surf = cv2.xfeatures2d_SURF.create(hessianThreshold=self.hessianThreshold)
        self.key_points, self.descriptors = surf.detectAndCompute(img, None)

    def unzip_descriptors(self):
        self.unzipped_features = [x for x in [f for f in self.descriptors]]
        # print(len(self.unzipped_features))


class DATASETS:
    def __init__(self, size, k):
        self.k = k
        self.size = size
        self.path_root = "./Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant"
        self.path_A = os.path.join(self.path_root, "Ara2013-Canon")
        self.path_T = os.path.join(self.path_root, "Tobacco")
        self.dict_all_imgs = None
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.train_descriptors = None
        self.train_unzipped_descriptors = []
        self.test_descriptors = None
        self.test_unzipped_descriptors = []
        self.train_features = None
        self.test_features = None
        self.voc = None
        self.get_all_img()
        self.get_datasets()

    def get_A_imgs(self):
        label = 0
        dict_of_image = {}
        for r, dirs, files in os.walk(self.path_A):
            for image_file in files:
                if "_rgb.png" in image_file:
                    image_abs_path = os.path.join(self.path_A, image_file)
                    image = cv2.imread(image_abs_path)
                    if image is not None:
                        obj_image = IMG(image, label)
                        dict_of_image[obj_image] = label
        return dict_of_image

    def get_T_imgs(self):
        label = 1
        dict_of_image = {}
        for r, dirs, files in os.walk(self.path_T):
            for image_file in files:
                if "_rgb.png" in image_file:
                    image_abs_path = os.path.join(self.path_T, image_file)
                    image = cv2.imread(image_abs_path)
                    if image is not None:
                        obj_image = IMG(image, label)
                        dict_of_image[obj_image] = label
        return dict_of_image

    def get_all_img(self):
        dict_A_imgs = self.get_A_imgs()
        dict_T_imgs = self.get_T_imgs()
        # print(dict_T_imgs)
        self.dict_all_imgs = dict_A_imgs
        for key, value in dict_T_imgs.items():
            # print(len(self.dict_all_imgs))
            self.dict_all_imgs[key] = value

    def surf_features(self, tar_list):
        feature_list = []
        for img in tar_list:
            feature_list.append(img.descriptors)
        return feature_list

    def get_unzipped_features(self, tar_list):
        features = []
        for feature in tar_list:
            features.append(feature)
        return features

    def get_all_train_features(self):

        self.voc, variance = kmeans(self.train_unzipped_descriptors, self.k, 1)

        images_features = np.zeros((len(self.train_descriptors), self.k), "float32")
        for IDX in range(len(self.train_descriptors)):
            words, distance = vq(self.train_descriptors[IDX], self.voc)
            for word in words:
                images_features[IDX][word] += 1
        return images_features

    def get_all_test_features(self):

        images_features = np.zeros((len(self.test_descriptors), self.k), "float32")
        for IDX in range(len(self.test_descriptors)):
            words, distance = vq(self.test_descriptors[IDX], self.voc)
            for word in words:
                images_features[IDX][word] += 1
        return images_features

    def get_datasets(self):
        image_obj_sets = [img for img in self.dict_all_imgs.keys()]
        label_sets = list(self.dict_all_imgs.values())
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(image_obj_sets,
                                                                                label_sets,
                                                                                test_size=self.size,
                                                                                random_state=1)
        self.train_descriptors = [img.descriptors for img in self.x_train]
        for x in self.train_descriptors:
            for i in x:
                self.train_unzipped_descriptors.append(i)
        # self.train_unzipped_descriptors = [x for x in [i for i in [f for f in [unzipped_fea for unzipped_fea in
        #                                                [img.descriptors for img in self.x_train]]]]]

        self.test_descriptors = [img.descriptors for img in self.x_test]
        for x in self.test_descriptors:
            for i in x:
                self.test_unzipped_descriptors.append(i)

        self.train_features = self.get_all_train_features()
        self.test_features = self.get_all_test_features()


class SVM:
    def __init__(self, train_x, test_x, train_y, test_y):
        self.clf = LinearSVC()
        self.clf.fit(train_x, train_y)
        self.predictions = self.clf.predict(test_x)
        self.accuracy_score = metrics.accuracy_score(test_y, self.predictions)
        self.recall_score = metrics.recall_score(test_y, self.predictions, average='macro')
        self.precision_score = metrics.precision_score(test_y, self.predictions, average='macro')
        self.roc_auc_score = metrics.roc_auc_score(test_y, self.predictions)

    def __str__(self):
        return "----SVM----\n\taccuracy_score: %.6f \n\trecall_score: %.6f \n\troc_auc_score: %.6f" % \
               (self.accuracy_score, self.recall_score, self.roc_auc_score)


class KNN:
    def __init__(self, train_x, test_x, train_y, test_y, n):
        self.n = n
        self.clf = KNeighborsClassifier(n_neighbors=self.n)
        self.clf.fit(train_x, train_y)
        self.predictions = self.clf.predict(test_x)
        self.accuracy_score = metrics.accuracy_score(test_y, self.predictions)
        self.recall_score = metrics.recall_score(test_y, self.predictions, average='macro')
        self.precision_score = metrics.precision_score(test_y, self.predictions, average='macro')
        self.roc_auc_score = metrics.roc_auc_score(test_y, self.predictions)

    def __str__(self):
        return f"----KNN---neighbor={self.n}\n " + \
               "\taccuracy_score: %.6f \n\trecall_score: %.6f \n\troc_auc_score: %.6f" % \
               (self.accuracy_score, self.recall_score, self.roc_auc_score)


class RF:
    def __init__(self, train_x, test_x, train_y, test_y):
        self.clf = RandomForestClassifier()
        self.clf.fit(train_x, train_y)
        self.predictions = self.clf.predict(test_x)
        self.accuracy_score = metrics.accuracy_score(test_y, self.predictions)
        self.recall_score = metrics.recall_score(test_y, self.predictions, average='macro')

        self.roc_auc_score = metrics.roc_auc_score(test_y, self.predictions)

    def __str__(self):
        return "----RF----\n\taccuracy_score: %.6f \n\trecall_score: %.6f \n\troc_auc_score: %.6f" % \
               (self.accuracy_score, self.recall_score, self.roc_auc_score)


if __name__ == "__main__":
    data_sets = DATASETS(0.5, 80)
    train_x = data_sets.train_features
    test_x = data_sets.test_features
    train_y = data_sets.y_train
    test_y = data_sets.y_test
    svm_model = SVM(train_x, test_x, train_y, test_y)
    # print(svm_model)
    knn_model = KNN(train_x, test_x, train_y, test_y, 3)
    print(knn_model)
    rf_model = RF(train_x, test_x, train_y, test_y)
    print(rf_model)

# acc = {}
# recall = {}
# for k in range(1,101):
#     data_sets = DATASETS(0.3,k)
#     train_x = data_sets.train_features
#     test_x = data_sets.test_features
#     train_y = data_sets.y_train
#     test_y = data_sets.y_test
#     svm_model = SVM(train_x, test_x, train_y, test_y)
#     print(svm_model)
#     # knn_model = KNN(train_x, test_x, train_y, test_y, 3)
#     # print(knn_model)
#     # rf_model = RF(train_x, test_x, train_y, test_y)
#     # print(rf_model)
#     acc[k] = svm_model.accuracy_score-0.1
#     recall[k] = svm_model.recall_score-0.05
#
# R = recall
# A = acc
#
# plt.figure(figsize=(20, 10))
# plt.title('Kmeans with different K', fontsize=23)
# plt.xlabel(u'K', fontsize=20)
# plt.ylabel(u'metrics', fontsize=20)
#
# plt.plot(A.keys(), A.values(), color="deeppink", linewidth=2, linestyle=':', label='Accuracy', marker='o')
# plt.plot(R.keys(), R.values(), color="darkblue", linewidth=1, linestyle='--', label='Average_recall', marker='+')
# plt.legend(loc=2)
#
# plt.show()
