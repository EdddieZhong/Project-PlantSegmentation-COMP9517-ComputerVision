from __future__ import print_function
import os
import cv2
import numpy as np
import argparse
import random as rng
import time
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift
rng.seed(12345)


def meanBlur(img):
    MeanBlur = cv2.blur(img, (7, 7))
    M2 = cv2.subtract(img, MeanBlur)
    meanBlur_sharpen = cv2.addWeighted(img, 1, M2, 1.5, 0)
    return meanBlur_sharpen


def medianBlur(img):
    medianBlur = cv2.medianBlur(img, 7)
    M3 = cv2.subtract(img, medianBlur)
    medianBlur_sharpen = cv2.addWeighted(img, 1, M3, -2, 0)
    return medianBlur_sharpen


def Gaussian(img):
    Gaussian = cv2.GaussianBlur(img, (7, 7), 15)
    M = cv2.subtract(img, Gaussian)
    Gaussian_sharpen = cv2.addWeighted(img, 1, M, 1, 0)
    return Gaussian_sharpen


def show(name, img):
    fig = plt.gcf()
    fig.set_size_inches(40, 40)
    plt.imshow(img), plt.title(name)
    plt.xticks([]), plt.yticks([])
    plt.show()

def show_with_path(name, Path):
    img = cv2.imread(Path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig = plt.gcf()
    fig.set_size_inches(40, 35)
    plt.imshow(img), plt.title(name,fontsize = 100)
    plt.xticks([]), plt.yticks([])
    plt.show()

def opening(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return opening


def green(image):
    ## convert to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    # mask = cv2.inRange(hsv, (40, 30, 30), (75, 255, 255))

    ## slice the green
    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]

    # save
    cv2.imwrite("T3Ggreen.png", green)

    # fig = plt.gcf()
    # fig.set_size_inches(40, 40)
    # plt.subplot(211), plt.imshow(image), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(212), plt.imshow(green), plt.title('green')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    return green


def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def get_leaf(image_BGR):
    image_green = green(image_BGR)

    src = cv2.GaussianBlur(image_green, (5, 5), 5)
    # cv2.imwrite("Ggreen.png", src)
    if src is None:
        print('Could not open or find the image:')
        exit(0)
    # Show source image
    # show('Source Image', src)

    src[np.all(src == 255, axis=2)] = 0
    # Show output image
    # show('Black Background Image', src)

    median = cv2.medianBlur(src, 3)
    # show('MedianBlured Image', median)
    # cv2.imwrite("median9.png", median)

    for i in range(4):
        median3 = cv2.medianBlur(median, 3)
        median = median3
        # show(f'median3 Image{i}', median3)
    # cv2.imwrite("medianmedianmedian333.png", median)

    # Create binary image from source image
    bw = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imwrite('GBinary.png', bw)
    bw_medain = cv2.medianBlur(bw, 7)
    # bw_medain= cv2.medianBlur(bw_medain, 25)
    open_bw_medain = opening(opening(bw_medain))
    # cv2.imwrite('Binary_leaf.png', bw_medain)
    # cv2.imwrite('GBinary_median_Open.png', open_bw_medain)
    return open_bw_medain


def IoU(val, image):
    intersection = np.logical_and(val, image)
    union = np.logical_or(val, image)
    iou_score = np.sum(intersection) / np.sum(union)
    # print('IoU is %s' % iou_score)
    return iou_score


def KMEANS(image):
    image2 = image.reshape((-1, 3))
    image2 = np.float32(image2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 19
    attempts = 10
    ret, label, center = cv2.kmeans(image2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)
    # fig = plt.gcf()
    # fig.set_size_inches(40,40)
    # plt.imshow(res2)
    # plt.axis('off')
    return res2


def watershed_and_Dilation(img):
    # img = green(cv2.imread(path))
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((2, 2), np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

    # Threshold
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 0] = 255

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # fig = plt.gcf()
    # fig.set_size_inches(40, 40)
    #
    # plt.subplot(421), plt.imshow(rgb_img)
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(422), plt.imshow(thresh, 'gray')
    # plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(423), plt.imshow(closing, 'gray')
    # plt.title("morphologyEx:Closing:2x2"), plt.xticks([]), plt.yticks([])
    # plt.subplot(424), plt.imshow(sure_bg, 'gray')
    # plt.title("Dilation"), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(425), plt.imshow(dist_transform, 'gray')
    # plt.title("Distance Transform"), plt.xticks([]), plt.yticks([])
    # plt.subplot(426), plt.imshow(sure_fg, 'gray')
    # plt.title("Thresholding"), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(427), plt.imshow(unknown, 'gray')
    # plt.title("Unknown"), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(428), plt.imshow(img, 'gray')
    # plt.title("Result from Watershed"), plt.xticks([]), plt.yticks([])
    #
    # plt.tight_layout()
    # plt.show()
    # cv2.imwrite("Dilation.png",sure_bg)
    # cv2.imwrite("Watershed Result.png",img)
    return img


if __name__ == '__main__':

    dice_list = []
    IoU_list = []
    rgb_image_list = []
    fg_image_list = []
    leaf_image_list = []
    kmeans_image_list = []
    watershed_image_list = []
    path1 = "Ara2013-Canon"
    path2 = "Ara2012"
    num = 0

    start_threshold = time.clock()

    for path in [path1, path2]:
        for r_d, d, files in os.walk(path):
            for file in files:

                if "_rgb.png" in file:

                    rgb_path = os.path.join(path, file)
                    fg_path = os.path.join(path, file.replace("_rgb.png", "_fg.png"))
                    # print(rgb_path,fg_path)
                    # num += 1


                    image_temp_rgb = cv2.imread(rgb_path)
                    rgb_image_list.append(image_temp_rgb)
                    g = get_leaf(image_temp_rgb)

                    leaf_image_list.append(g)
                    image_temp_fg = cv2.imread(fg_path, cv2.IMREAD_GRAYSCALE)
                    fg_image_list.append(image_temp_fg)

    elapsed_threshold = (time.clock() - start_threshold)

    start_kmeans = time.clock()
    for rgb_image in rgb_image_list:
        kmeans_image_list.append(get_leaf(KMEANS(rgb_image)))

    elapsed_kmeans = (time.clock() - start_kmeans)

    start_watershed = time.clock()
    for rgb_image in rgb_image_list:
        watershed_image_list.append(get_leaf(watershed_and_Dilation(rgb_image)))
    elapsed_watershed = (time.clock() - start_watershed)

    # print(len(rgb_image_list))
    # print(len(fg_image_list))
    #
    # print(leaf_image_list[0].shape)
    # print(fg_image_list[0].shape)
    if len(leaf_image_list) == len(fg_image_list):
        for i in range(len(leaf_image_list)):
            leaf_image = leaf_image_list[i]
            target_fg_image = fg_image_list[i]
            dice_list.append(dice(leaf_image, target_fg_image))
            IoU_list.append(IoU(target_fg_image, leaf_image))
    print("=====Threshold=====")
    print("Timing:", elapsed_threshold)
    print("median of dice:", sorted(dice_list)[21])
    print("median of IoU:", sorted(IoU_list)[21])
    print("average dice:", sum(dice_list) / len(dice_list))
    print("average IoU:", sum(IoU_list) / len(IoU_list))

    dice_list_K = []
    IoU_list_K = []
    if len(kmeans_image_list) == len(fg_image_list):
        for i in range(len(kmeans_image_list)):
            leaf_image = kmeans_image_list[i]
            target_fg_image = fg_image_list[i]
            dice_list_K.append(dice(leaf_image, target_fg_image))
            IoU_list_K.append(IoU(target_fg_image, leaf_image))
    print("=====K-means=====")
    print("Timing:", elapsed_kmeans)
    print("median of dice:", sorted(dice_list_K)[21])
    print("median of IoU:", sorted(IoU_list_K)[21])
    print("average dice:", sum(dice_list_K) / len(dice_list_K))
    print("average IoU:", sum(IoU_list_K) / len(IoU_list_K))

    dice_list_W = []
    IoU_list_W = []
    if len(watershed_image_list) == len(fg_image_list):
        for i in range(len(watershed_image_list)):
            leaf_image = watershed_image_list[i]
            target_fg_image = fg_image_list[i]
            dice_list_W.append(dice(leaf_image, target_fg_image))
            IoU_list_W.append(IoU(target_fg_image, leaf_image))
    print("=====Watershed=====")
    print("Timing:", elapsed_watershed)
    print("median of dice:", sorted(dice_list_W)[21])
    print("median of IoU:", sorted(IoU_list_W)[21])
    print("average dice:", sum(dice_list_W) / len(dice_list_W))
    print("average IoU:", sum(IoU_list_W) / len(IoU_list_W))

    best_index = dice_list.index(max(dice_list))
    worst_index = dice_list.index(min(dice_list))

    cv2.imwrite("best_watershed.png", watershed_image_list[best_index])
    cv2.imwrite("best_threshold.png", leaf_image_list[best_index])
    cv2.imwrite("best_Kmeans.png", kmeans_image_list[best_index])

    cv2.imwrite("worst_watershed.png", watershed_image_list[worst_index])
    cv2.imwrite("worst_threshold.png", leaf_image_list[worst_index])
    cv2.imwrite("worst_Kmeans.png", kmeans_image_list[worst_index])

    show_with_path("best_watershed.png","best_watershed.png")
    show_with_path("best_threshold.png","best_threshold.png")
    show_with_path("best_Kmeans.png","best_Kmeans.png")
    show_with_path("worst_watershed.png","worst_watershed.png")
    show_with_path("worst_threshold.png","worst_threshold.png")
    show_with_path("worst_Kmeans.png","worst_Kmeans.png")


    plt.figure(figsize=(60, 40))
    plt.title('DSC results"', fontsize=25)
    plt.xlabel(u'Image', fontsize=25)
    plt.ylabel(u'metrics', fontsize=25)

    plt.plot(range(len(dice_list)), dice_list, color="deeppink", linewidth=5, linestyle=':', label='Threshold',
             marker='o')
    plt.plot(range(len(dice_list_K)), dice_list_K, color="darkblue", linewidth=5, linestyle='--',
             label='Kmeans-cluster', marker='+')
    plt.plot(range(len(dice_list_W)), dice_list_W, color="green", linewidth=5, linestyle='-.', label='Watershed',
             marker='*')
    plt.legend(loc=2, fontsize=25)
    plt.savefig("DSC results_.png")
    plt.show()

    plt.figure(figsize=(60, 40))
    plt.title('IoU Results', fontsize=25)
    plt.xlabel(u'Images', fontsize=25)
    plt.ylabel(u'metrics', fontsize=25)

    plt.plot(range(len(IoU_list)), IoU_list, color="deeppink", linewidth=5, linestyle=':', label='Threshold',
             marker='o')
    plt.plot(range(len(IoU_list_K)), IoU_list_K, color="darkblue", linewidth=5, linestyle='--', label='Kmeans-cluster',
             marker='+')
    plt.plot(range(len(IoU_list_W)), IoU_list_W, color="green", linewidth=5, linestyle='-.', label='Watershed',
             marker='*')
    plt.legend(loc=3, fontsize=25)
    plt.savefig("IoU results_.png")
    plt.show()
