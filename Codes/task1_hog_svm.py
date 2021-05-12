import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import time
import random
import copy
import glob


def read_input(inpath, name, number, DATA):
    re_size = (1647, 1158)

    for i in range(1, number+1):
        if i < 10:
            file_name = name + '0' + str(i)
        else:
            file_name = name + str(i)

        img_path = os.path.join(inpath, file_name + '_rgb.png')
        img = cv2.imread(img_path)

        # resize the image
        original_size = img.shape
        img_resize = cv2.resize(img, re_size, interpolation=cv2.INTER_CUBIC)

        # load csv files
        csv_path = os.path.join(inpath, file_name + '_bbox.csv')
        csv_file = pd.read_csv(csv_path, header=None, names=['c1x','c1y', 'c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y'])
        # resize the csv
        x_resize = re_size[0] / original_size[1]
        y_resize = re_size[1] / original_size[0]
        size_anchor = pd.DataFrame(columns = ['c1x','c1y', 'c3x', 'c3y', 'dect'])
        size_anchor['c1x'] = csv_file['c1x'].map(lambda x: int(x * x_resize))
        size_anchor['c3x'] = csv_file['c3x'].map(lambda x: int(x * x_resize))
        size_anchor['c1y'] = csv_file['c1y'].map(lambda y: int(y * y_resize))
        size_anchor['c3y'] = csv_file['c3y'].map(lambda y: int(y * y_resize))
        size_anchor['dect'] = 0
        #draw_anchor(img_resize,size_anchor)
        ''' 
        new_img = copy.deepcopy(img_resize)
        for index, box in size_anchor.iterrows():
            cv2.rectangle(new_img, (int(box['c1x']), int(box['c1y'])), (int(box['c3x']), int(box['c3y'])),
                          (255, 255, 255), 7)

        nnname = str(i) + '.png'
        cur_path = os.getcwd()
        fro_path = os.path.join(cur_path, 'Plant_Phenotyping_Datasets/Tray/Result12_groundtruth')
        now_path = os.path.join(fro_path, nnname)
        cv2.imwrite(now_path, new_img)
        cv2.destroyAllWindows()
        '''

        cur_plant, new_image= extract_plant(img_resize, size_anchor)

        #cur_no_plant = label_zero(img_resize, size_anchor)

        #np.savetxt('img_gray.csv', img_gray, delimiter=',')

        #draw_anchor(img_gray, size_anchor)
        #draw_anchor(new_image, cur_no_plant_anchor)
        DATA['image'].append(img_resize)
        DATA['anchor'].append(size_anchor)
        DATA['pos_plants'].append(cur_plant)

    return DATA


# get positive sample
def extract_plant(image, anchor):
    plants = []
    new_image = copy.deepcopy(image)
    for index, box in anchor.iterrows():
        plant = image[int(box['c1y']):int(box['c3y']), int(box['c1x']):int(box['c3x'])]
        plant = cv2.resize(plant, (128, 128), interpolation=cv2.INTER_CUBIC)
        plants.append(plant)
        new_image[int(box['c1y']):int(box['c3y']), int(box['c1x']):int(box['c3x'])] = new_image[int(box['c1y'])-1][int(box['c1x'])-1]
    #plt.imshow(new_image, 'gray')
    #plt.show()
    return plants, new_image


# Extracting negative samples and saving images
def label_zero(image, anchor):
    no_plant = []

    anchor['side_size'] = anchor.apply(lambda x: ((x['c3x'] - x['c1x']) + (x['c3y'] - x['c1y'])) / 2, axis=1)
    max_size = int(anchor['side_size'].max(axis = 0))
    min_size = int(anchor['side_size'].min(axis = 0))

    plant_number = anchor.shape[0]
    image_h = image.shape[0]
    image_w = image.shape[1]
    i = 0
    while i != (plant_number)*3:
        cur_size = random.randint(min_size, max_size)
        y1 = random.randint(1, image_h - cur_size)
        x1 = random.randint(1, image_w - cur_size)

        zero_plant = image[y1:(y1+cur_size), x1:(x1+cur_size)]

        no_plant.append(zero_plant)
        # save images
        name = 'Neg'+str(i+x1+y1)+'.png'
        cur_path = os.getcwd()
        fro_path = os.path.join(cur_path,'Plant_Phenotyping_Datasets/Tray/Ara2013-RNeg')
        now_path = os.path.join(fro_path, name)
        cv2.imwrite(now_path, zero_plant)
        cv2.destroyAllWindows()

        i+=1
    return no_plant

# load negative sample from file
def negSamples(inputpath):
    neg_list = []
    name_list = ['Ara2012Neg',
                 #'Ara2013-CanonNeg',
                 #'Ara2013-RPiNeg'
                ]
    for name in name_list:
        cur_path = os.path.join(inputpath, name)
        paths = glob.glob(os.path.join(cur_path, '*.png'))
        for path in paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            neg_list.append(img)

    return [neg_list]


def draw_anchor(image, anchor):
    for index in range(anchor.shape[0]):
        box = anchor.loc[index]
        cv2.rectangle(image, (int(box['c1x']), int(box['c1y'])), (int(box['c3x']), int(box['c3y'])),(255,255,255), 7)
    plt.imshow(image, 'gray')
    plt.show()

    return


def set_HOG():
    winSize = (128, 128)
    blockSize = (16, 16)
    blockStep = (8, 8)
    cellSize = (8, 8)
    binNum = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStep, cellSize, binNum)
    return hog


# compute the hog features
def get_HOG_list(img_list, label):
    HOGlist = []
    label_list = []
    hog = set_HOG()
    for items in img_list:
        for image in items:
            HOG = hog.compute(image)
            HOGlist.append(HOG)
            label_list.append(label)
    return HOGlist, label_list


# get hog detectors
def get_HOG_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


# Hard example
def getHardExamples(negImageList, svm, hoglist, labellist):
    hardNegList = []
    hog = set_HOG()
    hog.setSVMDetector(get_HOG_detector(svm))
    for item in negImageList:
        for image in item:
            rects, scores = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, w, h) in rects:
                hardExample = image[y:y + h, x:x + w]
                hardNegList.append(cv2.resize(hardExample, (128, 128)))
    if hardNegList != []:
        hardNegHog, hardNegLabel = get_HOG_list([hardNegList], -1)
        hoglist.extend(hardNegHog)
        labellist.extend(hardNegLabel)
    return hoglist, labellist



def train_svm(train_list, label_list):
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
    svm.train(np.array(train_list), cv2.ml.ROW_SAMPLE, np.array(label_list))

    cur_path = os.getcwd()
    model_path = os.path.join(cur_path, 'svm13c.xml')
    svm.save(model_path)
    return svm


# NMS
def fastNonMaxSuppression(boxes, sc, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1,y1,x2,y2 = [boxes[:,m]for m in range(boxes.shape[1])]

    scores = sc
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


def test_hog(svm, data, overlapthreshold):
    hog = set_HOG()

    aa = get_HOG_detector(svm)
    hog.setSVMDetector(aa)
    hog.save('myHogDector13c.bin')

    k = 1
    ap_list = []
    for index in range(len(data['image'])):
        image = data['image'][index]
        rects, scores = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)

        # fastNonMaxSuppression-- the 1st parameters
        for i in range(len(rects)):
            r = rects[i]
            rects[i][2] = r[0] + r[2]
            rects[i][3] = r[1] + r[3]

        # fastNonMaxSuppression-- the 2nd parameter
        sc = [score[0] for score in scores]
        sc = np.array(sc)

        pick = []
        #print('rects_len', len(rects))
        pick = fastNonMaxSuppression(rects, sc, overlapThresh=overlapthreshold)
        #print('pick_len = ', len(pick))

        predict_bbox = pd.DataFrame(columns=['c1x', 'c1y', 'c3x', 'c3y', 'T/F'])
        for (x1, y1, x3, y3) in pick:
            #print(x1, y1, x3, y3)
            predict_bbox = predict_bbox.append({'c1x':int(x1),'c1y':int(y1),'c3x':int(x3), 'c3y':int(y3)}, ignore_index=True)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x3), int(y3)), (255, 255, 255), 7)

        plt.imshow(image)
        plt.show()

        data['predict_anchor'].append(predict_bbox)

        #save the result image

        name = str(k) + '.png'
        cur_path = os.getcwd()
        fro_path = os.path.join(cur_path, 'Plant_Phenotyping_Datasets/Tray/Result12')
        now_path = os.path.join(fro_path, name)
        cv2.imwrite(now_path, image)
        cv2.destroyAllWindows()

        k += 1

        # evaluate the performance

        recall, precision, ap = compute_prec_rec(data['anchor'][index], predict_bbox, 0.5)
        ap_list.append(round(ap,4))

        #result[index:,] = [recall, precision]
    print([n for n in ap_list])
    print('mean:',round(np.mean(ap_list),4))

    return data


# The model predicts the intersection ratio between bbox and Groud Truth.
# IOU measures how much the two sets overlap.
# When IOU is 0, the two boxes do not overlap and there is no intersection.
# When IOU is 1, the two boxes completely overlap.
# When the value of IOU is between 0 and 1, it represents the degree of overlap between the two boxes, the higher the value, the higher the degree of overlap.
def compute_prec_rec(ground_bbox, predict_bbox, IOU_threshold):
    rows = len(predict_bbox)
    truth_num = len(ground_bbox)
    tp = np.zeros(rows)
    fp = np.zeros(rows)

    for index in range(rows):
        predict = predict_bbox.loc[index]
        '''
        #===============================
        iou_max = 0
        for i in range(truth_num):
            ground = ground_bbox.loc[i]

            g_w = ground['c3x']-ground['c1x']
            g_h = ground['c3y']-ground['c1y']

            p_w = predict['c3x'] - predict['c1x']
            p_h = predict['c3y'] - predict['c1x']

            area_g = g_w * g_h
            area_p = p_w * p_h

            w = min(predict['c1x']+p_w, ground['c1x']+g_w) - max(ground['c1x'], predict['c1x'])
            h = min(predict['c1y']+p_h, ground['c1y']+g_h) - max(ground['c1y'], predict['c1y'])
            if w <= 0 or h <= 0:
                iou = 0
            else:
                area = w*h
                iou = area / (area_g + area_p - area)
            iou_max = max(iou_max, iou)
        iou_max
        '''
        # ===============================
        ovmax = -np.inf
        ixmin = np.maximum(ground_bbox['c1x'], predict['c1x'])
        iymin = np.maximum(ground_bbox['c1y'], predict['c1y'])
        ixmax = np.minimum(ground_bbox['c3x'], predict['c3x'])
        iymax = np.minimum(ground_bbox['c3y'], predict['c3y'])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        uni = ((predict['c3x']-predict['c1x']+1.) * (predict['c3y']-predict['c1y']+1.)+
                 (ground_bbox['c3x']-ground_bbox['c1x'] + 1.) *
                 (ground_bbox['c3y']-ground_bbox['c1y'] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        max_index = np.argmax(overlaps)

        if ovmax > IOU_threshold:
            if ground_bbox['dect'][max_index]== 0:
                predict['T/F'] = 1
                tp[index] = 1
                ground_bbox['dect'][max_index] = 1
            else:
                fp[index] = 1
        else:
            predict['T/F'] = 0
            fp[index] = 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(truth_num)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = compute_ap(recall, precision)
    return recall, precision, ap


def compute_ap(recall, precision):
    ap = 0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap


# ===============================================================================
def task1_hogsvm():
    t0 = time.time()
    current_path = os.getcwd()
    front_path = os.path.join(current_path,'Plant_Phenotyping_Datasets/Tray')
    ara12_path = os.path.join(front_path, 'Ara2012')
    ara13_canon_path = os.path.join(front_path, 'Ara2013-Canon')
    ara13_rpi_path = os.path.join(front_path,'Ara2013-RPi')

    data = {'image': [],
            'anchor': [],
            'pos_plants': [],
            'predict_anchor':[]
            }
    label_map = {-1: False,
                 1: True}

    # read the input and resize
    data = read_input(ara12_path, 'ara2012_tray', 16, data)
    #data = read_input(ara13_canon_path, 'ara2013_tray', 27, data)
    #data = read_input(ara13_rpi_path, 'ara2013_tray', 27, data)

    neg_sample_list = negSamples(front_path)
    #neg_sample_list = negSamples(front_path, 'Ara2012Neg')
    #neg_sample_list.extend(negSamples(front_path, 'Ara2013-CanonNeg')
    #neg_sample_list = negSamples(front_path, 'Ara2013-RPiNeg')

    # get the pos/neg samples, hog_list and label_list
    pos_hog, pos_label_list = get_HOG_list(data['pos_plants'], 1)
    neg_hog, neg_label_list = get_HOG_list(neg_sample_list, -1)
    # Combine positive and negative samples of hog and label.
    hogList = pos_hog+neg_hog
    label_list = pos_label_list+neg_label_list

    svm = train_svm(hogList, label_list)

    # Get a hard example based on initial training results
    hogList, label_list = getHardExamples(neg_sample_list, svm, hogList, label_list)

    # Retraining after  hard example
    svm.train(np.array(hogList), cv2.ml.ROW_SAMPLE, np.array(label_list))


    threshold = 0.06
    print('threshold = ', threshold)
    data = test_hog(svm, data, overlapthreshold=threshold)

    t1 = time.time()
    print('task1 hog+svm TIME:%f'%(t1-t0))




    return





if __name__ == '__main__':
    task1_hogsvm()

