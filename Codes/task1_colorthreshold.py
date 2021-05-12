import os
import cv2
import copy
from matplotlib import pyplot as plt
import numpy as np
import collections
import pandas as pd
import time


def read_input(inpath, name, number):
    data = {'image':[],
            'csv':[]}
    for i in range(1, number+1):
        if i < 10:
            file_name = name + '0' + str(i)
        else:
            file_name = name + str(i)
        img_path = os.path.join(inpath, file_name + '_rgb.png')
        img = cv2.imread(img_path)

        csv_path = os.path.join(inpath, file_name + '_bbox.csv')
        csv_file = pd.read_csv(csv_path, header=None, names=['c1x', 'c1y', 'c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y'])
        size_anchor = pd.DataFrame(columns=['c1x', 'c1y', 'c3x', 'c3y', 'dect'])
        size_anchor['c1x'] = csv_file['c1x'].map(lambda x: int(x))
        size_anchor['c3x'] = csv_file['c3x'].map(lambda x: int(x))
        size_anchor['c1y'] = csv_file['c1y'].map(lambda y: int(y))
        size_anchor['c3y'] = csv_file['c3y'].map(lambda y: int(y))
        size_anchor['dect'] = 0

        data['image'].append(img)
        data['csv'].append(size_anchor)

    return data


def Green(image):
    ## convert to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    #mask = cv2.inRange(hsv, (40, 30, 30), (75, 255, 255))

    ## slice the green
    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]
    #plt.imshow(green)
    #plt.show()
    # save
    #cv2.imwrite("Ggreen.png", green)
    src = cv2.GaussianBlur(green, (5, 5), 5)

    src[np.all(src == 255, axis=2)] = 0
    # Show output image
    # show('Black Background Image', src)

    median = cv2.medianBlur(src, 3)

    for i in range(2):
        median3 = cv2.medianBlur(median, 3)
        median = median3
        # show(f'median3 Image{i}', median3)

    # Create binary image from source image
    bw = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    bw_medain = cv2.medianBlur(bw, 7)
    #plt.imshow(bw_medain,'gray')
    #plt.show()
    return bw_medain


def white_black(gray_img):
    rows, cols = gray_img.shape
    new_image = copy.deepcopy(gray_img)
    for row in range(rows):
        for col in range(cols):
            if gray_img[row, col] != 0:
                new_image[row, col] = 255
    plt.imshow(new_image, 'gray')
    plt.show()
    return


def draw_anchor(image, anchor):
    for index in range(anchor.shape[0]):
        box = anchor.loc[index]
        cv2.rectangle(image, (int(box['c1x']), int(box['c1y'])), (int(box['c3x']), int(box['c3y'])),(0,0,255), 2)
    plt.imshow(image, 'gray')
    plt.show()
    return


# Merge lists with the same label
def merge_list(alist):
    L = copy.deepcopy(alist)
    length = len(L)
    for i in range(1, length):
        for j in range(i):
            L[i] = set(L[i])
            if L[i] == {0} or L[j] == {0}:
                continue
            a = L[i].union(L[j])
            b = len(L[i]) + len(L[j])
            if len(a) < b:
                L[i] = a
                L[j] = {0}
    new_list = [i for i in L if i != {0}]
    return new_list


# two-pass connected components algorithm
def two_pass(image):
    rows, cols = image.shape
    label_img = np.zeros((rows, cols),dtype=np.int)
    label_index = 0

    # first pass
    equ_neigh = []
    for r in range(rows):
        for c in range(cols):
            if image[r,c]!=255:
                neighbours = []
                if r==0:
                    if c!=0:
                        neighbours.append(label_img[r,c-1])
                else:
                    if c==0:
                        neighbours = [label_img[r-1,c],label_img[r-1,c+1]]
                    elif c==cols-1:
                        neighbours = [label_img[r-1, c-1], label_img[r-1, c], label_img[r, c-1]]
                    else:
                        neighbours = [label_img[r-1,c-1],label_img[r-1,c],label_img[r-1,c+1],
                                      label_img[r,c-1]]

                neighbours = list(set(neighbours))
                if 0 in neighbours:
                    neighbours.remove(0)
                if len(neighbours) == 0:
                    label_index += 1
                    label_img[r,c] = label_index
                    equ_neigh.append([label_index])
                else:
                    label_img[r,c] = min(neighbours)
                    if len(neighbours) > 1:
                        if neighbours not in equ_neigh:
                            equ_neigh.append(neighbours)

    # merge the list
    equ_update = merge_list(equ_neigh)
    # after merging, the num of labels
    count_label = len(equ_update)

    # Construct a dict for easy retrieval.
    # key is the label value in the genealogy, value is the corresponding min label value.
    equ_dict = {}
    for i in range(len(equ_update)):
        for curr_key in equ_update[i]:
            equ_dict[curr_key] = min(list(equ_update[i]))
    # print(equ_dict)

    # second pass
    label_sum_dict = collections.defaultdict(lambda: 0)

    for row in range(rows):
        for col in range(cols):
            if label_img[row, col] != 0:
                k = label_img[row, col]
                label_img[row, col] = equ_dict[k]
                label_sum_dict[equ_dict[k]] += 1

    label_anchor = {}
    for key in label_sum_dict.keys():
        if label_sum_dict[key] > 100:
            label_anchor[key] = {'r': set(), 'c': set()}

    return label_img, count_label, label_anchor


def anchor_component(original_img, label_image, label_anchor, model):
    rows, cols = label_image.shape
    key_list= label_anchor.keys()

    for row in range(rows):
        for col in range(cols):
            label = label_image[row, col]
            if label != 0:
                if label in key_list:
                    label_anchor[label]['r'].add(row)
                    label_anchor[label]['c'].add(col)
            else:
                continue

    bbox = pd.DataFrame(columns=['c1x', 'c1y', 'c3x', 'c3y', 'area','T/F'])

    for k in label_anchor.keys():
        y1 = min(label_anchor[k]['r'])
        x1 = min(label_anchor[k]['c'])
        y3 = max(label_anchor[k]['r'])
        x3 = max(label_anchor[k]['c'])
        area = (x3-x1)*(y3-y1)
        alist = [x1, y1, x3, y3]
        if model == 2:
            cv2.rectangle(original_img, (x1,y1),(x3,y3),(255,255,255), 7)
        bbox = bbox.append({'c1x':int(x1),'c1y':int(y1),'c3x':int(x3), 'c3y':int(y3), 'area':area}, ignore_index=True)
    if model == 2:
        plt.imshow(original_img)
        plt.show()

    return bbox


def merge_rectangle(original_img, bbox):
    rows, cols = original_img.shape[:2]
    label_img = np.zeros((rows, cols), dtype=np.int)
    bbox.sort_values("area",inplace=True, ascending=False)
    area_list = list(bbox['area'])
    threshold = sorted(area_list)[int(len(area_list) * 0.95)]

    k = 1
    for index, row in bbox.iterrows():
        x1=int(row['c1x'])
        y1=int(row['c1y'])
        x3=int(row['c3x'])
        y3=int(row['c3y'])
        cur_area = label_img[y1:y3 ,x1:x3]
        cur_label_set = set(cur_area.flatten('F'))
        if len(cur_label_set)==1 and list(cur_label_set)[0]==0:
            label_img[y1:y3 ,x1:x3] = k
            k+=1
        elif row['area']>threshold:
            label_img[y1:y3 ,x1:x3] = k
            k += 1
        else:
            cur_k = list(cur_label_set)[-1]
            label_img[y1:y3 ,x1:x3]  = cur_k

    new_label_anchor = {}
    for n in range(1, k):
        new_label_anchor[n] = {'r': set(), 'c': set()}
    new_bbox = anchor_component(original_img,label_img, new_label_anchor, 2)

    return new_bbox


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
        ovmax = -np.inf  # maximum negative
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
        ovmax = np.max(overlaps)  # max overlap
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
            p = np.max(precision[recall >= t])  # 插值
        ap = ap + p / 11.
    return ap


def task1_color_threshold():
    t0 = time.time()
    current_path = os.getcwd()
    front_path = os.path.join(current_path, 'Plant_Phenotyping_Datasets/Tray')
    ara12_path = os.path.join(front_path, 'Ara2012')
    ara13_canon_path = os.path.join(front_path, 'Ara2013-Canon')
    ara13_rpi_path = os.path.join(front_path, 'Ara2013-RPi')

    data = read_input(ara12_path, 'ara2012_tray', 16)
    # data = read_input(ara13_canon_path, 'ara2013_tray', 27)
    # data = read_input(ara13_rpi_path, 'ara2013_tray', 27)

    number = len(data['image'])
    for index in range(number):
        image = data['image'][index]
        size_anchor = data['csv'][index]

        green_gray = Green(image)
        # 反转颜色
        new_image = 255 - green_gray

        label_img, count_label, label_anchor_dict= two_pass(new_image)
        bbox = anchor_component(image, label_img, label_anchor_dict, 1)

        new_bbox = merge_rectangle(image, bbox)

        name = str(index+1) + '.png'
        cur_path = os.getcwd()
        fro_path = os.path.join(cur_path, 'Plant_Phenotyping_Datasets/Tray/Result12greendemo')
        now_path = os.path.join(fro_path, name)
        cv2.imwrite(now_path, image)
        cv2.destroyAllWindows()


        # new_image, damaged_percent = task3(green_gray, label_img, count_label, label_sum_dict)
        recall, precision, ap = compute_prec_rec(size_anchor, new_bbox, 0.5)
        print(index+1, ':',round(ap,4))

    t1 = time.time()
    print('task1 color_threshold TIME:%f' % (t1 - t0))

    return

if __name__ == '__main__':
    task1_color_threshold()
