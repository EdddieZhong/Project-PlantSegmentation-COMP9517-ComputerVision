# '''
# This deep learning model comes from network resources. We refer to it Mask-r-cnn,
# and the citation information shows below:
#
# @misc{matterport_maskrcnn_2017,
#   title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
#   author={Waleed Abdulla},
#   year={2017},
#   publisher={Github},
#   journal={GitHub repository},
#   howpublished={\url{https://github.com/matterport/Mask_RCNN}},
# }
# @article{ward2020scalable,
#   title={Scalable learning for bridging the species gap in image-based plant phenotyping},
#   author={Ward, Daniel and Moghadam, Peyman},
#   journal={Computer Vision and Image Understanding},
#   pages={103009},
#   year={2020},
#   publisher={Elsevier}
# }
#
# So before running the code, you must download the model from link and get the initial weights from train.py.
#
# This python file focus on evaluating dataset and getting results using trained SBD.
#
# '''

import os
import argparse
import matplotlib.pyplot as plt
import task3_config_cvppp as config_cvppp
import task3_train as train
from task3_inference import mask_to_rgb
from mrcnn import model
import numpy as np


OVERALL_RESULTS_FILENAME = 'overallResults.csv'


class MetricTracker(object):
    def __init__(self):
        self.metrics = []
        self.stds = []

    def add(self, metric, std):
        self.metrics.append(metric)
        self.stds.append(std)

    def calc_mean(self):
        return np.mean(self.metrics), np.std(self.metrics)


# gain arguments from keyboard
def arguments():
    parser = argparse.ArgumentParser(description='Evaluates a Mask RCNN model on the provided data with ground truth labels')
    parser.add_argument('--inputPath', type=str, required=True,
                        help='Directory containing training data stored in the expected format. See dataset_cvppp.py')
    parser.add_argument('--outputPath', type=str, required=True,
                        help='Directory to save all outputs to')
    parser.add_argument('--weightsPath', type=str, required=True,
                        help='Path to model weights to use (h5 file)')
    parser.add_argument('--showPredictions', dest='showPredictions', action='store_true')
    parser.set_defaults(showPredictions=True)

    return parser.parse_args()


# save a visualisation of prediction
def visualise_prediction(image, prediction, ground_truth, save_path):

    prediction = mask_to_rgb(prediction)
    ground_truth = mask_to_rgb(ground_truth)

    # Plot Images and save
    fig, arr = plt.subplots(1, 3)
    # RGB image
    arr[0].set_title('RGB')
    arr[0].imshow(image)
    arr[0].set_xticklabels([])
    arr[0].set_yticklabels([])
    # Ground Truth
    arr[1].set_title('Ground Truth')
    arr[1].imshow(ground_truth)
    arr[1].set_xticklabels([])
    arr[1].set_yticklabels([])
    # Prediction
    arr[2].set_title('Prediction')
    arr[2].imshow(prediction)
    arr[2].set_xticklabels([])
    arr[2].set_yticklabels([])

    plt.savefig(save_path)
    plt.cla()
    plt.close(fig)


def SBD2(mask, gt_mask, seg_val=1):

    return np.sum(mask[gt_mask==seg_val])*2.0 / (np.sum(mask) + np.sum(gt_mask))


def SBD1(masks_gt, mask_pre):
    DICEs = []
    for pred_mask_idx in range(mask_pre.shape[2]):

        best_gt = (0, None)

        for gt_mask_idx in range(masks_gt.shape[2]):
            # dice_sim = np.sum(mask_pre[:, :, pred_mask_idx][masks_gt[:, :, gt_mask_idx]==1])*2.0 / \
            #            (np.sum(mask_pre[:, :, pred_mask_idx]) + np.sum(masks_gt[:, :, gt_mask_idx]))
            dice_sim = SBD2(mask_pre[:, :, pred_mask_idx], masks_gt[:, :, gt_mask_idx])
            if dice_sim > best_gt[0]:
                best_gt = (dice_sim, gt_mask_idx)
        DICEs.append(best_gt[0])
    mDICE = np.mean(DICEs)
    std_dice = np.std(DICEs)
    return mDICE, std_dice


def SBD(prediction, ground_truth):
    dice1 = SBD1(ground_truth, prediction)
    dice2 = SBD1(prediction, ground_truth)

    if dice2[0] < dice1[0]:
        return dice2
    else:
        return dice1


# The main evaluate function
def evaluate_model():

    args = arguments()

    # Create output dir
    assert not os.path.isdir(args.outputPath), "output dir already exists"
    os.mkdir(args.outputPath)

    # Init config
    configuration = config_cvppp.InferenceConfig()

    # Init model
    inference_model = model.MaskRCNN(mode="inference",
                                     config=configuration,
                                     model_dir=args.outputPath)

    # load initial model weights
    inference_model.load_weights(args.weightsPath, by_name=True)

    # Load dataset
    test_dataset = train.CVPPP_Dataset()

    if os.path.isdir(os.path.join(args.inputPath, 'test')):
        test_dataset.load_imgs(args.inputPath, 'test')
    else:
        test_dataset.load_imgs(args.inputPath, '')
    test_dataset.prepare()

    # init metrics
    DiC = MetricTracker()

    # save predictions
    for image_id in test_dataset.image_ids:
        # Activate the choice of image
        test_dataset.image_reference(image_id)

        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            model.load_image_gt(test_dataset, configuration, image_id, use_mini_mask=False)

        image_path = test_dataset.image_reference(image_id)['path']

        # Run inference
        results = inference_model.detect([image], verbose=False)
        r = results[0]

        # SBD
        sbd_dic_res, sbd_dic_std = SBD(r['masks'], gt_mask)
        DiC.add(sbd_dic_res, sbd_dic_std)

        if args.showPredictions:
            # Save visualisation of prediction
            save_path = os.path.join(args.outputPath, os.path.basename(image_path))
            visualise_prediction(image, r['masks'], gt_mask, save_path)

    # save overall results
    with open(os.path.join(args.outputPath, OVERALL_RESULTS_FILENAME), 'a') as results_file:
        results_file.write("Metric, Result, STD\n")
        overall_dic = DiC.calc_mean()
        results_file.write("SBD, " + str(overall_dic[0]) + ', ' + str(overall_dic[1]) + '\n')
        # results_file.write("SBD, " + str(overall_dic[0]))
if __name__ == '__main__':
    evaluate_model()