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
# So before running the code, you must download the model from link, get the initial weights from train.py.
#
# This python file focus on predicting dataset and getting output images using trained model.
#
# '''


import os
from glob import glob
import argparse
import task3_config_cvppp as config_cvppp
from mrcnn import model, visualize
import numpy as np
import cv2 as cv
from skimage import io
from matplotlib import pyplot as plt
import pylab


# Converts a mask to RGB Format
def mask_to_rgb(mask):

    colours = visualize.random_colors(mask.shape[2])
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(mask.shape[2]):
        for c in range(3):
            rgb_mask[:, :, c] = np.where(mask[:, :, i] != 0, int(colours[i][c] * 255), rgb_mask[:, :, c])

    return rgb_mask


### 333
def load_image(im_path):

    image = cv.imread(im_path, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Check for alpha channel
    if not image.shape[2] <= 3:
        image = image[:, :, :3]

    return image


# gain arguments from keyboard
def arguments():
    parser = argparse.ArgumentParser(description='Performs inference using a Mask RCNN Model')
    parser.add_argument('--dataPattern', type=str, required=True,
                        help="A glob file path pattern in quotations. e.g. 'path/*_rgb.png'")
    parser.add_argument('--outputPath', type=str, required=True,
                        help='Directory to save all outputs')
    parser.add_argument('--weightsPath', type=str, required=True,
                        help='Path to model weights (.h5)')

    return parser.parse_args()


# The main prediction function
def predict_segmentations():

    args = arguments()

    image_pattern = args.dataPattern

    print("Image Pattern:", image_pattern)

    # Create output dir
    assert not os.path.isdir(args.outputPath), "output dir already exists, please try again"
    os.mkdir(args.outputPath)

    # Init config
    configuration = config_cvppp.InferenceConfig()

    # Init model
    inference_model = model.MaskRCNN(mode="inference",
                                     config=configuration,
                                     model_dir=args.outputPath)

    inference_model.load_weights(args.weightsPath, by_name=True)

    # Predict Images
    with open(os.path.join(args.outputPath, 'leafCounts.csv'), 'a') as count_file:
        count_file.write("Image, Count\n")
        for im_path in glob(image_pattern):
            out_path = os.path.join(args.outputPath, os.path.basename(im_path))

            print("Saving prediction for", im_path, "at", out_path)

            try:
                image = load_image(im_path)
            except:
                print("Bad File for prediction:", im_path)
                continue

            # blur images
            # image = cv.GaussianBlur(image, (101, 101), 92, 0)

            # predict images
            results = inference_model.detect([image])
            # import matplotlib.pyplot as plt
            # plt.imshow(results)

            # convert images to RGB format
            rgb_mask = mask_to_rgb(results[0]['masks'])

            # store images
            # cv.imwrite(out_path, rgb_mask.astype(np.uint8), cv.IMWRITE_PNG_COMPRESSION)
            io.imsave(out_path, rgb_mask.astype(np.uint8))
            io.imshow(rgb_mask.astype(np.uint8))
            # view.show()
            plt.show()
            # sore result of leaf-counting

            count_file.write(os.path.basename(im_path) + ", " + str(results[0]['masks'].shape[2]) + "\n")

if __name__ == '__main__':
    predict_segmentations()