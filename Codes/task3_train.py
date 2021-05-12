#
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
# So before running the code, you must download the model from link, get the initial weights in advance and store it in src director.
#
# This python file focus on loading dataset and getting weights of model from training dataset.
#
# '''


import os
import argparse
import task3_config_cvppp as config_cvppp
from mrcnn import model
from imgaug import augmenters as iaa
import numpy as np
from skimage import io, color
import cv2 as cv
from mrcnn import utils


# Directory names
CHECKPOINT_DIR = 'weights_output'
CLASS_NAME = 'cvppp'
OBJECT_CLASS_NAME = 'leaf'


# gain arguments from keyboard
def arguments():
    # CMD arguments
    parser = argparse.ArgumentParser(description='Trains a Mask RCNN Model')
    parser.add_argument('--inputPath', type=str, required=True,
                        help='Directory containing training data.')
    parser.add_argument('--outputPath', type=str, required=True,
                        help='Directory to save all outputs')
    # parser.add_argument('--name', type=str, default='MaskRCNN_exp',
    #                     help='Experiment name')
    # parser.add_argument('--init ', type=str, required=True,
    #                     help='The initial weights of the model.')
    parser.add_argument('--numEpochs', type=int, required=True,
                        help='Number of training epochs')
    return parser.parse_args()


# load images
def load_datasets(args):
    train_dataset = Fine_Tune_CVPPP_Dataset(blur_images=True)
    train_dataset.load_imgs(args.inputPath, 'train')
    train_dataset.prepare()

    crossVal_dataset = Fine_Tune_CVPPP_Dataset(blur_images=True)
    crossVal_dataset.load_imgs(args.inputPath, 'crossVal')
    crossVal_dataset.prepare()

    return train_dataset, crossVal_dataset


# overwrite from mrcnn.utils.Dataset
class CVPPP_Dataset(utils.Dataset):
    """
    Loads the CVPP dataset to play with
    """

    def __init__(self, blur_images=False, class_map=None, image_type='png'):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        self.blur_images = blur_images

        self.image_type = image_type

    def load_imgs(self, input_dir, subset):
        """
        Loads the images from the subset
        """
        import glob
        image_dir = os.path.join(input_dir, subset)

        self.add_class(CLASS_NAME, 1, OBJECT_CLASS_NAME)

        image_id = 0
        for i, fname in enumerate(glob.glob(os.path.join(image_dir, '*_rgb.png'))):

            self.add_image(CLASS_NAME, image_id, fname,
                           mask_path=fname.replace('rgb', 'label'))
            # print(fname)
            image_id += 1

        if image_id == 0:
            raise OSError('No images in: ' + image_dir)


    def load_mask(self, image_id):
        """
        Loads an image mask from its id

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a CVPP image, delegate to parent class.
        image_info = self.image_info[image_id]
        mask = color.rgb2grey(io.imread(image_info['mask_path']))

        # Remove alpha channel if it exists
        if mask.ndim == 4:
            mask = mask[:, :, :3]

        leaves = np.unique(mask)

        expMask = np.zeros((mask.shape[0], mask.shape[1], len(leaves) - 1))
        i = 1
        while i < len(leaves):
            l = mask.copy()
            l[l != leaves[i]] = False
            l[l != 0] = True
            expMask[:, :, i - 1] = l
            i += 1

        # Filter out masks smaller than a threshold
        smallMaskThresh = 3
        validMaskIndexes = []

        for i in range(expMask.shape[2]):
            if not np.sum(expMask[:, :, i]) < smallMaskThresh:
                validMaskIndexes.append(i)

        filteredExpandedMask = np.zeros((mask.shape[0], mask.shape[1], len(validMaskIndexes)))

        for maskNum, maskIdx in enumerate(validMaskIndexes):
            filteredExpandedMask[:, :, maskNum] = expMask[:, :, maskIdx]

        class_ids = np.array([1] * expMask.shape[2])
        return expMask, class_ids.astype(np.int32)


    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
            If it is an RGBA image, drop the alpha channel
        """
        # Load image
        image = cv.imread(self.image_info[image_id]['path'], 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Check for alpha channel
        if image.shape[2] == 4:
            image = image[:, :, :3]

        if self.blur_images:
            image = self.gaussian_blur(image)
        return image

    def gaussian_blur(self, image):
        '''
        use a gaussian-blur for each image.
             kernel size = 5x5, sigmaX = 3, sigmaY = 0
        '''
        return cv.GaussianBlur(image, (5, 5), 3, 0)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == CLASS_NAME:
            return info
        else:
            return info


# overwrite from CVPPP_Dataset
class Fine_Tune_CVPPP_Dataset(CVPPP_Dataset):

    #Loads all the images
    def __init__(self, blur_images=False, image_type='png'):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        self.blur_images = blur_images

        self.image_type = image_type

    def load_imgs(self, input_dir, subset):
        """
        Loads the images from the subset
        """
        import glob
        image_dir = os.path.join(input_dir, subset)

        if not os.path.isdir(image_dir):
            raise OSError('Invalid Data Directory at: ' + image_dir)

        self.add_class(CLASS_NAME, 1, OBJECT_CLASS_NAME)

        image_id = 0
        image_files = glob.glob(os.path.join(image_dir, '*_rgb.png'))

        for i, fname in enumerate(image_files):
            self.add_image(CLASS_NAME, image_id, fname,
                          mask_path=fname.replace('rgb', 'label'))
            image_id += 1

        if image_id == 0:
            raise OSError('No training images in: ' + image_dir)


# Define the augmentation for training
def get_augmentation_sequence():

    # Macro to apply something with 50% chance
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Augmentation applied to every image
    # Augmentors sampled one value per channel
    aug_sequence = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),

            sometimes(iaa.CropAndPad(
                percent=(-0.25, 0.25),
                pad_mode=['constant', 'edge'],
                pad_cval=(0, 0)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 0),
                mode='constant'
            )),
            iaa.GaussianBlur((0, 3.0)),
            iaa.Add((-10, 10), per_channel=0.7),
            iaa.AddToHueAndSaturation((-20, 20)),
        ],
        random_order=True
    )

    return aug_sequence


# The main training function
def train():
    # get arguments
    args = arguments()

    if not os.path.isdir(args.outputPath):
            os.mkdir(args.outputPath)

    # for folder in [CHECKPOINT_DIR]:
    folder_path = os.path.join(args.outputPath, CHECKPOINT_DIR)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    # Load Params
    configuration = config_cvppp.TrainConfig()

    ## Load dataset API (Already logged in the args log step)
    train_dataset, crossVal_dataset = load_datasets(args)

    # Init the model
    checkpoint_path = os.path.join(args.outputPath, CHECKPOINT_DIR)
    training_model = model.MaskRCNN('training', configuration, checkpoint_path)

    # Load weights
    # if not os.path.exists(args.init):
    #     raise OSError('No weights at: ' + args.init)

    training_model.load_weights('mask_rcnn_cvppp.h5', by_name=True)

    # Train the model
    augmentation = get_augmentation_sequence()

    custom_callbacks = None

    training_model.train(train_dataset, crossVal_dataset,
                         learning_rate=configuration.LEARNING_RATE,
                         epochs=args.numEpochs,
                         augmentation=augmentation,
                         layers='all',
                         custom_callbacks=custom_callbacks)  # Train all layers

if __name__ == "__main__":
    train()