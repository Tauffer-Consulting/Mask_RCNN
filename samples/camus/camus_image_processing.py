import os
import sys

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


class CamusConfig(Config):
    """
    Configuration for training on the Camus dataset.
    Derives from the base Config class and overrides values specific
    to the Camus dataset.
    """
    # Give the configuration a recognizable name
    NAME = "camus"

    # Train on 1 GPU and 4 images per GPU. Batch size is 4 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 heart structures

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because we have few objects (3) in the images 
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch (<100) or medium (<300)
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class CamusDataset(utils.Dataset):
    """
    Formats the Camus dataset. The dataset consists of frames of
    echocardiograms exams.
    """

    def load_camus(self, patients_path, height, width):
        """Loads an image from a file and adds to dataset."""
        # Add classes
        self.add_class("camus", 1, "ventricule")
        self.add_class("camus", 2, "muscle")
        self.add_class("camus", 3, "atrium")
    
        i = 0
        patients_dir = glob(patients_path)
        for patient_path in tqdm(patients_dir, ncols=80):
            filenames = glob(patient_path + "*_resized.png")
            for image_filename in filenames:
                if '_gt' in image_filename:
                    continue

                self.add_image("camus", image_id=i, path=image_filename,
                               width=width, height=height)
                i += 1

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID."""
        info = self.image_info[image_id]
        mask_image_path = info['path'].replace("_resized.png", "_gt_resized.png")
        mask = cv2.imread(mask_image_path)
        # If grayscale. Convert to RGB for consistency.
        if mask.ndim != 3:
            mask = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if mask.shape[-1] == 4:
            mask = mask[..., :3]
        
        return mask, np.array([1, 2, 3])

    def __len__(self):
        """Return the number of read images"""
        return len(self.image_info)

class InferenceConfig(CamusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def eval_detection(gt_mask, r):
    ids = np.unique(r['class_ids'])
    merged_masks = {}
    qualities = dict([(1,0), (2,0), (3,0)])

    for i in ids:
        merged_masks[i] = (np.zeros_like(r['masks'][:,:,0]) > 0)
        n_masks = r['masks'].shape[-1]
        for c in range(n_masks):
            if i != r['class_ids'][c] or r['scores'][c] < .7:
                continue

            mask = r['masks'][:,:,c]
            merged_masks[i] |= mask

        gt_mask_i = (gt_mask[:,:,i-1] > 0)
        intersection = gt_mask_i & merged_masks[i]
        union = gt_mask_i | merged_masks[i]
        qualities[i] = np.sum(intersection) / np.sum(union)
    
    return qualities, merged_masks


def fix_resolution(image, side=128):
    """Adjust image resolution to a square shape without distortion"""
    f = 128 / np.max(image.shape)
    
    fixed_image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    
    result_image = np.zeros((side, side))
    
    h, l = fixed_image.shape[:2]
    result_image[:h,:l] = fixed_image
    
    return result_image

def pre_filter(image):
    """Apply morphological filter"""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3)))