import os
import sys

import json
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

CURR_FILE_PATH= os.path.realpath(__file__)
JSON_PATH = CURR_FILE_PATH.replace("camus_image_processing.py", "config.json")

with open(JSON_PATH, 'r') as config_json:
    CONFIG_PARAMETERS = json.load(config_json)

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
    NUM_CLASSES = 1 + int(CONFIG_PARAMETERS["NUM_CLASSES"])  # background + additional classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = int(CONFIG_PARAMETERS["IMAGE_MIN_DIM"])
    IMAGE_MAX_DIM = int(CONFIG_PARAMETERS["IMAGE_MAX_DIM"])

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = eval(CONFIG_PARAMETERS["RPN_ANCHOR_SCALES"])  # anchor side in pixels

    # Reduce training ROIs per image because we have few objects (3) in the images 
    TRAIN_ROIS_PER_IMAGE = 32
    
    NUM_EPOCHS = int(CONFIG_PARAMETERS["NUM_EPOCHS"])

    # Use a small epoch (<100) or medium (<300)
    STEPS_PER_EPOCH = int(CONFIG_PARAMETERS["STEPS_PER_EPOCH"])

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = int(CONFIG_PARAMETERS["VALIDATION_STEPS"])

class CamusDataset(utils.Dataset):
    """
    Formats the Camus dataset. The dataset consists of frames of
    echocardiograms exams.
    """

    def prepare_data(self, patients_path, images_subdir="images/", masks_subdir="masks/",
                     height=128, width=128):
        images_path = os.path.join(patients_path, images_subdir)
        masks_path = os.path.join(patients_path, masks_subdir)
        images_filenames = sorted(glob(images_path + '*'))
        masks_filenames = sorted(glob(masks_path + '*'))
        
        assert len(images_filenames) == len(masks_filenames), f"There are {len(images_filenames)} images and {len(masks_filenames)} masks"
    
    def load_camus(self, patients_path, height, width):
        """Loads an image from a file and adds to dataset."""
        # Add classes
        self.add_class("camus", 1, "chamber")
    
        i = 0
        if isinstance(patients_path, str):
            patients_dir = glob(patients_path)
            for patient_path in tqdm(patients_dir, ncols=80):
                filenames = glob(patient_path + "*.jpg")
                for image_filename in filenames:
                    self.add_image("camus", image_id=i, path=image_filename,
                                   width=width, height=height)
                    i += 1
        elif isinstance(patients_path, list):
            filenames = [p for p in patients_path if p.endswith(".jpg")]
            for image_filename in filenames:
                self.add_image("camus", image_id=i, path=image_filename,
                               width=width, height=height)
                i += 1

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID."""
        info = self.image_info[image_id]
        mask_image_path = info['path'].replace("images", "masks")
        mask = cv2.imread(mask_image_path)
        mask = (np.max(mask, axis=2) if len(mask.shape) > 2 else mask).reshape((128,128,1))
        
        return mask, np.array([1,])

    def __len__(self):
        """Return the number of read images"""
        return len(self.image_info)

class InferenceConfig(CamusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def eval_detection(gt_mask, r, n_classes=1):
    """Apply the IoU metric to compare the ground truth and the detections"""
    ids = np.unique(r['class_ids'])
    merged_masks = {}
    qualities = dict([(i,0) for i in range(1, n_classes+1)])

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
    f = side / np.max(image.shape)
    
    fixed_image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    
    result_image = np.zeros((side, side) + image.shape[2:])
    
    h, l = fixed_image.shape[:2]
    result_image[:h,:l] = fixed_image
    
    return result_image

def pre_filter(image):
    """Apply morphological filter"""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3)))