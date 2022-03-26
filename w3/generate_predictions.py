from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from dataset_gestions import update_labels
from dataset_gestions import get_frames_paths
from tqdm import tqdm
from dataset_gestions import write_predictions
from os.path import exists
import numpy as np
import os

import cv2

def predict(frames_paths, model_name, rewrite=False):
    """
    Does inference in detectron2
    :param frames_paths: path where all the frames of the sequence are located
    :param model_name: name of the model that is going to be used in detectron2
    :param rewrite: rewrites .txt
    :return: dictionary of labels updated
    """

    # id of the model for detectron2
    model_id = "COCO-Detection/" + model_name

    # CONFIGURATION
    # Model config
    cfg = get_cfg()

    # Run a model in detectron2's core library: get file and weights
    cfg.merge_from_file(model_zoo.get_config_file(model_id))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)

    # Hyper-params
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold used to filter out low-scored bounding boxes in predictions
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'

    # Initialize predictor
    predictor = DefaultPredictor(cfg)

    # Dictionary of all the annotations of the frames
    labels= {}
    print('generating predictions...')
    for frame_path in tqdm(frames_paths):
        # Read image and take the string of the path (example: frame 2 => '0002')
        im = cv2.imread(frame_path)
        imgname = (frame_path.split('/')[-1]).split('.')[0]

        # Do inference and get the bboxes, confidence and classes
        outputs = predictor(im)
        bboxes = outputs["instances"].pred_boxes
        conf = outputs["instances"].scores
        classes = outputs["instances"].pred_classes

        # For each frame upload the labels dictionary
        for bbox, conf, pred_class in zip(bboxes, conf, classes):
            if pred_class == 2:  # we detect a car
                score = conf.cpu().numpy()
                bbox_det = bbox.cpu().numpy()
                update_labels(labels, imgname, bbox_det[0], bbox_det[1], bbox_det[2], bbox_det[3], score)

    print('Labels uploaded')

    # save predictions in the txt if rewrite=True of it not exists
    model_name = model_name.replace('.yaml', '')

    if rewrite or not exists(f'off_the_shelve/{model_name}.txt'):
        write_predictions(labels, model_name)

    return labels



