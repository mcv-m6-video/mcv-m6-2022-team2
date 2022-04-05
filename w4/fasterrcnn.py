import os
import wandb
import torch
import cv2
import numpy as np
import motmetrics as mm
from tqdm import tqdm
from sort.sort import Sort

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from utils.image_utils import read_txt_save_videos
from register_dataset import register_city_challenge
from utils.detectron_utils import MyTrainer

DDBB_ROOT =  '../../data/AICity_data/train/'
MODEL_ID = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" # Faster R-CNN (X101-FPN)

# Dictionary to chose what to do:
# If TRAIN in ACTIONS, the model will be trained
# If EVALUATE in ACTIONS, IDF1 will be calculated
# If INFER in ACTIONS, some gifs will be generated
# ACTIONS = ['TRAIN', 'EVALUATE', 'INFER']
ACTIONS = ['EVALUATE', 'INFER']

if __name__ == "__main__":

    # --- PREPARE ENVIRONMENT ---
    # Create the folder to store the frames individually
    print('CAREFUL!!! The following function will store around 34000 files and 29.5GB of data in your computer.')
    read_txt_save_videos(txt_path='splits/train.txt')
    read_txt_save_videos(txt_path='splits/test.txt')

    # val same as test since we are training and testing with the same sequence WTF
    for txt_paths, set in zip(['splits/train.txt', 'splits/test.txt', 'splits/test.txt'], ["train", "val", "test"]):
        DatasetCatalog.register("CITY_CHALLENGE_" + set, lambda set=set: register_city_challenge(txt_path=txt_paths))
        MetadataCatalog.get("CITY_CHALLENGE_" + set).set(thing_classes=["Car"])

    # --- CONFIGURATIONS ---
    # Model config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ID))  # model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ID)  # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # car

    # Dataset config
    cfg.DATASETS.TRAIN = ("CITY_CHALLENGE_train",)
    cfg.DATASETS.VAL = ("CITY_CHALLENGE_val",)
    cfg.DATASETS.TEST = ("CITY_CHALLENGE_test",)

    # Hyper-params config
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.TEST.EVAL_PERIOD = 0
    cfg.SOLVER.BASE_LR = 0.001  # learning rate
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # batch size per image
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Raw model name
    model_name = MODEL_ID.replace('.yaml', '').split('/')[-1]   # Model name without .yml and COCO-Detection

    # --- RUN TRAINING ---
    if 'TRAIN' in ACTIONS:
        # Init wandb
        wandb.init(project="M6-week4", entity='celulaeucariota', name=model_name, sync_tensorboard=True)

        # --- TRAINING ---
        trainer = MyTrainer(cfg)  # Create object
        trainer.resume_or_load(resume=True)  # If the model has been already trained, load it
        trainer.train()  # Train

        # save the model
        os.makedirs('models', exist_ok=True)
        torch.save(trainer.model.state_dict(), os.path.join('models', f"{model_name}.pth"))

    # --- RUN EVALUATION ---
    if 'EVALUATE' in ACTIONS:
        # --- EVALUATION ---
        # Load the saved weights
        cfg.MODEL.WEIGHTS = os.path.join('models', model_name + ".pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold

        # Create the predictor
        predictor = DefaultPredictor(cfg)

        # Load the test dataset
        dataset_dicts = register_city_challenge('splits/test.txt')

        # Find the different cameras i.e. c010, c011, c012, ...
        # Create a dictionary in which the key is the camera name and the value is the mm.MOTAccumulator object of that camera
        cameras = {}

        # Obtain the name of the first camera and initialize the sort and accumulator
        camera_name = dataset_dicts[0]['file_name'].split('/')[2]
        tracker = Sort()
        accumulator = mm.MOTAccumulator(auto_id=True)

        for d in tqdm(dataset_dicts, desc='Tracking and its evaluation of the test set'):
            image_path = d['file_name']                         # Image path
            gt_ids = [an['id'] for an in d['annotations']]      # Ground truth ids
            gt_bboxes = [an['bbox'] for an in d['annotations']] # Ground truth boxes
            gt_centers = [(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) for bbox in gt_bboxes]

            # If the camera is not in the dictionary create a new entry and the tracker
            if image_path.split('/')[2] is not camera_name:
                cameras[camera_name] = accumulator
                tracker = Sort()
                accumulator = mm.MOTAccumulator(auto_id=True)

            camera_name = image_path.split('/')[2]
            im = cv2.imread(d["file_name"])                     # Read the image
            output = predictor(im)                              # Predictions

            pred_bboxes = output['instances'].pred_boxes.tensor.cpu().numpy()  # Predicted boxes
            pred_bboxes = [box.tolist() for box in pred_bboxes]  # Convert to list

            print(pred_bboxes)
            trackers = tracker.update(np.array(pred_bboxes))     # Update the tracker

            pred_centers = []
            pred_ids = []
            for t in trackers:
                pred_centers.append((int(t[0] + t[2] / 2), int(t[1] + t[3] / 2)))
                pred_ids.append(int(t[4]))

            accumulator.update(
                gt_ids,  # Ground truth objects in this frame
                pred_ids,  # Detector hypotheses in this frame
                mm.distances.norm2squared_matrix(gt_centers, pred_centers)
                # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
            )

        cameras[camera_name] = accumulator

        for cam_name in cameras:
            print(cam_name)
            mh = mm.metrics.create()
            print(mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc'))


