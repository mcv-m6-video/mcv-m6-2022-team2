import glob
import os
import wandb
import torch
import cv2
import numpy as np
import motmetrics as mm
from tqdm import tqdm
from sort.sort import Sort
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.application_util import preprocessing

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from utils.image_utils import read_txt_save_videos, plotBBoxes
from register_dataset import register_city_challenge
from utils.detectron_utils import MyTrainer
from utils.dataset_gestions import write_predictions, load_labels

DDBB_ROOT =  '../../data/AICity_data/train/'
MODEL_ID = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" # Faster R-CNN (X101-FPN)

# Dictionary to chose what to do:
# If TRAIN in ACTIONS, the model will be trained
# If EVALUATE in ACTIONS, IDF1 will be calculated
# If INFER in ACTIONS, some gifs will be generated
# ACTIONS = ['TRAIN', 'EVALUATE', 'INFER']
ACTIONS = ['TRACK']

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


    # --- RUN INFERENCE ---
    if 'INFER' in ACTIONS:

        # Load the saved weights
        cfg.MODEL.WEIGHTS = os.path.join('models', model_name + ".pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold

        # Create the predictor
        predictor = DefaultPredictor(cfg)

        # Load the test dataset
        dataset_dicts = register_city_challenge('splits/test.txt')

        # Divide the dataset in cameras
        dataset_cams = {}
        for d in dataset_dicts:
            if dataset_cams.get(d['file_name'].split('/')[2]) is None:
                dataset_cams[d['file_name'].split('/')[2]] = [d]

            else: dataset_cams[d['file_name'].split('/')[2]].append(d)

        # Create the folder in which the predictions will be stored
        os.makedirs('inference', exist_ok=True)

        pbar = tqdm(desc='Generating predictions on the test set', total=len(dataset_dicts))
        # Iterate through the cameras to generate the detections txt file
        for cam_name, dataset in zip(dataset_cams.keys(), dataset_cams.values()):

            annotations = []
            for d in dataset:
                img_path = d['file_name']
                img = cv2.imread(img_path)
                frame_num = img_path.split('/')[-1].split('.')[0]

                output = predictor(img)

                pred_bboxes = output['instances'].pred_boxes.tensor.cpu().numpy()  # Predicted boxes
                pred_bboxes = [box.tolist() for box in pred_bboxes]  # Convert to list
                pred_scores = output['instances'].scores.cpu().numpy().tolist()  # Predicted boxes

                for bbox, score in zip(pred_bboxes, pred_scores):
                    x1, y1, x2, y2 = bbox
                    annotations.append([int(frame_num), -1, x1, y1, x2, y2, score])

                pbar.update(1)

            write_predictions(f"inference/{cam_name}.txt", annotations)

    # --- RUN INFERENCE ---
    if 'TRACK' in ACTIONS:

        txt = open('splits/test.txt', 'r')
        test_files = txt.read().splitlines()
        for file in test_files:
            print(file.split('/')[1])
            predicions = load_labels('inference', file.split('/')[1] + '.txt')
            ground_truth = load_labels(f'frames/{file}', 'gt.txt')

            # Create an accumulator that will be updated during each frame
            accumulator = mm.MOTAccumulator(auto_id=True)

            # Initialize tracker
            max_cosine_distance = 0.5
            nn_budget = None
            nms_max_overlap = 0.3

            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            tracker = Tracker(metric)

            for frame in sorted(glob.glob(f'frames/{file}/*.jpg')):
                frame_num = frame.split('/')[-1].split('.')[0]

                frame_gt = ground_truth.get(frame_num, [])
                frame_pred = predicions.get(frame_num, [])
                frame_pred = frame_gt
                # Obtain the gt and detected bboxes of the current frame
                gt_bboxes = [detection['bbox'] for detection in frame_gt]
                pred_bboxes = [Detection(detection['bbox'], 1.0, None) for detection in frame_pred]

                # Run non-maxima suppression on the detected bboxes
                boxes = np.array([d.tlwh for d in pred_bboxes])
                scores = np.array([d.confidence for d in pred_bboxes])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                pred_bboxes = [pred_bboxes[i] for i in indices]

                # # Obtain the centers of the ground truth detections
                # gt_centers = [(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) for bbox in gt_bboxes]
                # gt_ids = [detection['id'] for detection in frame_gt]

                print(frame_num, pred_bboxes)

                # Pass the frame detections to the tracker
                tracker.predict()
                tracker.update(pred_bboxes)

                # pred_centers = []
                # pred_ids = []
                # for track in tracker.tracks:
                #     if not track.is_confirmed() or track.time_since_update > 1:
                #         continue
                #     bbox = track.to_tlwh()
                #     pred_centers.append((bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2))
                #     pred_ids.append(track.track_id)



            #     accumulator.update(
            #         gt_ids,  # Ground truth objects in this frame
            #         pred_ids,  # Detector hypotheses in this frame
            #         mm.distances.norm2squared_matrix(gt_centers, pred_centers)
            #         # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
            #     )
            # mh = mm.metrics.create()
            # summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
            # print(summary)
            # Get the detections

        # # --- EVALUATION ---
        # # Load the saved weights
        # cfg.MODEL.WEIGHTS = os.path.join('models', model_name + ".pth")
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
        #
        # # Create the predictor
        # predictor = DefaultPredictor(cfg)
        #
        # # Load the test dataset
        # dataset_dicts = register_city_challenge('splits/test.txt')
        #
        # # Find the different cameras i.e. c010, c011, c012, ...
        # # Create a dictionary in which the key is the camera name and the value is the mm.MOTAccumulator object of that camera
        # cameras = {}
        #
        # # Obtain the name of the first camera and initialize the sort and accumulator
        # camera_name = dataset_dicts[0]['file_name'].split('/')[2]
        # tracker = Sort(min_hits=5)
        # accumulator = mm.MOTAccumulator(auto_id=True)
        #
        # for idx, d in enumerate(dataset_dicts):
        #     image_path = d['file_name']                         # Image path
        #     gt_ids = [an['id'] for an in d['annotations']]      # Ground truth ids
        #     gt_bboxes = [an['bbox'] for an in d['annotations']] # Ground truth boxes
        #     gt_centers = [(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) for bbox in gt_bboxes]
        #
        #     # If the camera is not in the dictionary create a new entry and the tracker
        #     if image_path.split('/')[2] is not camera_name:
        #         cameras[camera_name] = accumulator
        #         tracker = Sort(min_hits=5)
        #         accumulator = mm.MOTAccumulator(auto_id=True)
        #
        #     camera_name = image_path.split('/')[2]
        #     im = cv2.imread(d["file_name"])                     # Read the image
        #     output = predictor(im)                              # Predictions
        #
        #     pred_bboxes = output['instances'].pred_boxes.tensor.cpu().numpy()  # Predicted boxes
        #     pred_bboxes = [box.tolist() for box in pred_bboxes]  # Convert to list
        #
        #     pred_scores = output['instances'].scores.cpu().numpy().tolist()  # Predicted boxes
        #
        #     bboxes_score = []
        #     for bbox, score in zip(pred_bboxes, pred_scores):
        #         bbox.append(score)
        #         bboxes_score.append(bbox)
        #
        #     print(idx, np.array(bboxes_score))
        #     trackers = tracker.update(np.array(bboxes_score))     # Update the tracker
        #
        #     pred_centers = []
        #     pred_ids = []
        #     for t in trackers:
        #         pred_centers.append((int(t[0] + t[2] / 2), int(t[1] + t[3] / 2)))
        #         pred_ids.append(int(t[4]))
        #     plotBBoxes(im, saveFrames=None, pred_bboxes=pred_bboxes)
        #
        #     cv2.imshow('image', im)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        #     accumulator.update(
        #         gt_ids,  # Ground truth objects in this frame
        #         pred_ids,  # Detector hypotheses in this frame
        #         mm.distances.norm2squared_matrix(gt_centers, pred_centers)
        #         # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        #     )
        #
        # cameras[camera_name] = accumulator
        #
        # for cam_name in cameras:
        #     print(cam_name)
        #     mh = mm.metrics.create()
        #     print(mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc'))
        #
        #
