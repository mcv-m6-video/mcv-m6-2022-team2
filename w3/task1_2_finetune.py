from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import wandb
import torch
from tqdm import tqdm
import cv2

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from dataset_gestions import get_frames_paths, load_labels, update_labels, write_predictions
from register_dataset import register_city_challenge
from detectron_utils import MyTrainer

# MODEL YAML
model_id = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"         # RetinaNet (R101)
#model_id = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" # Faster R-CNN (X101-FPN)

# PATHS
path_data = '../../data/AICity_data/train/S03/c010'             # root of the data
path_gt = path_data + '/gt'                                     # ground truth folder
ai_city_path = '../../data/AICity_data/train/S03/c010/vdo'      # folder of the frames of the video

TRAIN = True

if __name__ == "__main__":

    model_name = model_id.replace('.yaml', '').split('/')[-1]   # Model name without .yml and COCO-Detection

    # --- PREPARATION ---
    # Obtain the ground truth annotations of the sequence
    ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

    # If the folder does not exist, create it. Then, return a list with the path of all the frames
    frames = get_frames_paths(ai_city_path)

    # Divide train and test sets (25% - 75%)
    train_frames = frames[:int(len(frames)*0.25)]
    test_frames = frames[int(len(frames)*0.25):]

    # val same as test since we are training and testing with the same sequence WTF
    for frames, set in zip([train_frames, test_frames, test_frames], ["train", "val", "test"]):
        DatasetCatalog.register("CITY_CHALLENGE_" + set, lambda set=set: register_city_challenge(frames, ground_truth))
        MetadataCatalog.get("CITY_CHALLENGE_" + set).set(thing_classes=["Car"])

    # --- CONFIGURATION ---
    # Model config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_id))    # model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)  # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # car

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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold used to filter out low-scored bounding boxes in predictions
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if TRAIN:
        # Init wandb
        wandb.init(project="detectron2-week3", entity='celulaeucariota', name=model_name, sync_tensorboard=True)

        # --- TRAINING ---
        trainer = MyTrainer(cfg)                # Create object
        trainer.resume_or_load(resume=True)    # If the model has been already trained, load it
        trainer.train()                         # Train

        # # save the model
        os.makedirs('models', exist_ok=True)
        torch.save(trainer.model.state_dict(), os.path.join('models', f"{model_name}.pth"))

        # --- INFERENCE/EVALUATION ---
        # Load the saved weights
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)

        # We can also evaluate its performance using AP metric implemented in COCO API.
        evaluator = COCOEvaluator('CITY_CHALLENGE_val', cfg, False, output_dir=cfg.OUTPUT_DIR)
        test_loader = build_detection_test_loader(cfg, 'CITY_CHALLENGE_val')
        print(inference_on_dataset(trainer.model, test_loader, evaluator))

    # --- RUN ONLY INFERENCE ---
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)

        predicor = DefaultPredictor(cfg)

        # We can also evaluate its performance using AP metric implemented in COCO API.
        evaluator = COCOEvaluator('CITY_CHALLENGE_val', cfg, False, output_dir=cfg.OUTPUT_DIR)
        test_loader = build_detection_test_loader(cfg, 'CITY_CHALLENGE_val')
        print(inference_on_dataset(predicor.model, test_loader, evaluator))

    # --- SAVE PREDICTIONS ---
    cfg.MODEL.WEIGHTS = os.path.join('models', f"{model_name}.pth")
    predictor = DefaultPredictor(cfg)

    labels = {}
    print('Generating predictions...')
    for frame_path in tqdm(test_frames):
        im = cv2.imread(frame_path)
        imgname = (frame_path.split('/')[-1]).split('.')[0]

        # Do inference and get the bboxes, confidence and classes
        outputs = predictor(im)
        bboxes = outputs["instances"].pred_boxes
        conf = outputs["instances"].scores
        classes = outputs["instances"].pred_classes

        # For each frame upload the labels dictionary
        for bbox, conf, pred_class in zip(bboxes, conf, classes):
            score = conf.cpu().numpy()
            bbox_det = bbox.cpu().numpy()
            update_labels(labels, imgname, -1, bbox_det[0], bbox_det[1], bbox_det[2] - bbox_det[0],
                          bbox_det[3] - bbox_det[1], score)

    print('Labels uploaded')
    # save predictions in the txt if rewrite=True of it not exists
    path = 'fine_tune'
    write_predictions(path, labels, model_name)




