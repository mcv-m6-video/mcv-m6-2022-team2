import os
import cv2
import glob
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog

from dataset_gestions import get_frames_paths, load_labels, update_labels, write_predictions
from register_dataset import register_city_challenge

# MODEL YAML
# model_id = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"         # RetinaNet (R101)
model_id = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" # Faster R-CNN (X101-FPN)

# PATHS
path_data = '../../data/AICity_data/train/S03/c010'             # root of the data
path_gt = path_data + '/gt'                                     # ground truth folder
ai_city_path = '../../data/AICity_data/train/S03/c010/vdo'      # folder of the frames of the video

if __name__ == "__main__":

    # --- PREPARATION ---
    # Obtain the ground truth annotations of the sequence
    ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

    # If the folder does not exist, create it. Then, return a list with the path of all the frames
    frames = get_frames_paths(ai_city_path)

    # Divide train and test sets (25% - 75%)
    train_frames = frames[:int(len(frames)*0.25)]
    test_frames = frames[int(len(frames)*0.25):]

    for frames, set in zip([train_frames, test_frames], ["train", "test"]):
        DatasetCatalog.register("CITY_CHALLENGE_" + set, lambda set=set: register_city_challenge(frames, ground_truth))
        MetadataCatalog.get("CITY_CHALLENGE_" + set).set(thing_classes=["car"])

    # --- CONFIGURATION ---
    # Model config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_id))    # model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)  # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # car

    # Dataset config
    cfg.DATASETS.TRAIN = ("CITY_CHALLENGE_train",)
    cfg.DATASETS.TEST = ("CITY_CHALLENGE_test",)

    # Hyper-params config
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001  # learning rate
    cfg.SOLVER.MAX_ITER = 10
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # batch size per image
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold used to filter out low-scored bounding boxes in predictions
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- TRAINING ---
    trainer = DefaultTrainer(cfg)       # Create object
    trainer.resume_or_load(resume=True) # If the model has been already trained, load it
    trainer.train()                     # Train

    # save the model
    os.makedirs('models', exist_ok=True)
    torch.save(trainer.model.state_dict(), os.path.join('models', 'faster_rcnn_X_101.pth.tar'))

    # --- INFERENCE/EVALUATION ---
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)   # Predictor

    labels = {}
    for img_path in tqdm(test_frames, desc='Inference'):
        im = cv2.imread(img_path)
        imgname = (img_path.split('/')[-1]).split('.')[0]

        outputs = predictor(im[:,:,::-1])
        bboxes = outputs["instances"].pred_boxes
        conf = outputs["instances"].scores
        classes = outputs["instances"].pred_classes

        for bbox, conf, pred_class in zip(bboxes, conf, classes):
            score = conf.cpu().numpy()
            bbox_det = bbox.cpu().numpy()
            labels = update_labels(labels, imgname, bbox_det[0], bbox_det[1], bbox_det[2], bbox_det[3], score)

    # --- PREDICTION ---

    model_name = model_id.replace('.yaml', '').split('/')[-1]
    os.makedirs('fine_tune', exist_ok=True)

    with open(os.path.join('fine_tune', f'{model_name}.txt'), 'w') as file:
        for label in labels.items():
            for detection in label[1]:
                bbox = detection['bbox']
                conf = detection['confidence']
                # frame_id, id_detection (-1 bc only cars), bboxes, conf, x, y, z
                file.write(f'{int(label[0]) + 1},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{conf},-1,-1,-1\n')