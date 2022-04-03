import os
import wandb
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from utils.image_utils import read_txt_save_videos
from register_dataset import register_city_challenge
from utils.detectron_utils import MyTrainer

DDBB_ROOT =  '../../data/AICity_data/train/'
MODEL_ID = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" # Faster R-CNN (X101-FPN)
TRAIN = True

if __name__ == "__main__":

    # --- PREPARE ENVIRONMENT ---
    # Create the folder to store the frames individually
    print('CAREFUL!!! The following function will store around 34000 files and 29.5GB of data in your computer.')
    read_txt_save_videos(txt_path='splits/train.txt')
    read_txt_save_videos(txt_path='splits/test.txt')

    # Register the datasets
    for text_files, split in zip(['splits/train.txt', 'splits/test.txt'], ['train', 'test']):
        DatasetCatalog.register("CITY_CHALLENGE_" + split, lambda split=split: register_city_challenge(text_files))
        MetadataCatalog.get("CITY_CHALLENGE_" + split).set(thing_classes=["car"])

    # --- CONFIGURATIONS ---
    # Model config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ID))  # model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ID)  # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # car

    # Dataset config
    cfg.DATASETS.TRAIN = ("CITY_CHALLENGE_train",)
    cfg.DATASETS.VAL = ()
    cfg.DATASETS.TEST = ()

    # Hyper-params config
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    #cfg.MODEL.BACKBONE.FREEZE_AT = 2
    #cfg.TEST.EVAL_PERIOD = 0
    cfg.SOLVER.BASE_LR = 0.001  # learning rate
    cfg.SOLVER.MAX_ITER = 100
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # batch size per image
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Raw model name
    model_name = MODEL_ID.replace('.yaml', '').split('/')[-1]   # Model name without .yml and COCO-Detection

    # --- RUN TRAINING AND EVALUATION ---
    if TRAIN:
        # Init wandb
        wandb.init(project="M6-week4", entity='celulaeucariota', name=model_name, sync_tensorboard=True)

        # --- TRAINING ---
        trainer = DefaultTrainer(cfg)                # Create object
        trainer.resume_or_load(resume=False)    # If the model has been already trained, load it
        trainer.train()                         # Train

        # # save the model
        os.makedirs('models', exist_ok=True)
        torch.save(trainer.model.state_dict(), os.path.join('models', f"{model_name}.pth"))


    # --- RUN INFERENCE ---
    cfg.MODEL.WEIGHTS = os.path.join('models', model_name + '.pth')

    predictor = DefaultPredictor(cfg)

    # We can also evaluate its performance using AP metric implemented in COCO API.
    evaluator = COCOEvaluator('CITY_CHALLENGE_test', cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, 'CITY_CHALLENGE_test')
    print(inference_on_dataset(predictor.model, test_loader, evaluator))











