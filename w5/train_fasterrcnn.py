import os
import random
import cv2
import torch
import wandb
from glob import glob
from os.path import join, exists
from tqdm import tqdm

from utilities.image_utils import all_videos_to_frames
from utilities.json_loader import load_json, save_json
from utilities.dataset_utils import load_annot, write_predictions
from utilities.detectron_utils import MyTrainer

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor



class AICity:
    """
    Class for the AI City challenge. It contains the methods the data distribution and the methods to load data.
    """
    def __init__(
        self,
        data_path="../../data/AICity_data/train/",
        model_yaml="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        epochs=5000,
        batch_size=16,
        train_seq=["S01", "S04"],
        test_seq=["S03"],
    ):
        """
        Initialize the AICity class
        :param data_path: root path to the data ("../../data/AICity_data/train/" by default)
        :param model: model yaml file (for Faster R-CNN - ResNeXt101: COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml)
        :param epochs: number of epochs (5000 by default)
        :param batch_size: batch size (16 by default)
        :param train_seq: list of sequences to train on (["S01", "S04"] by default)
        :param test_seq: list of sequences to test on (["S03"] by default)
        """

        self.data_path = data_path
        self.model = model_yaml

        self.epochs = epochs
        self.batch_size = batch_size

        self.seq_train = train_seq
        self.seq_test = test_seq

        # Create the video frames if not done before
        print('CAREFUL!!! The following function will store around 34000 files and 29.5GB of data in your computer.')
        all_videos_to_frames(data_root=self.data_path)

        # Create the folder in which all the output is stored
        os.makedirs('data', exist_ok=True)
        os.makedirs(join('data', 'fasterrcnn'), exist_ok=True)
        os.makedirs(join('data', 'fasterrcnn', '-'.join(self.seq_train)), exist_ok=True)

        # --- DETECTRON CONFIGURATIONS ---
        # 1. Model configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(self.model))  # model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)  # Model
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # car

        # Raw model name
        self.model_name = self.model.replace('.yaml', '').split('/')[-1]  # Model name without .yml and COCO-Detection

        # 2. Dataset configuration
        self.cfg.DATASETS.TRAIN = ("AICity_train",)
        self.cfg.DATASETS.TEST = ("AICity_test",)

        # 3. Hyper-params configuration
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        self.cfg.MODEL.BACKBONE.FREEZE_AT = 2
        self.cfg.TEST.EVAL_PERIOD = 0
        self.cfg.SOLVER.BASE_LR = 0.001  # learning rate
        self.cfg.SOLVER.MAX_ITER = self.epochs
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # batch size per image
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.DEVICE = "cuda"
        self.cfg.OUTPUT_DIR = 'output'
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def train_val_split(self, split=0.2):
        """
        Split the training data into train and validation in a variable called self.data which has the following formart:
        ---------------------
        self.data = {
            "train": [
                {"S01": ['c001', ..., 'c005']},
                {"S04": ['c016', ..., 'c040']}
            ],
            "val": [
                {"S01": ['c003']},
                {"S04": ['c018', 'c022', 'c030']}
            ]
            ,
            "test": [
                {"S03": ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'}
            ]
        }
        ---------------------
        :param split: split ratio (0.2 by default)
        :return: list of train/val files
        """
        self.data = {"train": [], "val": [], "test": []}

        for seq in self.seq_train:
            cams = sorted(os.listdir(join(self.data_path, seq)))
            cams_val = random.sample(cams, int(len(cams) * split))
            cams_train = list(set(cams) - set(cams_val))
            self.data["train"].append({seq: cams_train})
            self.data["val"].append({seq: cams_val})

        for seq in self.seq_test:
            cams = sorted(os.listdir(join(self.data_path, seq)))
            self.data["test"].append({seq: cams})

    def register_dataset(self, mode):
        """
        Function to load the AI-city-challenge dataset as coco format for detectron2
        As the AI-city-challenge is very large, the first time the datasets are registered, the annotations are saved
        in a json file in the folder data/fasterrcnn/{S01-S03-S04}/dataset/{train, val, test}.json
        :param mode: str, train, val or test
        :return: list of annotations
            dataset_dicts = [{'file_name': str
                              'image_id': str
                              'height': int
                              'width': int
                              'annotations': [{'bbox': [x, y, x, y],
                                              'bbox_mode: BoxMode.XYXY_ABS,
                                              'category_id': 0,
                                              'id': int
                                              },
                                              ...
                                              ]
                              ...
                             }]
        """

        # Path of the json file
        json_path = join('data', 'fasterrcnn', '-'.join(self.seq_train), 'dataset', '{}.json'.format(mode))

        # If the json file exists, load it
        if exists(json_path):
            print(f"Loading dataset from {json_path}...")
            dataset_dicts = load_json(json_path)

        # If don't create it
        else:
            print(f"Creating {mode} dataset...")
            os.makedirs(join('data', 'fasterrcnn', '-'.join(self.seq_train), 'dataset'), exist_ok=True)

            dataset_dicts = []              # Dataset annotations in COCO format
            for seq in self.data[mode]:     # Obtain sequences corresponding to the mode (train, val, test)
                seq_name = list(seq.keys())[0]
                # Iterate through the cameras, e.g. c001, ..., c005 for S01 and c016, ..., c040 for S04
                for cam in seq[seq_name]:

                    # Load the ground truth annotations for the current camera
                    ground_truth = load_annot(join(self.data_path, seq_name, cam, 'gt'), 'gt.txt')

                    # Iterate through the frames of the current camera
                    for frame_path in sorted(glob(join(self.data_path, seq_name, cam, 'frames', '*.jpg'))):

                        # Obtain the frame number (0001, 0002, ..., 2174)
                        frame_num = frame_path.split('/')[-1].split('.')[0]

                        # The frame id consists in the camera name and the frame number
                        frame_id = cam + frame_num

                        # Height and Width of the frame
                        height, width, _ = cv2.imread(frame_path).shape

                        # All the detections of the current frame. To avoid crashes, when there are no detections
                        # in a frame, we use the .get() method which sets to an empty list when there are no frame
                        # annotations.
                        objs = []
                        frame_annot = ground_truth.get(frame_num, [])

                        # Iterate through the frame detections
                        for annot in frame_annot:
                            objs.append({
                                "bbox": annot['bbox'],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": 0,
                                "id": int(annot['obj_id']),
                            })

                        # Add the frame annotations to the dataset_dicts
                        dataset_dicts.append(
                            {"file_name": frame_path,
                             "image_id": frame_id,
                             "height": height,
                             "width": width,
                             "annotations": objs,
                             })

            save_json(json_path, dataset_dicts)

        return dataset_dicts

    def train_detectron2(self):
        """
        Train the model using Detectron2
        :return:
        """

        # --- PREPARE THE ENVIRONMENT ---

        # 1. Divide the dataset into train, val and test
        self.train_val_split(split=0)

        # 2. Register the dataset splits
        for mode in ['train', 'val', 'test']:
            DatasetCatalog.register("AICity_" + mode, lambda mode=mode: self.register_dataset(mode=mode))
            MetadataCatalog.get('AICity_' + mode).set(thing_classes=['car'])


        # --- TRAINING ---
        # 1. Init WandB
        wandb.init(project="M6-week5", entity='celulaeucariota', name=self.model_name, sync_tensorboard=True)

        # 2. Train the model
        trainer = DefaultTrainer(self.cfg)           # Create object
        trainer.resume_or_load(resume=True)     # If the model has been already trained, load it
        trainer.train()                         # Train

        # 3. Save the Model
        os.makedirs(join('data', 'fasterrcnn', '-'.join(self.seq_train), 'models'), exist_ok=True)
        torch.save(trainer.model.state_dict(), join('data', 'fasterrcnn', '-'.join(self.seq_train), 'models', f"{self.model_name}.pth"))

    def detection_inference(self):
        """
        Perform detection inference on the test set
        :return:
        """

        self.train_val_split(split=0)

        # Load the saved weights
        self.cfg.MODEL.WEIGHTS = join('data', 'fasterrcnn', '-'.join(self.seq_train), 'models', f"{self.model_name}.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold

        # Create the predictor
        predictor = DefaultPredictor(self.cfg)

        # Load the test dataset
        dataset_dicts = self.register_dataset(mode='test')

        # Divide the dataset into cameras (key: camera name, value: list of annotations)
        dataset_cams = {}
        for annot in dataset_dicts:
            if dataset_cams.get(annot['file_name'].split('/')[-3]) is None:
                dataset_cams[annot['file_name'].split('/')[-3]] = [annot]

            else: dataset_cams[annot['file_name'].split('/')[-3]].append(annot)

        # Create the folder in which the raw predictions will be saved
        os.makedirs(join('data', 'fasterrcnn', '-'.join(self.seq_train), 'predictions'), exist_ok=True)

        pbar = tqdm(desc='Generating predictions on the test set', total=len(dataset_dicts))

        # Iterate through the cameras to generate the predicted detections txt file
        for cam_name, dataset in zip(dataset_cams.keys(), dataset_cams.values()):

            annotations = []
            for annot in dataset:
                img_path = annot['file_name']
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

            write_predictions(join('data', 'fasterrcnn', '-'.join(self.seq_train), 'predictions', f"{cam_name}.txt"), annotations)

    def sc_tracking(self):
        # TODO: Implement Single-Camera Tracking algorithm
        """
        :return:
        """

        # Create folder to store the tracking results
        os.makedirs(join('data', 'fasterrcnn', '-'.join(self.seq_train), 'sc_tracking'), exist_ok=True)



if __name__== "__main__":
    aic = AICity(data_path="../../data/AICity_data/train/",
                 model_yaml="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
                 epochs=100,
                 batch_size=2,
                 train_seq=["S01", "S04"],
                 test_seq=["S03"],
                 )

    aic.detection_inference()






