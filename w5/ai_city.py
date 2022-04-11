import os
import torch
import glob
import random
from os.path import join
import sys

from utilities.dataset_utils import to_yolo, load_annot, get_weights, write_yaml_file

import sys
sys.path.insert(0, "yolov5")

from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size
from yolov5.train import main as train_yolov5

class AICity:
    """
    Class for the AI City challenge. It contains the methods the data distribution and the methods to load data.
    """
    def __init__(
        self,
        data_path="../../data/AICity_data/train/",
        model="yolov5m",
        weights="",
        hyp='yolov5/data/hyps/hyp.VOC.yaml',
        epochs=200,
        batch_size=16,
        img_size=[640, 640],
        output_path="./",
        train_seq=["S01", "S04"],
        test_seq=["S03"],
        mode="train",
    ):
        """
        Initialize the AICity class
        :param data_path: root path to the data
        :param model: model to use 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
        :param output_path: path to the output
        :param train_seq: list of train sequences i.e. ['S01', 'S04']
        :param test_seq: list of test sequences i.e. ['S03']
        """

        self.data_path = data_path  # Root path to the database
        self.model = weights  # yolov5n, yolov5s, yolov5m, yolov5l, yolov5x

        self.output_path = output_path  # Output path

        # DETECTOR PARAMETERS
        self.det_params = dict(
            mode=mode,
            model=model,
            weights=weights,
            hyp=hyp,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            conf_thres=0.2,
            iou_thres=0.45,
            name=model + '_'.join(train_seq),
            #coco_model=args.coco_model
        )

        self.seq_train = train_seq  # List of train sequences
        self.seq_test = test_seq  # List of test sequences

        self.sequences = (
            {}
        )  # Dictionary with the sequences (key: sequence name, value: LoadSeq object)
        for seq in os.listdir(self.data_path):
            if "." not in seq[0]:
                self.sequences.update(
                    {seq: LoadSeq(self.data_path, seq, self.output_path)}
                )

    def data_to_yolo(self):
        """
        Convert the data to yolo format
        """
        save_path = 'data/yolo/'
        save_path = join(save_path, '-'.join(self.seq_train))
        os.makedirs(save_path, exist_ok=True)

        if len(os.listdir(save_path)) == 4:
            print("Yolo data already converted!")
            self.det_params.update({'data_yolo': join(save_path, 'cars.yaml')})
            return

        print("Preparing data for YOLOv5...")
        files_txt = {"train": [], "val": [], "test": []}
        for seq, sequence in self.sequences.items():
            # seq: name of the sequence (S01, S04, S03), sequence: LoadSeq object

            if seq in self.seq_train:
                [files_txt[split].append(paths)for split, paths in sequence.data_to_yolo(mode="train").items()]
            else:
                [files_txt[split].append(paths) for split, paths in sequence.data_to_yolo(mode='test').items()]

        yaml_dict = dict(
            nc=1,
            names=['car']
        )

        for split, paths in files_txt.items():
            paths = [path for cam in paths for path in cam]
            file_out = open(join(save_path, split + '.txt'), 'w')
            print(join(save_path, split + '.txt'))
            file_out.writelines(paths)
            yaml_dict.update({split: join(save_path, split + '.txt')})

        write_yaml_file(yaml_dict, join(save_path, 'cars.yaml'))
        self.det_params.update({'data_yolo': join(save_path, 'cars.yaml')})

        print('DONE!')

    def train(self):
        """
        Train the model
        """
        #print(vars(self.model_params))
        self.data_to_yolo()
        model = Ultralytics(weights=self.det_params["weights"], args=self.det_params)
        model.train()

class LoadSeq:
    """
    Class for the loading of a SINGLE SEQEUNCE (S01, S03, S04). It contains the methods to load the data and the methods
    to convert the data to yolo format.
    """
    def __init__(self, data_path, seq, output_path):
        """
        Initialize the LoadSeq class in charge of loading the data for a specific sequence
        :param data_path: root path to the data
        :param seq: name of the sequence (S01, S04, S03)
        :param output_path: path to the output
        """
        self.data_path = data_path
        self.seq = seq
        self.output_path = output_path
        # Load detections and load frame paths and filter by gt
        self.gt_bboxes = {}
        self.det_bboxes = {}
        self.frames_paths = {}
        self.tracker = {}
        self.mask = {}

        self.accumulators = {}

        for cam in os.listdir(join(data_path, seq)):
            # Save paths to frames
            cam_paths = glob.glob(join(data_path, seq, cam, "frames/*." + "jpg"))
            # cam_paths = [path for frame_id,_ in self.gt_bboxes[cam].items() for path in cam_paths if frame_id in path]
            cam_paths.sort()
            self.frames_paths.update({cam: cam_paths})
            # Load cam mask (roi)
            # self.mask.update({cam: dist_to_roi(join(data_path, seq, cam, 'roi.jpg'))})

            # Load gt
            self.gt_bboxes.update(
                {cam: load_annot(join(data_path, seq, cam), "gt/gt.txt")}
            )

        self.data = {"train": [], "val": [], "test": []}

    def train_val_split(self, split=0.25, mode="train"):
        """
        Apply split to specific proportion of the dataset. In other words, the training sequences will be split in
        train and val in order to train the model.
        :param split: proportion of the dataset to use for validation
        :param mode: mode to use 'train', 'test'
        """
        self.data = {"train": {}, "val": {}, "test": {}}
        if mode in "train":
            # Define cams used to train and validate
            cams = self.frames_paths.keys()
            cams_val = random.sample(cams, int(len(cams) * split))
            cams_train = list(set(cams) - set(cams_val))

            self.data.update({"train": dict(filter(lambda cam: cam[0] in cams_train, self.frames_paths.items()))})
            self.data.update({"val": dict(filter(lambda cam: cam[0] in cams_val, self.frames_paths.items()))})

        else:
            # The whole sequence used to test
            self.data.update({"test": self.frames_paths})

    def data_to_yolo(self, split=.25, mode='train'):
        self.train_val_split(split, mode)
        return to_yolo(self.data, self.gt_bboxes)

class Ultralytics():
    def __init__(self,
                 weights=None,
                 device='0',
                 agnostic_nms=False,
                 args=None):
        """
        Class Initialization for the models from Utralytics
        :param weight:
        :param device:
        :param agnostics_nms:
        :param args:
        """

        # If no weights are provided, use the default weights from the corresponding model
        # yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
        if weights is None:
            weights = args['weights']
        weights = get_weights(weights)

        # Class position for car
        classes = [0]

        if args['mode'] in 'inference' or args['mode'] in 'eval':
            self.device = select_device(device)

            # Load model
            self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
            self.img_size = check_img_size(args['img_size'][0], s=self.model.stride.max())  # check img_size

            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

            self.conf_thres = args['conf_thres']
            self.iou_thres = args['iou_thres']
            self.classes = classes
            self.agnostic_nms = agnostic_nms

            img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
            _ = self.model(img) if self.device.type != 'cpu' else None  # run once

        elif args['mode'] in 'train':
            self.weights = weights
            self.args = args

    def train(self):
        """
        Train the model
        """
        train_yolov5(self.weights, self.args)

if __name__ == "__main__":
    aic = AICity()
    aic.train()
