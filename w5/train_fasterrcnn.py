import os
import random
from glob import glob
from os.path import join
from utilities.image_utils import all_videos_to_frames

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

        self.seq_train = train_seq
        self.seq_test = test_seq



        # Create the video frames if not done before
        print('CAREFUL!!! The following function will store around 34000 files and 29.5GB of data in your computer.')
        all_videos_to_frames(data_root=self.data_path)

        # Create the folder in which all the output is stored
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/fasterrcnn', exist_ok=True)
        os.makedirs('data/fasterrcnn/')

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
        :param mode: train or test
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
        #TODO: implement this function
        """
        :param mode:
        :return:
        """
        return None

if __name__== "__main__":
    aic = AICity()
    aic.train_val_split()






