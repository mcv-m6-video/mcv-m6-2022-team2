import cv2
import glob
import os
from tqdm import tqdm
from detectron2.structures import BoxMode

from dataset_utils import load_annot
from json_loader import saveData, loadData

def register_city_challenge(txt_path):
    """
    Function to load the AI-city-challenge dataset as coco format
    As the AI-city-challenge is very large, the first time the datasets are registered, the annotations are saved
    in a json file in the folder splits/AICity_{train,test}.json
    :param txt_path: path to the txt file containing the dataset
    The txt file contains the name of the sequences which correspond to the train/test set. It must have the
    following format:
    ----------------
    S01/c001/
    S01/c002/
    ...
    S01/c005/
    S04/c016/
    S04/c017/
    ...
    S04/c039/
    S04/c040/
    ----------------

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
    json_path = os.path.join('splits', f"AICity_{txt_path.split('.')[0].split('/')[1]}.json")

    # If the json file already exists, load it
    if os.path.exists(json_path):
        print(f"Loading dataset from {json_path}...")
        dataset_dicts = loadData(json_path)

    # If json file does not exist, create it
    else:
        dataset_dicts = []                      # List of annotations
        txt = open(txt_path, 'r')               # Open the text file which there are the sequences of the corresponding set
        txt_lines = txt.read().splitlines()     # Read all the lines of the text file

        # For each sequence
        for sequence_name in tqdm(txt_lines, desc=f"registering {txt_path.split('.')[0].split('/')[1]} set"):
            frame_paths = sorted(glob.glob(os.path.join('frames', sequence_name) + '/*.jpg'))   # Get the frames paths
            gt_detections = load_annot(os.path.join('frames', sequence_name), 'gt.txt')        # Get the ground truth

            for frame_path in frame_paths:         # For each frame
                # To avoid crashing when the frame does not contain annotations
                frame_num = frame_path.split('/')[-1].split('.')[0]
                frame_id = sequence_name + frame_num
                height, width, _ = cv2.imread(frame_path).shape

                # All the objects in the frame
                objs = []
                annotations = gt_detections.get(frame_num, [])

                # If there is at least one object in the frame
                if len(annotations) > 0:
                    for annotation in annotations:
                        objs.append({'bbox': annotation['bbox'],
                                     'bbox_mode': BoxMode.XYXY_ABS,
                                     'category_id': 0,
                                     'id': int(annotation['id'])
                                     })

                # Load the annotations for the current frame
                dataset_dicts.append({'file_name': frame_path,
                                      'image_id': frame_id,
                                      'height': height,
                                      'width': width,
                                      'annotations': objs,
                                      })

        # Save the annotations in a json file
        saveData(json_path, dataset_dicts)

    return dataset_dicts

if __name__ == "__main__":
    # Register the train set
    train_set = register_city_challenge('splits/train.txt')

    # Register the test set
    test_set = register_city_challenge('splits/test.txt')