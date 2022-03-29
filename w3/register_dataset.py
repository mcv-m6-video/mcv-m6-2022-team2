import cv2
from tqdm import tqdm

from dataset_gestions import load_labels, get_frames_paths

from detectron2.structures import BoxMode

def register_city_challenge(frame_paths, gt):
    """
    Function to load the AI-city-challenge dataset as coco format
    :param frame_paths: List of img path
    :param gt: List of ground truth annotations
    :return: list of annotations
            dataset_dicts = [{'file_name': str
                              'image_id': str
                              'height': int
                              'width': int
                              'annotations': [{'bbox': [x, y, x, y],
                                              'bbox_mode: BoxMode.XYXY_ABS,
                                              'category_id': 0,
                                              },
                                              ...
                                              ]
                              ...
                             }]
    """
    dataset_dicts = []  # List of annotations
    for img_path in tqdm(frame_paths):
        record = {} # Annotations of one image

        img = cv2.imread(img_path)
        record["file_name"] = img_path
        record["image_id"] = img_path[-8:-4] # The image_id is a string of 4 numbers
        record['height'] = img.shape[0]
        record['width'] = img.shape[1]

        objs = []   # List of annotations of the current image

        # Iterate truth all the image objects
        for obj in gt[img_path[-8:-4]]:
            obj = {
                "bbox": obj['bbox'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

if __name__ == "__main__":
    path_data = '../../data/AICity_data/train/S03/c010'
    path_gt = path_data + '/gt'
    ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')
    ai_city_path = '../../data/AICity_data/train/S03/c010/vdo'
    frames = get_frames_paths(ai_city_path)

    d = register_city_challenge(frames, ground_truth)
    print()