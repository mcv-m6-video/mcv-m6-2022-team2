from dataset_gestions import load_labels, get_frames_paths
from generate_predictions import predict
from metric_functions import evaluation_single_class
from utils import plotBBox
import cv2

"""
In this .py is implemented the following:
- Inference in detectron2 and AP computation
"""
# Model name to do inference in detectron2
model_name = 'retinanet_R_101_FPN_3x.yaml'
# retinanet_R_101_FPN_3x.yaml
# faster_rcnn_X_101_32x8d_FPN_3x.yaml
# mask_rcnn_X_101_32x8d_FPN_3x.yaml

# Set this variable to true to draw the bounding boxes
draw = True

# Directions where all the sequence is located
path_data = '../../data/AICity_data/train/S03/c010'

# Direction where all the ground truth annotations are located
path_gt = path_data + '/gt'

# Direction where the video of the sequence is located
path_video = path_data + '/vdo'

# Load and update ground truth labels
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

# Path to detections
path_detections = 'off_the_shelve'

# Before, generation of the frames from the video with ffmpeg is needed. Then, extract the paths of the frames.
frames_paths = get_frames_paths(path_video)

# Does prediction from the model and saves them in a txt if it does not exist.
# If it exists and you want to rewrite it, set rewrite parameter to True.
predict(frames_paths, model_name, rewrite=False)

# Load labels detected from the txt
real_model_name = model_name.replace('.yaml', '')
detections = load_labels(path_detections, real_model_name + '.txt')
#detections2 = load_labels('/home/francesc/PycharmProjects/Visual-Recognition/M6/data/AICity_data/train/S03/c010/det', 'det_mask_rcnn.txt')

# Evaluate results
rec, prec, ap = evaluation_single_class(ground_truth, frames_paths, detections, class_name='car', iou_threshold=0.5)
print(f'AP: {ap}')


if draw:
    labels = {
        'gt' : ground_truth,
        'det': detections
    }

    frames = plotBBox(frames_paths, initalFrame=500, finalFrame=600, modelName=real_model_name,
                      saveFrames=True, **labels)
