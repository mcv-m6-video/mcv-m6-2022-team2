from dataset_gestions import load_labels, get_frames_paths
from generate_predictions import predict

"""
In this .py is implemented the following:
- Inference in detectron2 and AP computation
"""

# Directions where all the sequence is located
path_data = '../../data/AICity_data/train/S03/c010'

# Direction where all the ground truth annotations are located
path_gt = path_data + '/gt'

# Direction where the video of the sequence is located
path_video = path_data + '/vdo'

# Load and update ground truth labels
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

# Before, generation of the frames from the video with ffmpeg is needed. Then, extract the paths of the frames.
frames_paths = get_frames_paths(path_video)

# Model name to do inference in detectron2
model_name = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'

# Does prediction from the model and saves it in a txt
labels = predict(frames_paths, model_name, rewrite=True)

print('finished')
