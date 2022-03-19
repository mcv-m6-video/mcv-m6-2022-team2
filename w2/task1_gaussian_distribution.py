import numpy as np

from dataset_gestions import get_frames_paths, load_labels
from background_estimation import single_gaussian_estimation
from metric_functions import evaluation_single_class
import os
import pickle

# Directions where all the sequence is located
path_data = '../../data/AICity_data/train/S03/c010'

# Direction where all the ground truth annotations are located
path_gt = path_data + '/gt'

# Direction where the video of the sequence is located
path_video = path_data + '/vdo'

# Load and update ground truth labels
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

# Create frames if not created and get its paths
frames_paths = get_frames_paths(path_video)

# drop the frames that have been used to estimate the model.
n_frames_modeling_bg = round(len(frames_paths) * 0.25)
keys = range(87, 87+n_frames_modeling_bg)
keys_groundTruth = []
for key in keys:
    if key < 100:
        keys_groundTruth = np.append(keys_groundTruth, '00' + str(key))
    else:
        keys_groundTruth = np.append(keys_groundTruth, '0' + str(key))

for key_groundTruth in keys_groundTruth:
    del ground_truth[key_groundTruth]
frames_paths_cropped = frames_paths[n_frames_modeling_bg:]

# Create variables dir where we will put variables to save computations
os.makedirs('variables', exist_ok=True)

alpha = [0.1, 0.2, 0.3]

for alpha_value in alpha:
    print(f'estimating bg with alpha: {alpha_value}...')
    # Estimates bg with gaussian estimation
    labels = single_gaussian_estimation(frames_paths, alpha=alpha_value)

    # Evaluate model
    recall, precision, ap = evaluation_single_class(ground_truth, frames_paths_cropped, labels)
    print(f'AP computed is: {round(ap, 4)}')