import numpy as np

from dataset_gestions import get_frames_paths, load_labels
from background_estimation import single_gaussian_estimation
from metric_functions import evaluation_single_class
from utils import read_frames
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
frames_paths = get_frames_paths(path_video) # Extract frames from video and get its paths
frames = np.array(read_frames(frames_paths)) # Read all frames from the paths


# Drop the frames that have been used to estimate the model.
train_frames = round(frames.shape[0] * 0.25)

ground_truth_keys = list(ground_truth.keys())
ground_truth_keys.sort()

ground_truth_list = []
for key in ground_truth_keys[train_frames:]:
    ground_truth_list.append(ground_truth[key])

# Create variables dir where we will put variables to save computations
os.makedirs('variables', exist_ok=True)

# Estimates bg with gaussian estimation
labels = single_gaussian_estimation(frames, alpha=5, rho=0.5,adaptive=True)

# Evaluate model
recall, precision, ap = evaluation_single_class(ground_truth_list, labels, train_frames+1)
print(f'AP computed is: {round(ap, 4)}')