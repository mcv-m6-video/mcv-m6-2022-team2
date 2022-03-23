import numpy as np

from dataset_gestions import get_frames_paths, load_labels
from background_estimation import single_gaussian_estimation
from metric_functions import evaluation_single_class
from utils import read_frames
import os
import pickle

path_data = '../../data/AICity_data/train/S03/c010' # Directions where all the sequence is located
path_gt = path_data + '/gt' # Direction where all the ground truth annotations are located
path_video = path_data + '/vdo' # Direction where the video of the sequence is located

# Load and update ground truth labels
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

frames_paths = get_frames_paths(path_video) # Extract frames from video and get its paths
frames = np.array(read_frames(frames_paths)) # Read all frames from the paths

# Drop the frames that have been used to estimate the model.
train_frames = round(frames.shape[0] * 0.25)

ground_truth_list = []
for idx in range(train_frames,frames.shape[0]):
    if f'{idx:04}' in ground_truth:
        ground_truth_list.append(ground_truth[f'{idx:04}'])
    else:
        ground_truth_list.append([])
    
    

# Create variables dir where we will put variables to save computations
os.makedirs('variables', exist_ok=True)

alpha = [6.75]

for alpha_value in alpha:
    print(f'estimating background with alpha: {alpha_value}...')
    # Estimates bg with gaussian estimation
    labels = single_gaussian_estimation(frames, alpha=alpha_value)


    print(ground_truth_list)
    # Evaluate model
    recall, precision, ap = evaluation_single_class(ground_truth_list, labels, train_frames)
    print(f'AP computed is: {round(ap, 4)}')