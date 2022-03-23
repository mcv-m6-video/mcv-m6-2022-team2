import numpy as np

from dataset_gestions import get_frames_paths, load_labels
from background_estimation import single_gaussian_estimation
from metric_functions import evaluation_single_class
from utils import plot_precision_recall_one_class, read_frames
import os
import pickle

path_data = '../../data/AICity_data/train/S03/c010' # Directions where all the sequence is located
path_gt = path_data + '/gt' # Direction where all the ground truth annotations are located
path_video = path_data + '/vdo' # Direction where the video of the sequence is located

frames_paths = get_frames_paths(path_video) # Extract frames from video and get its paths
np.array(read_frames(frames_paths,color=True)) # Read all frames from the paths

# Load and update ground truth labels
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt') 

alpha = 6.75


labels_channels = []
for c in range(3):
    print(c)
    with open(f'variables/frames_channel{c}.pickle', 'rb') as f:
        frames = pickle.load(f)
        
    print(f'estimating background with alpha: {alpha}...')
    # Estimates bg with gaussian estimation
    labels = single_gaussian_estimation(frames, alpha=alpha)
    
    labels_channels.append(labels)


labels_total = {}
for lc in labels_channels:
    for key in lc.keys():
        if key not in labels_total:
            labels_total[key] = lc[key]
        else:
            for l in lc[key]:
                labels_total[key].append(l)

# Drop the frames that have been used to estimate the model.
train_frames = round(frames.shape[0] * 0.99)

ground_truth_list = []
for idx in range(train_frames,frames.shape[0]):
    if f'{idx:04}' in ground_truth:
        ground_truth_list.append(ground_truth[f'{idx:04}'])
    else:
        ground_truth_list.append([])

# Evaluate model
recall, precision, ap = evaluation_single_class(ground_truth_list, labels_total, train_frames)
print(f'AP computed is: {round(ap, 4)}')