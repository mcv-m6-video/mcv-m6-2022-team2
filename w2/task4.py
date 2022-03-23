import numpy as np

from dataset_gestions import get_frames_paths, load_labels
from background_estimation3D import single_gaussian_estimation
from metric_functions import evaluation_single_class
from utils import read_frames
import os
import pickle

path_data = '../../data/AICity_data/train/S03/c010' # Directions where all the sequence is located
path_gt = path_data + '/gt' # Direction where all the ground truth annotations are located
path_video = path_data + '/vdo' # Direction where the video of the sequence is located

frames_paths = get_frames_paths(path_video) # Extract frames from video and get its paths
# Load and update ground truth labels
alpha = 7
color_space="HSV"

   
print(f'estimating background with alpha: {alpha}...')
# Estimates bg with gaussian estimation
labels = single_gaussian_estimation(frames_paths, color_space=color_space,alpha=alpha)

# Drop the frames that have been used to estimate the model.
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

train_frames = round(len(frames_paths) * 0.25)

ground_truth_list = []
for idx in range(train_frames,len(frames_paths)):
    if f'{idx:04}' in ground_truth:
        ground_truth_list.append(ground_truth[f'{idx:04}'])
    else:
        ground_truth_list.append([])

# Evaluate model
recall, precision, ap = evaluation_single_class(ground_truth_list, labels, train_frames)
print(f'AP for alpha={alpha} and {color_space} color space computed is: {round(ap, 4)}')