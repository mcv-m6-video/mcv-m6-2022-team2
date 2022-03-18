import cv2
import matplotlib.pyplot as plt
import os

from dataset_gestions import load_labels, get_frames_paths
from metric_functions import evaluation_single_class
from noise_generator import noise_bboxes
from utils import plot_precision_recall_one_class, plot_iou_vs_time, plotBBox

"""
In this .py is implemented the following:
- IoU and mAP for ground truth and noise
- mAP for the provided detections (mask_rcnn, ssd512, yolo3)
- iou vs time
"""

# If you want to add noise to the bounding boxes and delete some of them, set this variable to true:
add_noise = False

# If you want to plot graphics, set this variable to true:
plot_results = True

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

# To see if our method is robust, we set the detections = ground truth and plot results.
if add_noise:
    detections_noisy = {}
    for d in ground_truth:
        detections_noisy[d] = noise_bboxes(ground_truth[d],mean = 1, std = 0, dropout = 0.5, generate = 0.4)

    recall, precision, ap = evaluation_single_class(ground_truth, frames_paths, detections_noisy)
    print(f'AP for detection ground_truth + noise is: {round(ap, 4)}')
    if plot_results:
        plot_precision_recall_one_class(recall, precision, ap, "Ground truth + noise")
        plot_iou_vs_time("Ground truth + noise", frames_paths, ground_truth, detections_noisy)
else:
    recall, precision, ap = evaluation_single_class(ground_truth, frames_paths, ground_truth)
    print('AP if there is not noise: ', round(ap, 4))

# Direction where all the detected annotations are located
det_path = path_data + '/det'

# Files with all the detections
det_file_paths = ['det_mask_rcnn.txt', 'det_ssd512.txt', 'det_yolo3.txt']

# Compute precision, recall and AP. Also compute IoU vs time (if plot_results is activated).
for det_file_path in det_file_paths:
    detections = load_labels(det_path, det_file_path)
    recall, precision, ap = evaluation_single_class(ground_truth, frames_paths, detections)
    print(f'AP for detection {det_file_path.split(".txt")[0]} is: {round(ap, 4)}')

    frames = plotBBox(frames_paths, 800, 1000, ground_truth=ground_truth, detections=detections)
    
    if not os.path.exists(det_file_path.split(".txt")[0] + '_bboxes'):
        os.mkdir(det_file_path.split(".txt")[0] + '_bboxes')
        
        for i in range(len(frames)):
            if i < 1000:
                frame = '0' + str(i + 800) 
            else:
                frame = str(i + 800)

            cv2.imwrite(f'{det_file_path.split(".txt")[0]}_bboxes/bboxes_{frame}.png',frames[i])

    if plot_results:
        #plot_precision_recall_one_class(recall, precision, ap, det_file_path.split(".txt")[0])
        plot_iou_vs_time(det_file_path, frames_paths, ground_truth, detections)