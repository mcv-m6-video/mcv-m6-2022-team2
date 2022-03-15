from dataset_gestions import load_labels, get_frames_paths
from metric_functions import evaluation_single_class
from utils import plot_precision_recall_one_class, plot_iou_vs_time

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
path_data = '../AICity_data/train/S03/c010'

# Direction where all the ground truth annotations are located
path_gt = path_data + '/gt'

# Direction where the video of the sequence is located
path_video = path_data + '/vdo'

# Load and update ground truth labels
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

# Before, generation of the frames from the video with ffmpeg is needed. Then, extract the paths of the frames.
frames_paths = get_frames_paths(path_video)

if add_noise:
    # TODO: create function that moves the bboxes and drops some of them (my idea was to do it in noise_generator.py)
    # TODO: evaluate results with evaluation_single_class function
    # TODO: plot precision recall and compare results (more noise, worse results...)
    print('esta por hacer')

else:
    # To see if our method is robust, we set the detections = ground truth and plot results.
    recall, precision, ap = evaluation_single_class(ground_truth, frames_paths, ground_truth)
    print('AP if there is not noise: ', round(ap, 2))

# Direction where all the detected annotations are located
det_path = path_data + '/det'

# Files with all the detections
det_file_paths = ['det_mask_rcnn.txt', 'det_ssd512.txt', 'det_yolo3.txt']

# Compute precision, recall and AP. Also compute IoU vs time (if plot_results is activated).
for det_file_path in det_file_paths:
    detections = load_labels(det_path, det_file_path)
    recall, precision, ap = evaluation_single_class(ground_truth, frames_paths, detections)
    print(f'AP for detection {det_file_path.split(".txt")[0]} is: {round(ap, 2)}')
    if plot_results:
        plot_precision_recall_one_class(recall, precision, ap, det_file_path.split(".txt")[0])
        plot_iou_vs_time(det_file_path, frames_paths, ground_truth, detections)