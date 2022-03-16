import matplotlib.pyplot as plt
import numpy as np

from dataset_gestions import load_labels, get_frames_paths
from metric_functions import evaluation_single_class
from noise_generator import noise_bboxes
from utils import plot_precision_recall_one_class, plot_iou_vs_time

"""
In this .py is implemented the following:
- Experiment the impact on the Average Precission metric when adding noise to the GT bounding boxes

"""

# If you want to add noise to the bounding boxes and delete some of them, set this variable to true:
add_noise = True

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
std_list = np.arange(0, 20, 0.2)
dropout_list = np.arange(0, 1, 0.05)
generate_list = np.arange(0, 1, 0.05)
ap_list = []
# To see if our method is robust, we set the detections = ground truth and plot results.
if add_noise:

    for gen in generate_list:
        detections_noisy = {}
        for d in ground_truth:
            detections_noisy[d] = noise_bboxes(ground_truth[d],mean = 0, std = 0, dropout = 0, generate = gen)

        recall, precision, ap = evaluation_single_class(ground_truth, frames_paths, detections_noisy)
        ap_list.append(ap)

    if plot_results:
        plt.plot(generate_list, ap_list)
        plt.title('Average precision over random generation')
        plt.xlabel('generation probability')
        plt.ylabel('AP')
        plt.show()
        # plot_precision_recall_one_class(recall, precision, ap, "Ground truth + noise")
        # plot_iou_vs_time("Ground truth + noise", frames_paths, ground_truth, detections_noisy)


