from dataset_gestions import get_frames_paths, load_labels
from background_estimation import single_gaussian_estimation
from metric_functions import evaluation_single_class
import os

# Directions where all the sequence is located
path_data = '../../data/AICity_data/train/S03/c010'

# Direction where all the ground truth annotations are located
path_gt = path_data + '/gt'

# Direction where the video of the sequence is located
path_video = path_data + '/vdo'

# Load and update ground truth labels
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

# Create variables dir where we will put variables to save computations
os.makedirs('variables', exist_ok=True)

# Create frames if not created and get its paths
frames_paths = get_frames_paths(path_video)

# Estimates bg with gaussian estimation
labels = single_gaussian_estimation(frames_paths)

# todo: Eval model: first drop the frames that has been used to estimate the model. estoy en ello
n_frames_modeling_bg = round(len(frames_paths) * 0.25)
recall, precision, ap = evaluation_single_class(ground_truth, frames_paths, labels)
print(f'AP for detection ground_truth + noise is: {round(ap, 4)}')

# todo: calcular ap con evaluacion. sacar frames del principio!
