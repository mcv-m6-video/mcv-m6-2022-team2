from dataset_gestions import get_frames_paths
from background_estimation import single_gaussian_estimation
import os

# Directions where all the sequence is located
path_data = '../../data/AICity_data/train/S03/c010'

# Direction where the video of the sequence is located
path_video = path_data + '/vdo'

# Create variables dir where we will put variables to save computations
os.makedirs('variables', exist_ok=True)

# Create frames if not created and get its paths
frames_paths = get_frames_paths(path_video)

# Estimates bg with gaussian estimation
labels = single_gaussian_estimation(frames_paths)


