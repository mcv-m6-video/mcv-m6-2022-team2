import cv2
import matplotlib.pyplot as plt
import numpy as np
from background_estimation import preprocess_mask, foreground_bboxes
from utils import plotBBox, read_frames
from dataset_gestions import update_labels, load_labels, get_frames_paths
from metric_functions import evaluation_single_class

path_data = '../../data/AICity_data/train/S03/c010' # Directions where all the sequence is located
path_gt = path_data + '/gt' # Direction where all the ground truth annotations are located
path_video = path_data + '/vdo' # Direction where the video of the sequence is located

frames_paths = get_frames_paths(path_video) # Extract frames from video and get its paths
frames = np.array(read_frames(frames_paths)) # Read all frames from the paths

# Load and update ground truth labels
ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

# Drop the frames that have been used to estimate the model.
train_frames = 0

ground_truth_keys = list(ground_truth.keys())
ground_truth_keys.sort()

ground_truth_list = []
for key in ground_truth_keys[train_frames:]:
    ground_truth_list.append(ground_truth[key])

ALGORITHM = 'MOG2'

bg = cv2.createBackgroundSubtractorMOG2()

capture = cv2.VideoCapture(path_video + '.avi')
if not capture.isOpened():
    print('Unable to open')
    exit(0)

labels = {}

counter = 1
while True:
    print(counter)
    ret, frame = capture.read()

    if frame is None:
        break

    fgMask = bg.apply(frame)
    fgMask[fgMask!=255] = 0

    kernel = np.ones((5, 5), np.uint8)
    fgMask = preprocess_mask(fgMask)

    bboxes = foreground_bboxes(fgMask)
    #if not bboxes:
    #    labels = update_labels(labels, counter, 2, 2, 2, 2, 1)
    for bbox in bboxes:
        frame = plotBBox([frame], 0, 1, background=bboxes)[0]
        labels = update_labels(labels, counter, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], 1)

    #cv2.imshow('frame', frame)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    counter += 1

recall, precision, ap = evaluation_single_class(ground_truth_list, labels, 0)
print(f'AP computed is: {round(ap, 4)}')
