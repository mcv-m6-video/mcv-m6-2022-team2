from sort.sort import Sort
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataset_gestions import load_labels, get_frames_paths

path_data = '../../data/AICity_data/train/S03/c010'             # root of the data
path_gt = path_data + '/gt'                                     # ground truth folder
ai_city_path = '../../data/AICity_data/train/S03/c010/vdo'      # folder of the frames of the video


if __name__ == "__main__":

    # Obtain the ground truth annotations of the sequence
    ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')
    detections = load_labels('fine_tune', 'faster_rcnn_X_101_32x8d_FPN_3x.txt')

    # If the folder does not exist, create it. Then, return a list with the path of all the frames
    frames = get_frames_paths(ai_city_path)

    # Initialize tracker
    mot_tracker = Sort()

    frame_id = int(list(detections.keys())[0])
    ending_frame = int(list(detections.keys())[-1]) - 1
    print(f'Starting in the frame {frame_id} until {ending_frame}')

    for idx in range(frame_id, ending_frame):
        id_seen = []

        # Obtain the gt and detected bboxes of the current frame
        gt_bboxes = [detection['bbox'] for detection in ground_truth[f'{idx:04}']]
        det_bboxes = [detection['bbox'] for detection in detections[f'{idx:04}']]

        # Pass the frame detections to the tracker
        trackers = mot_tracker.update(np.array(det_bboxes))

        # Read the current frame
        frame = cv2.imread(frames[idx])

        # Draw the detected bboxes and track ids
        for t in trackers:
            cv2.rectangle(frame, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (0, 0, 255), 2)
            cv2.putText(frame, str(int(t[4])), (int(t[0]), int(t[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frames', frame)
        cv2.waitKey(1)