import matplotlib.pyplot as plt
import os
import numpy as np

from metric_functions import mean_iou

def plot_precision_recall_one_class(prec, recall, ap, info):
    """
    Plot precision and recall
    :param prec
    :param recall
    :param ap
    """
    plt.plot(prec, recall)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision vs recall in model {info} with AP: {ap}')
    plt.show()


def plot_iou_vs_time(det_file_path, frames_paths, ground_truth, detections):
    """
    Plot iou vs time
    (params are explained in other functions)
    :param det_file_path
    :param frames_paths
    :param ground_truth
    :param detections
    """
    miou = np.empty(0, )
    idFrames = np.array(())
    for frame in frames_paths:
        if os.name == 'nt':
            frame = frame.replace(os.sep, '/')
        frame_id = (frame.split('/')[-1]).split('.')[0]

        idFrames = np.append(idFrames, int(frame_id))
        gt_frame = np.array(dict_to_list(ground_truth[frame_id]))
        dets_frame = np.array(dict_to_list(detections[frame_id]))

        mean = mean_iou(gt_frame, dets_frame)
        miou = np.hstack((miou, mean))

    plt.plot(idFrames, miou)
    plt.ylim([0, 1])
    plt.xlabel('Frame')
    plt.ylabel('mean IoU')
    plt.title(f'Mean IoU in function of the frame in model {det_file_path.split(".txt")[0]}')
    plt.show()


def dict_to_list(frame_info):
    """
    Transform the bbox that is in the dictionary into a list
    :param frame_info: dictionary with the information needed to create the list
    :return: return the list created
    """
    return [[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]] for obj in frame_info]
